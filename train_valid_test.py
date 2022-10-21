import imp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchaudio

from torch.utils.tensorboard import SummaryWriter
from basic import Count

from core.registry import CONFIG

from models.torch.dfsmn import DfsmnModel
from models.torch.bidfsmn import BiDfsmnModel, BiDfsmnModel_thinnable, DfsmnModel_pre

from speech_commands.dataset.speech_commands import SpeechCommandV1
from speech_commands.dataset.transform import ChangeAmplitude, \
    FixAudioLength, ChangeSpeedAndPitchAudio, TimeshiftAudio

from torch_utils import mixup

from pytorch_wavelets import DWTForward, DWTInverse

def loss_term(A):
    a = torch.abs(A)
    Q = a * a
    return Q

def total_loss(Q_s, Q_t):
    Q_s = loss_term(Q_s)
    Q_t = loss_term(Q_t)
    Q_s_norm = Q_s / torch.norm(Q_s, p=2)
    Q_t_norm = Q_t / torch.norm(Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = torch.norm(tmp, p=2)
    return loss

def pass_filter(x, select_pass, J=1, wave='haar', mode='zero'):
    xfm = DWTForward(J=J, mode=mode, wave=wave) # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode=mode, wave=wave)
    if x.is_cuda:
        xfm, ifm = xfm.cuda(), ifm.cuda()

    if len(x.shape) == 3:
        yl, yh = xfm(x.unsqueeze(1))
    elif len(x.shape) == 4:
        yl, yh = xfm(x)
    else:
        assert(False) # error

    if select_pass == 'high':
        yl.zero_()
    
    y = ifm((yl, yh))
    if len(x.shape) == 3:
        y = y.squeeze(1)
    return y


def get_model2(model_type: str, in_channels=1, **kwargs):
    if model_type == 'Vgg19Bn':
        return Vgg19BN(in_channels=in_channels, **kwargs) # [Batch, 1, 32, 32]
    elif model_type == 'Mobilenetv1':
        return MobileNetV1(in_channels=in_channels, **kwargs)
    elif model_type == 'Mobilenetv2':
        return MobileNetV2(in_channels=in_channels, **kwargs)
    elif model_type == 'BCResNet':
        return BCResNet(in_channels=in_channels, **kwargs) # [Batch, 1, 40, 32]
    elif model_type == 'fsmn':
        return FSMN(in_channels=in_channels, **kwargs)
    elif model_type == 'Dfsmn':
        return DfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn':
        return BiDfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable_pre':
        return DfsmnModel_pre(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable':
        return BiDfsmnModel_thinnable(in_channels=in_channels, **kwargs)
    else:
        raise RuntimeError('unsupport model type: ', model_type)
    

def get_model(model_type: str, in_channels=1, method="no", **kwargs):
    if method == "no":
        model = get_model2(model_type, in_channels, **kwargs)
        return model
    else:
        from basic import Count, Modify
        model = get_model2(model_type, in_channels, **kwargs)
        model.method = method
        cnt = Count(model)
        model, _ = Modify(model, method=method, id=0, first=1, last=cnt)
        return model


def create_dataloader(dataset_type, configs, use_gpu, version):
    train_transform = Compose([
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        TimeshiftAudio(),
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=configs.n_mels,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])
    valid_transform = Compose([
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=configs.n_mels,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])

    dataset_train = SpeechCommandV1(configs.dataroot,
                                    subset='training',
                                    download=True,
                                    transform=train_transform,
                                    num_classes=configs.num_classes,
                                    noise_ratio=0.3,
                                    noise_max_scale=0.3,
                                    cache_origin_data=False,
                                    version=version)

    dataset_valid = SpeechCommandV1(configs.dataroot,
                                    subset='validation',
                                    download=True,
                                    transform=valid_transform,
                                    num_classes=configs.num_classes,
                                    cache_origin_data=True,
                                    version=version)

    dataset_test = SpeechCommandV1(configs.dataroot,
                                   subset='testing',
                                   download=True,
                                   transform=valid_transform,
                                   num_classes=configs.num_classes,
                                   cache_origin_data=True,
                                    version=version)

    dataset_dict = {
        'training': dataset_train,
        'validation': dataset_valid,
        'testing': dataset_test
    }
    return DataLoader(dataset_dict[dataset_type],
                      batch_size=configs.batch_size,
                      shuffle=dataset_type == 'training',
                      sampler=None,
                      pin_memory=use_gpu,
                      num_workers=16,
                      persistent_workers=True)


def create_lr_schedule(configs, optimizer):
    if configs.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=configs.lr_scheduler_patience,
            factor=configs.lr_scheduler_gamma)
    elif configs.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=configs.lr_scheduler_stepsize,
            gamma=configs.lr_scheduler_gamma,
            last_epoch=configs.epoch - 1)
    elif configs.lr_scheduler == 'cosin':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.epoch)
    else:
        raise RuntimeError('unsupported lr schedule type: ',
                           configs.lr_scheduler)
    return lr_scheduler


def create_optimizer(configs, model):
    if configs.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.lr,
                                    momentum=0.9,
                                    weight_decay=configs.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs.lr,
                                     weight_decay=configs.weight_decay)

    return optimizer


weights = [1, 0.5, 0.25]
loss_lim = 50.0
distillation_pred = torch.nn.MSELoss()
pred = False

def train_epoch_distill(model: nn.Module,
                teacher_model: nn.Module,
                optimizer,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10,
                writer: SummaryWriter = None,
                mixup_alpha=0,
                distill_alpha=0,
                select_pass='no',
                J=1,
                num_classes=None):
    """
    training one epoch
    """
    model.train()
    if with_gpu:
        model = model.cuda()

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)

    
    running_loss = 0
    i = 0
    for inputs, target in pbar:
        if with_gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        if 0 < mixup_alpha < 1:
            inputs, target = mixup.mixup(inputs, target,
                                       np.random.beta(mixup_alpha, mixup_alpha),
                                       num_classes)
        # forward
        teacher_out, teacher_feature = teacher_model(inputs)
        if select_pass != 'no':
            teacher_feature = [f1 / torch.std(f1) + f2 / torch.std(f2) for f1, f2 in [(pass_filter(f, select_pass=select_pass, J=J), f) for f in teacher_feature]]
        
        loss = 0

        if model.__class__.__name__[-9:] != 'thinnable':
            out, feature = model(inputs)
            
            if 0 < mixup_alpha < 1:
                loss_one_hot = mixup.naive_cross_entropy_loss(out, target)
            else:
                loss_one_hot = criterion(out, target)

            if hasattr(model, 'method') and model.method == 'Laq':
                distr_loss1, distr_loss2 = model.laq_loss(inputs)
                distr_loss1 = distr_loss1.mean()
                distr_loss2 = distr_loss2.mean()
                # remove distrloss after args.distr_epoch epochs
                if epoch < 100:
                    loss = loss + (distr_loss1 + distr_loss2)

            loss = loss + loss_one_hot

            if len(teacher_feature) % len(feature) == 0:
                loss_distill = None
                for k in range(len(feature)):
                    j = int((len(teacher_feature) / len(feature)) * (k+1) - 1)
                    if loss_distill == None:
                        # loss_distill = distillation(feature[j] / torch.std(feature[j]), teacher_feature[k] / torch.std(teacher_feature[k]))
                        loss_distill = total_loss(feature[k], teacher_feature[j])
                    else:
                        # loss_distill += distillation(feature[j] / torch.std(feature[j]), teacher_feature[k] / torch.std(teacher_feature[k]))
                        loss_distill = loss_distill + total_loss(feature[k], teacher_feature[j])
                loss = loss + loss_distill * distill_alpha
                if pred:
                    loss_pred = distillation_pred(out, teacher_out)
                    loss = loss + loss_pred * distill_alpha
            else:
                print ('Distiilation Error: teacher {}, student {}!'.format(len(teacher_feature), len(feature)))
        else:
            for op in range(model.thin_n):
                weight = weights[op]
                out, feature = model(inputs, op)

                if 0 < mixup_alpha < 1:
                    loss_one_hot = mixup.naive_cross_entropy_loss(out, target)
                else:
                    loss_one_hot = criterion(out, target)
                loss = loss + loss_one_hot * weight

                if len(teacher_feature) % len(feature) == 0:
                    loss_distill = None
                    for k in range(len(feature)):
                        j = int((len(teacher_feature) / len(feature)) * (k+1) - 1)
                        if loss_distill == None:
                            # loss_distill = distillation(feature[j] / torch.std(feature[j]), teacher_feature[k] / torch.std(teacher_feature[k]))
                            loss_distill = total_loss(feature[k], teacher_feature[j])
                        else:
                            # loss_distill += distillation(feature[j] / torch.std(feature[j]), teacher_feature[k] / torch.std(teacher_feature[k]))
                            loss_distill = loss_distill + total_loss(feature[k], teacher_feature[j])
                    loss = loss + loss_distill * distill_alpha * weight
                    if pred:
                        loss_pred = distillation_pred(out, teacher_out)
                        loss = loss + loss_pred * distill_alpha * weight
                else:
                    print ('Distiilation Error: teacher {}, student {}!'.format(len(teacher_feature), len(feature)))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        # if loss.item() > loss_lim:
        #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        #     print('[loss ont_hot]: %.4f, [loss distill]: %.4f' % (loss_one_hot, loss_distill))
        optimizer.step()
        running_loss += loss.item()
        if i % log_iter == 0 and writer is not None:
            writer.add_scalar('Train/iter_loss', loss.item(),
                              i + epoch * epoch_size)
            writer.file_writer.flush()

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (loss.item()),
        })
        i += 1

    running_loss /= i
    if writer is not None:
        writer.add_scalar('Train/epoch_loss', running_loss, epoch)
        writer.file_writer.flush()

    return running_loss

def train_epoch(model: nn.Module,
                optimizer,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10,
                writer: SummaryWriter = None,
                mixup_alpha=0,
                num_classes=None):
    """
    training one epoch
    """
    model.train()
    if with_gpu:
        model = model.cuda()

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)

    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        i = 0
        for feat, target in pbar:
            if with_gpu:
                feat = feat.cuda()
                target = target.cuda()

            if 0 < mixup_alpha < 1:
                feat, target = mixup.mixup(feat, target,
                                        np.random.beta(mixup_alpha, mixup_alpha),
                                        num_classes)
            # forward
            out = model(feat)
            if 0 < mixup_alpha < 1:
                loss = mixup.naive_cross_entropy_loss(out, target)
            else:
                loss = criterion(out, target)
            
            if hasattr(model, 'method') and model.method == 'Laq':
                distr_loss1, distr_loss2 = model.laq_loss(feat)
                distr_loss1 = distr_loss1.mean()
                distr_loss2 = distr_loss2.mean()
                # remove distrloss after args.distr_epoch epochs
                if epoch < 100:
                    loss = loss + (distr_loss1 + distr_loss2)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % log_iter == 0 and writer is not None:
                writer.add_scalar('Train/iter_loss', loss.item(),
                                i + epoch * epoch_size)
                writer.file_writer.flush()

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (loss.item()),
            })
            i += 1

        running_loss /= i
        if writer is not None:
            writer.add_scalar('Train/epoch_loss', running_loss, epoch)
            writer.file_writer.flush()

        return running_loss
    else:
        thin_n = model.thin_n
        running_loss = 0
        i = 0
        for inputs, target in pbar:
            if with_gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            if 0 < mixup_alpha < 1:
                inputs, target = mixup.mixup(inputs, target,
                                        np.random.beta(mixup_alpha, mixup_alpha),
                                        num_classes)
            
            loss = 0
            
            # forward
            for op in range(thin_n):
                weight = weights[op]
                out = model(inputs, op)
                if 0 < mixup_alpha < 1:
                    loss += mixup.naive_cross_entropy_loss(out, target) * weight
                else:
                    loss += criterion(out, target) * weight
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            # if loss.item() > loss_lim:
            #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            #     print('[loss ont_hot]: %.4f, [loss distill]: %.4f' % (loss_one_hot, loss_distill))
            optimizer.step()
            running_loss += loss.item()
            
            if i % log_iter == 0 and writer is not None:
                writer.add_scalar('Train/iter_loss', loss.item(),
                                i + epoch * epoch_size)
                writer.file_writer.flush()

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (loss.item()),
            })
            i += 1

        running_loss /= i
        if writer is not None:
            writer.add_scalar('Train/epoch_loss', running_loss, epoch)
            writer.file_writer.flush()

        return running_loss


def valid_epoch_distill(model: nn.Module,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10,
                writer: SummaryWriter = None):
    """
    valid on dataset
    """
    model.eval()
    if with_gpu:
        model = model.cuda()

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)

    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        running_acc = 0
        i = 0
        with torch.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda()
                    target = target.cuda()
                # forward
                out, feature = model(feat)
                loss = criterion(out, target)

                pred = out.max(1, keepdim=True)[1]
                acc = pred.eq(target.view_as(pred)).sum() / target.size(0)

                running_loss += loss.item()
                running_acc += acc.item()

                # log per 10 iter
                if i % log_iter == 0 and writer is not None:
                    writer.add_scalar('Valid/iter_loss', loss.item(),
                                    i + epoch * epoch_size)
                    writer.file_writer.flush()

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (loss.item()),
                })
                i += 1

        running_acc /= i
        running_loss /= i

        # log for tensorboard
        if writer is not None:
            writer.add_scalar('Valid/epoch_loss', running_loss, epoch)
            writer.add_scalar('Valid/epoch_accuracy', running_acc, epoch)
            writer.file_writer.flush()

        return running_loss, running_acc
    else:
        thin_n = model.thin_n
        running_loss = 0.0
        running_acc = [0 for op in range(thin_n)]
        i = 0
        with torch.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda()
                    target = target.cuda()
                # forward
                for op in range(thin_n):
                    out, feature = model(feat, op)
                    loss = criterion(out, target)

                    pred = out.max(1, keepdim=True)[1]
                    acc = pred.eq(target.view_as(pred)).sum() / target.size(0)

                    running_loss += loss.item()
                    running_acc[op] += acc.item()

                    # log per 10 iter
                    if i % log_iter == 0 and writer is not None:
                        writer.add_scalar('Valid/iter_loss[%d]' % [8, 4, 2, 1][op], loss.item(),
                                        i + epoch * epoch_size)
                        writer.file_writer.flush()

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (loss.item()),
                })
                i += 1

        running_acc = [acc / i for acc in running_acc]
        running_loss = running_loss / i

        # log for tensorboard
        if writer is not None:
            writer.add_scalar('Valid/epoch_loss', running_loss, epoch)
            for op in range(thin_n):
                writer.add_scalar('Valid/epoch_accuracy_%d' % [8, 4, 2, 1][op], running_acc[op], epoch)
            writer.file_writer.flush()

        return running_loss, running_acc

def valid_epoch(model: nn.Module,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10,
                writer: SummaryWriter = None):
    """
    valid on dataset
    """
    model.eval()
    if with_gpu:
        model = model.cuda()

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)

    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        running_acc = 0
        i = 0
        with torch.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda()
                    target = target.cuda()
                # forward
                out = model(feat)
                loss = criterion(out, target)

                pred = out.max(1, keepdim=True)[1]
                acc = pred.eq(target.view_as(pred)).sum() / target.size(0)

                running_loss += loss.item()
                running_acc += acc.item()

                # log per 10 iter
                if i % log_iter == 0 and writer is not None:
                    writer.add_scalar('Valid/iter_loss', loss.item(),
                                    i + epoch * epoch_size)
                    writer.file_writer.flush()

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (loss.item()),
                })
                i += 1

        running_acc /= i
        running_loss /= i

        # log for tensorboard
        if writer is not None:
            writer.add_scalar('Valid/epoch_loss', running_loss, epoch)
            writer.add_scalar('Valid/epoch_accuracy', running_acc, epoch)
            writer.file_writer.flush()

        return running_loss, running_acc
    else:
        thin_n = model.thin_n
        running_loss = 0
        running_acc = [0 for op in range(thin_n)]
        i = 0
        with torch.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda()
                    target = target.cuda()
                # forward
                for op in range(thin_n):
                    out = model(feat, op)
                    loss = criterion(out, target)

                    pred = out.max(1, keepdim=True)[1]
                    acc = pred.eq(target.view_as(pred)).sum() / target.size(0)

                    running_loss += loss.item()
                    running_acc[op] += acc.item()

                    # log per 10 iter
                    if i % log_iter == 0 and writer is not None:
                        writer.add_scalar('Valid/iter_loss[%d]' % [8, 4, 2, 1][op], loss.item(),
                                        i + epoch * epoch_size)
                        writer.file_writer.flush()

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (loss.item()),
                })
                i += 1

        running_acc = [acc / i for acc in running_acc]
        running_loss = running_loss / i

        # log for tensorboard
        if writer is not None:
            writer.add_scalar('Valid/epoch_loss', running_loss, epoch)
            for op in range(thin_n):
                writer.add_scalar('Valid/epoch_accuracy_%d' % [8, 4, 2, 1][op], running_acc[op], epoch)
            writer.file_writer.flush()

        return running_loss, running_acc
