import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import modules, Parameter
from torch.autograd import Function

activations = {
    'ReLU': nn.ReLU,
    'Hardtanh': nn.Hardtanh
}

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input

class BinaryQuantize_Vanilla(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        if scale != None:
            out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input, None

class BiLinearVanilla(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BiLinearVanilla, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None

    def forward(self, input):
        bw = self.weight
        ba = input
        sw = bw.abs().mean(-1).view(-1, 1).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw, self.bias)
        self.output_ = output
        return output

biLinears = {
    False: nn.Linear,
    'Vanilla': BiLinearVanilla,
}

class BiConv1dVanilla(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1dVanilla, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        sw = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

biConv1ds = {
    False: nn.Conv1d,
    'Vanilla': BiConv1dVanilla,
}

class BiConv2dVanilla(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv2dVanilla, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        sw = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1, -1).detach()
        bw = BinaryQuantize_Vanilla().apply(bw, sw)
        ba = BinaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

biConv2ds = {
    False: nn.Conv2d,
    'Vanilla': BiConv2dVanilla,
}

def Count(module: nn.Module, id = -1):
    id = 0 if id == -1 else id
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.ModuleList):
            for child_child_module in child_module:
                id = Count(child_child_module, id)
        else:
            id = Count(child_module, id)
            if isinstance(child_module, nn.Linear):
                id += 1
            elif isinstance(child_module, nn.Conv1d):
                id += 1
            elif isinstance(child_module, nn.Conv2d):
                id += 1
    return id

def Modify(module: nn.Module, method='Sign', id=-1, first=-1, last=-1):
    id = 0 if id == -1 else id
    if method != False:
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.ModuleList):
                for child_child_module in child_module:
                    _, id = Modify(child_child_module, method=method, id=id, first=first, last=last)
            else:
                _, id = Modify(child_module, method=method, id=id, first=first, last=last)
                if isinstance(child_module, nn.Linear):
                    id += 1
                    if id == first or id == last:
                        continue
                    new_layer = biLinears[method](child_module.in_features,
                                                            child_module.out_features,
                                                            False if child_module.bias == None else True)
                    new_layer.weight = module._modules[name].weight
                    new_layer.bias = module._modules[name].bias
                    module._modules[name] = new_layer
                elif isinstance(child_module, nn.Conv1d):
                    id += 1
                    if id == first or id == last:
                        continue
                    new_layer = biConv1ds[method](in_channels=child_module.in_channels,
                                                            out_channels=child_module.out_channels,
                                                            kernel_size=child_module.kernel_size,
                                                            stride=child_module.stride,
                                                            padding=child_module.padding,
                                                            dilation=child_module.dilation,
                                                            groups=child_module.groups,
                                                            bias=False if child_module.bias == None else True,
                                                            padding_mode=child_module.padding_mode)
                    new_layer.weight = module._modules[name].weight
                    new_layer.bias = module._modules[name].bias
                    module._modules[name] = new_layer
                elif isinstance(child_module, nn.Conv2d):
                    id += 1
                    if id == first or id == last:
                        continue
                    new_layer = biConv2ds[method](in_channels=child_module.in_channels,
                                                            out_channels=child_module.out_channels,
                                                            kernel_size=child_module.kernel_size,
                                                            stride=child_module.stride,
                                                            padding=child_module.padding,
                                                            dilation=child_module.dilation,
                                                            groups=child_module.groups,
                                                            bias=False if child_module.bias == None else True,
                                                            padding_mode=child_module.padding_mode)
                    new_layer.weight = module._modules[name].weight
                    new_layer.bias = module._modules[name].bias
                    module._modules[name] = new_layer
    return module, id
