import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin
import math


# ---------------------------------------------------------------------------------------------------------------------
#
#           For getting the features by index
#
def get_center_feature(inputs, k):
    """
    k is equal to idx.shape[-1]
    :param inputs:
    :param k:
    :return:
    """
    inputs = inputs.repeat(1, 1, k, 1)
    return inputs


def batched_index_select(inputs, idx):
    """
    This can be used for neighbors features fetching
    faster bached index select, return a feature by a tensor idx.
    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param idx: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices = inputs.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = inputs.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 2, 1).contiguous()
    return feature


def batched_index_select2(inputs, index):
    """
    Same function as batched_index_select
    # The second way to fetch features.
    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices, _ = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.view(batch_size, -1)

    inputs = inputs.transpose(2, 1).contiguous().view(-1, num_dims)
    index = index.view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.view(-1)

    return torch.index_select(inputs, 0, index).view(batch_size, -1, num_dims).transpose(2, 1).view(batch_size, num_dims, -1, k)


# ---------------------------------------------------------------------------------------------------------------------
#
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


# ---------------------------------------------------------------------------------------------------------------------
#
#           Basic layers
#
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer1d(nc, norm='batch'):
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


def norm_layer2d(nc, norm='batch'):
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MultiSeq(Seq):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# ---------------------------------------------------------------------------------------------------------------------
#
#           Basic Convolution Layer
#
class MLP1dLayer(nn.Module):
    """
    only suitable for 3-d tensor
    """
    def __init__(self, channels, act='relu', norm=None, bias=False, drop=0):
        super(MLP1dLayer, self).__init__()
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if norm:
                m.append(norm_layer1d(channels[-1], norm))
            if act:
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout(drop))

        self.body = Seq(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Lin):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.body(x)


class Conv1dLayer(Seq):
    """
    only suitable for 3-d tensor. If kernel_size=1, then Conv1dLayer works the same as MLP1dLayer
    """

    def __init__(self, channels, act='relu', norm=None, bias=False, kernel_size=1, stride=1, dilation=1, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size,
                               stride, dilation*(kernel_size//2), dilation, bias=bias))
            if norm:
                m.append(norm_layer1d(channels[-1], norm))
            if act:
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout(drop))
        super(Conv1dLayer, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MLP2dLayer(Seq):
    """
    suitable for 4-d tensor.
    The convolution kernel along the 3-nd dim (the neighborhood dim) is always 1. kernel is shared among the neighbors.
    """
    def __init__(self, channels, act='relu', norm='batch', bias=False, kernel_size=9, stride=1, dilation=1, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], [kernel_size, 1],
                               [stride, 1], 0, [dilation, 1], bias=bias))
            if norm:
                m.append(norm_layer2d(channels[-1], norm))
            if act:
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(MLP2dLayer, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Conv2dLayer(Seq):
    """
    suitable for 4-d tensor.
    """
    def __init__(self, channels, act='relu', norm=None, bias=True, kernel_size=1, stride=1, dilation=1, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size,
                               stride, dilation*(kernel_size//2), dilation, bias=bias))
            if norm:
                m.append(norm_layer2d(channels[-1], norm))
            if act:
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(Conv2dLayer, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


