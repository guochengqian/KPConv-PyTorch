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

    if len(inputs.shape) == 4:
        # for torch_vertex2d
        inputs = inputs.repeat(1, 1, 1, k)

    else:
        # for torch_vertex1d
        inputs = inputs.unsqueeze(-1).repeat(1, 1, k)
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
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 2, 3, 1).contiguous()
    return feature

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


class BatchNormBlock2d(nn.Module):

    def __init__(self, in_dim, use_bn=True, bn_momentum=0.1):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock2d, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm2d(in_dim, momentum=bn_momentum)
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(0)
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
            return x.squeeze(0)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class BatchNormBlock1d(nn.Module):

    def __init__(self, in_dim, use_bn=True, bn_momentum=0.1):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock1d, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(0)
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
            return x.squeeze(0)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


# ---------------------------------------------------------------------------------------------------------------------
#
#           Basic Convolution Layer
#
class MLP1dLayer(nn.Module):
    """
    only suitable for 3-d tensor
    """
    def __init__(self, channels, act='relu', norm=False, bias=False, drop=0):
        super(MLP1dLayer, self).__init__()
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if norm:
                m.append(BatchNormBlock1d(channels[-1]))
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

    def __init__(self, channels, act='relu', norm=False, bias=False, kernel_size=1, stride=1, dilation=1, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size,
                               stride, dilation*(kernel_size//2), dilation, bias=bias))
            if norm:
                m.append(BatchNormBlock2d(channels[-1]))
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
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
