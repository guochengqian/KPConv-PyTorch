import torch
from torch import nn
from .torch_nn import Conv1dLayer, get_center_feature, act_layer, norm_layer1d, glorot
from .torch_edge import DilatedKNN2d, add_self_loops, remove_self_loops
import torch.nn.functional as F


"""
torch_vertex1d is used for stacked point clouds. 
point cloud may have different size, here a mini-batch of points are stacked to a vector with shape [N, C]. 
(this is the main difference with torch_vertex2d, where a mini-batch is with shape [B, C, N])

However, the number of neighbors for each point is the same. (this is the main difference with gcn_lib/sparse. )

"""


class MRConv1d(nn.Module):
    r"""Revised Max-Relative Graph Convolution layer (with activation, batch normalization)
    from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
    <https://arxiv.org/abs/1910.06849>`_ paper
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='max'):
        super(MRConv1d, self).__init__()
        self.nn = Conv1dLayer([in_channels * 2, out_channels], act, norm, bias)
        self.k = k

        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        # edge_index = remove_self_loops(edge_index)
        x_i = get_center_feature(x, self.k)
        x_j = torch.index_select(x, 0, edge_index[0])
        aggr_out, _ = self.aggr(x_j - x_i, -1, keepdim=False)
        return self.nn(torch.cat([x, aggr_out], dim=1))


class EdgeConv1d(nn.Module):
    r"""Revised Edge convolution layer (with activation, batch normalization)
    from the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/abs/1801.07829>`_ paper
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='max'):
        super(EdgeConv1d, self).__init__()
        self.nn = Conv1dLayer([in_channels * 2, out_channels], act, norm, bias)
        self.k = k

        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean

    def forward(self, x, edge_index):
        # edge_index = remove_self_loops(edge_index)
        x_i = get_center_feature(x, self.k)

        # todo: debug.
        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((x, torch.zeros_like(x[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        x_j = s_pts[edge_index[0], :].transpose(1, 2)

        aggr_out, _ = self.aggr(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=False)
        return aggr_out


class GraphConv1d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels,
                 conv='edge', act='relu', norm=None, bias=True,
                 k=9
                 ):
        super(GraphConv1d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv1d(in_channels, out_channels, act, norm, bias, k)
        elif conv == 'mr':
            self.gconv = MRConv1d(in_channels, out_channels, act, norm, bias, k)
        # elif conv.lower() == 'gat':
        #     self.gconv = GATConv2d(in_channels, out_channels, act, norm, bias, k)
        # elif conv.lower() == 'gcn':
        #     self.gconv = SemiGCNConv2d(in_channels, out_channels, act, norm, bias, k)
        # elif conv.lower() == 'gin':
        #     self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias, k)
        # elif conv.lower() == 'sage':
        #     self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, k, relative=False)
        # elif conv.lower() == 'rsage':
        #     self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, k, relative=True)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class PlainGraphBlock1d(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, k=9):
        super(PlainGraphBlock1d, self).__init__()
        self.body = GraphConv1d(channels, channels, conv, act, norm, bias, k)

    def forward(self, inputs):
        x, edge_index = inputs
        return self.body(x, edge_index), edge_index


class ResGraphBlock1d(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, k=9):
        super(ResGraphBlock1d, self).__init__()
        self.body = GraphConv1d(channels, channels, conv, act, norm, bias, k)

    def forward(self, inputs):
        x, edge_index = inputs
        return self.body(x, edge_index)+x, edge_index


class DenseGraphBlock1d(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, conv='edge', act='relu', norm=None, bias=True, k=9):
        super(DenseGraphBlock1d, self).__init__()
        self.body = GraphConv1d(in_channels, out_channels, conv, act, norm, bias, k)

    def forward(self, inputs):
        x, edge_index = inputs
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index


# class DynConv2d(GraphConv2d):
#     """
#     Dynamic graph convolution layer
#     """
#
#     def __init__(self, in_channels, out_channels,
#                  conv='edge', act='relu', norm=None, bias=True,
#                  k=9,  # k: number of neighbors
#                  dilation=1, stochastic=False, epsilon=0.0):
#         super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
#         self.k = k
#         self.d = dilation
#         self.dilated_knn_graph = DilatedKNN2d(k, dilation,
#                                               self_loop=True, stochastic=stochastic, epsilon=epsilon)
#
#     def forward(self, x):
#         edge_index = self.dilated_knn_graph(x)
#         return super(DynConv2d, self).forward(x, edge_index)
#
#
# class PlainDynBlock2d(nn.Module):
#     """
#     Plain Dynamic graph convolution block
#     """
#
#     def __init__(self, in_channels, conv='edge',
#                  act='relu', norm=None, bias=True,
#                  k=9, dilation=1,
#                  stochastic=False, epsilon=0.0):
#         super(PlainDynBlock2d, self).__init__()
#         self.body = DynConv2d(in_channels, in_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class ResDynBlock2d(nn.Module):
#     r"""Revised residual dynamic Graph convolution layer (with activation, batch normalization)
#     from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
#     <https://arxiv.org/abs/1910.06849>`_ paper
#     """
#
#     def __init__(self, in_channels, conv='edge',
#                  act='relu', norm=None, bias=True,
#                  k=9,
#                  dilation=1, stochastic=False, epsilon=0.0):
#         super(ResDynBlock2d, self).__init__()
#         self.body = DynConv2d(in_channels, in_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)
#
#     def forward(self, x):
#         return self.body(x) + x
#
#
# class DenseDynBlock2d(nn.Module):
#     r"""Revised densely connected dynamic Graph Convolution layer (with activation, batch normalization)
#     from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
#     <https://arxiv.org/abs/1910.06849>`_ paper
#     """
#
#     def __init__(self, in_channels, out_channels=64, conv='edge',
#                  act='relu', norm=None, bias=True,
#                  k=9, dilation=1, stochastic=False, epsilon=0.0):
#         super(DenseDynBlock2d, self).__init__()
#         self.body = DynConv2d(in_channels, out_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)
#
#     def forward(self, x):
#         dense = self.body(x)
#         return torch.cat((x, dense), 1)
#
#
# class GraphPool2d(nn.Module):
#     """
#     Dense Dynamic graph pooling block
#     """
#
#     def __init__(self, in_channels, ratio=0.5, conv='edge', **kwargs):
#         super(GraphPool2d, self).__init__()
#         self.gnn = DynConv2d(in_channels, 1, conv=conv, **kwargs)
#         self.ratio = ratio
#
#     def forward(self, x):
#         """"""
#         score = torch.tanh(self.gnn(x))
#         _, indices = score.topk(int(x.shape[2] * self.ratio), 2)
#         return torch.gather(x, 2, indices.repeat(1, x.shape[1], 1, 1))
#
#
# class VLADPool2d(torch.nn.Module):
#     def __init__(self, in_channels, num_clusters=64, alpha=100.0):
#         super(VLADPool2d, self).__init__()
#         self.in_channels = in_channels
#         self.num_clusters = num_clusters
#         self.alpha = alpha
#
#         self.lin = nn.Linear(in_channels, self.num_clusters, bias=True)
#
#         self.centroids = nn.Parameter(torch.rand(self.num_clusters, in_channels))
#         self._init_params()
#
#     def _init_params(self):
#         self.lin.weight = nn.Parameter((2.0 * self.alpha * self.centroids))
#         self.lin.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))
#
#     def forward(self, x, norm_intra=False, norm_L2=False):
#         B, C, N, _ = x.shape
#         x = x.squeeze().transpose(1, 2)  # B, N, C
#         K = self.num_clusters
#         soft_assign = self.lin(x)  # soft_assign of size (B, N, K)
#         soft_assign = F.softmax(soft_assign, dim=1).unsqueeze(1)  # soft_assign of size (B, N, K)
#         soft_assign = soft_assign.expand(-1, C, -1, -1)  # soft_assign of size (B, C, N, K)
#
#         # input x of size (NxC)
#         xS = x.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, K)  # xS of size (B, C, N, K)
#         cS = self.centroids.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).transpose(2, 3)  # cS of size (B, C, N, K)
#
#         residual = (xS - cS)  # residual of size (B, C, N, K)
#         residual = residual * soft_assign  # vlad of size (B, C, N, K)
#
#         vlad = torch.sum(residual, dim=2).unsqueeze(-1)  # (B, C, K)
#
#         if (norm_intra):
#             vlad = F.normalize(vlad, p=2, dim=1)  # intra-normalization
#             # print("i-norm vlad", vlad.shape)
#         if (norm_L2):
#             vlad = vlad.view(-1, K * C)  # flatten
#             vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
#
#         # return vlad.view(B, -1, 1, 1)
#         return vlad
