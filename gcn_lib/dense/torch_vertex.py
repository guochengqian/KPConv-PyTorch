import torch
from torch import nn
from .torch_nn import MLP2dLayer, get_center_feature, batched_index_select, act_layer, norm_layer2d, glorot
from .torch_edge import DilatedKNN2d, add_self_loops, remove_self_loops
import torch.nn.functional as F


class MRConv2d(nn.Module):
    r"""Revised Max-Relative Graph Convolution layer (with activation, batch normalization)
    from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
    <https://arxiv.org/abs/1910.06849>`_ paper
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='max'):
        super(MRConv2d, self).__init__()
        self.nn = MLP2dLayer([in_channels * 2, out_channels], act, norm, bias)
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
        x_j = batched_index_select(x, edge_index[0])
        aggr_out, _ = self.aggr(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, aggr_out], dim=1))


class EdgeConv2d(nn.Module):
    r"""Revised Edge convolution layer (with activation, batch normalization)
    from the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/abs/1801.07829>`_ paper
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='max'):
        super(EdgeConv2d, self).__init__()
        self.nn = MLP2dLayer([in_channels * 2, out_channels], act, norm, bias)
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
        x_j = batched_index_select(x, edge_index[0])
        aggr_out, _ = self.aggr(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return aggr_out


class GATConv2d(nn.Module):
    r"""Revised one head graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='sum',
                 negative_slope=0.2, dropout=0):
        super(GATConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.k = k
        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean

        self.nn = MLP2dLayer([in_channels, out_channels], act, norm, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index):
        x = self.nn(x)
        # edge_index = add_self_loops(edge_index)
        x_i = get_center_feature(x, self.k)
        x_j = batched_index_select(x, edge_index[0])

        # x_i BxCxNxk
        alpha = (torch.cat([x_i, x_j], dim=1) * self.att).sum(dim=1, keepdim=True)  # -1 xk
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_j * alpha

        aggr_out, _ = self.aggr(x_j, -1, keepdim=True)
        # aggr_out = x_j.sum(dim=-1, keepdim=True)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class SemiGCNConv2d(nn.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='sum'):
        super(SemiGCNConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = MLP2dLayer([in_channels, out_channels], act, norm, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index):
        """"""
        x = self.nn(x)
        # edge_index = add_self_loops(edge_index)
        x_j = batched_index_select(x, edge_index[0])

        deg = edge_index.shape[-1]
        norm = 1 / deg
        x_j = x_j * norm
        # aggr_out = x_j.sum(dim=-1, keepdim=True)
        aggr_out, _ = self.aggr(x_j, -1, keepdim=True)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GINConv2d(nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),
    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 k=9, aggr='sum',
                 eps=0, train_eps=False):
        super(GINConv2d, self).__init__()
        self.nn = MLP2dLayer([in_channels, out_channels], act, norm, bias)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        # edge_index = remove_self_loops(edge_index)
        x_j = batched_index_select(x, edge_index[0])
        aggr_out, _ = self.aggr(x_j, -1, keepdim=True)
        out = self.nn((1 + self.eps) * x + aggr_out)
        return out


class RSAGEConv2d(nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)
        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels, out_channels,
                 act='relu', norm=True, bias=True,
                 k=9, aggr='max',
                 relative=False):
        super(RSAGEConv2d, self).__init__()
        self.relative = relative
        self.nn = MLP2dLayer([out_channels + in_channels, out_channels], act, norm=None, bias=False)
        self.pre_nn = MLP2dLayer([in_channels, out_channels], act, norm=None, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.norm = norm

        if aggr == 'max':
            self.aggr = torch.max
        elif aggr == 'mean':
            self.aggr = torch.mean
        elif aggr == 'sum':
            self.aggr = torch.mean

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index):
        """"""
        x_j = batched_index_select(x, edge_index[0])
        if self.relative:
            x_i = get_center_feature(x, self.k)
            x_j = self.pre_nn(x_j - x_i)
        else:
            x_j = self.pre_nn(x_j)
        aggr_out, _ = torch.max(x_j, -1, keepdim=True)

        out = self.nn(torch.cat((x, aggr_out), dim=1))
        if self.bias is not None:
            out = out + self.bias
        if self.norm is not None:
            out = F.normalize(out, dim=1)
        return out


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels,
                 conv='edge', act='relu', norm=None, bias=True,
                 k=9
                 ):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias, k)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias, k)
        elif conv.lower() == 'gat':
            self.gconv = GATConv2d(in_channels, out_channels, act, norm, bias, k)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv2d(in_channels, out_channels, act, norm, bias, k)
        elif conv.lower() == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias, k)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, k, relative=False)
        elif conv.lower() == 'rsage':
            self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, k, relative=True)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels,
                 conv='edge', act='relu', norm=None, bias=True,
                 k=9,  # k: number of neighbors
                 dilation=1, stochastic=False, epsilon=0.0):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = k
        self.d = dilation
        self.dilated_knn_graph = DilatedKNN2d(k, dilation,
                                              self_loop=True, stochastic=stochastic, epsilon=epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """

    def __init__(self, in_channels, conv='edge',
                 act='relu', norm=None, bias=True,
                 k=9, dilation=1,
                 stochastic=False, epsilon=0.0):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)

    def forward(self, x):
        return self.body(x)


class ResDynBlock2d(nn.Module):
    r"""Revised residual dynamic Graph convolution layer (with activation, batch normalization)
    from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
    <https://arxiv.org/abs/1910.06849>`_ paper
    """

    def __init__(self, in_channels, conv='edge',
                 act='relu', norm=None, bias=True,
                 k=9,
                 dilation=1, stochastic=False, epsilon=0.0):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)

    def forward(self, x):
        return self.body(x) + x


class DenseDynBlock2d(nn.Module):
    r"""Revised densely connected dynamic Graph Convolution layer (with activation, batch normalization)
    from the `"DeepGCNs: Making GCNs Go as Deep as CNNs"
    <https://arxiv.org/abs/1910.06849>`_ paper
    """

    def __init__(self, in_channels, out_channels=64, conv='edge',
                 act='relu', norm=None, bias=True,
                 k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, conv, act, norm, bias, k, dilation, stochastic, epsilon)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)


class GraphPooling(nn.Module):
    """
    Dense Dynamic graph pooling block
    """

    def __init__(self, in_channels, ratio=0.5, conv='edge', **kwargs):
        super(GraphPooling, self).__init__()
        self.gnn = DynConv2d(in_channels, 1, conv=conv, **kwargs)
        self.ratio = ratio

    def forward(self, x):
        """"""
        score = torch.tanh(self.gnn(x))
        _, indices = score.topk(int(x.shape[2] * self.ratio), 2)
        return torch.gather(x, 2, indices.repeat(1, x.shape[1], 1, 1))


class VLADPool(torch.nn.Module):
    def __init__(self, in_channels, num_clusters=64, alpha=100.0):
        super(VLADPool, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.alpha = alpha

        self.lin = nn.Linear(in_channels, self.num_clusters, bias=True)

        self.centroids = nn.Parameter(torch.rand(self.num_clusters, in_channels))
        self._init_params()

    def _init_params(self):
        self.lin.weight = nn.Parameter((2.0 * self.alpha * self.centroids))
        self.lin.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))

    def forward(self, x, norm_intra=False, norm_L2=False):
        B, C, N, _ = x.shape
        x = x.squeeze().transpose(1, 2)  # B, N, C
        K = self.num_clusters
        soft_assign = self.lin(x)  # soft_assign of size (B, N, K)
        soft_assign = F.softmax(soft_assign, dim=1).unsqueeze(1)  # soft_assign of size (B, N, K)
        soft_assign = soft_assign.expand(-1, C, -1, -1)  # soft_assign of size (B, C, N, K)

        # input x of size (NxC)
        xS = x.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, K)  # xS of size (B, C, N, K)
        cS = self.centroids.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).transpose(2, 3)  # cS of size (B, C, N, K)

        residual = (xS - cS)  # residual of size (B, C, N, K)
        residual = residual * soft_assign  # vlad of size (B, C, N, K)

        vlad = torch.sum(residual, dim=2).unsqueeze(-1)  # (B, C, K)

        if (norm_intra):
            vlad = F.normalize(vlad, p=2, dim=1)  # intra-normalization
            # print("i-norm vlad", vlad.shape)
        if (norm_L2):
            vlad = vlad.view(-1, K * C)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # return vlad.view(B, -1, 1, 1)
        return vlad
