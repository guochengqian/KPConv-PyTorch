import torch
from gcn_lib.dense import MLP2dLayer, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, DenseDynBlock2d, DilatedKNN2d
from torch.nn import Sequential as Seq


class DeepGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeepGCN, self).__init__()
        in_channels = args.in_channels
        n_classes = args.n_classes
        channels = args.n_filters
        block = args.block
        conv = args.conv

        k = args.k
        act = args.act
        norm = args.norm
        bias = args.bias
        stochastic = args.stochastic
        epsilon = args.epsilon
        dropout = args.dropout

        self.n_blocks = args.n_blocks

        c_growth = channels

        self.knn = DilatedKNN2d(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, channels, conv, act, norm, bias)

        if block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(channels, conv, act, norm, bias, k, 1+i, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        elif block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, conv, act, norm, bias,
                                                  k, 1+i, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = MLP2dLayer([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[MLP2dLayer([fusion_dims+1024, 512], act, norm, bias),
                                MLP2dLayer([512, 256], act, norm, bias, drop=dropout),
                                MLP2dLayer([256, n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, batch, config):
        x = batch.features.clone().detach()

        feats = [self.head(x, self.knn(x[:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        return self.output_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

