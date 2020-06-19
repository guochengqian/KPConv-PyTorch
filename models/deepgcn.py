import torch
from gcn_lib.dense import DilatedKNN2d
from gcn_lib.dense.torch_vertex1d import MLP1dLayer, GraphConv1d, DenseGraphBlock1d, ResGraphBlock1d, PlainGraphBlock1d
from torch.nn import Sequential as Seq
import numpy as np


class DeepGCN(torch.nn.Module):
    def __init__(self, config, lbl_values, ign_lbls):
        super(DeepGCN, self).__init__()
        in_channels = config.in_channels
        n_classes = config.n_classes
        channels = config.n_filters
        block = config.block
        conv = config.conv

        k = config.k
        self.k = k
        act = config.act
        norm = config.norm
        bias = config.bias
        stochastic = config.stochastic
        epsilon = config.epsilon
        dropout = config.dropout

        self.n_blocks = config.n_blocks

        c_growth = 0

        # self.knn = DilatedKNN2d(k, 1, stochastic, epsilon)
        self.head = GraphConv1d(in_channels, channels, conv, act, norm, bias, k)

        if block.lower() == 'res':
            self.backbone = Seq(*[ResGraphBlock1d(channels, conv, act, norm, bias, k)
                                  for _ in range(self.n_blocks - 1)])
        elif block.lower() == 'plain':
            self.backbone = Seq(*[PlainGraphBlock1d(channels, conv, act, norm, bias, k)
                                  for _ in range(self.n_blocks - 1)])

        elif block.lower() == 'dense':
            c_growth = channels
            self.backbone = Seq(*[DenseGraphBlock1d(channels + i * c_growth, c_growth, conv, act, norm, bias, k)
                                  for i in range(self.n_blocks - 1)])
        else:
            raise NotImplementedError

        fusion_dims = int(channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))

        self.fusion_block = MLP1dLayer([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[MLP1dLayer([fusion_dims+1024, 512], act, norm, bias),
                                MLP1dLayer([512, 256], act, norm, bias, drop=dropout),
                                MLP1dLayer([256, n_classes], None, None, bias)])
        self.model_init()

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.reg_loss = 0.

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, batch, config):
        x = batch.features.clone().detach()
        edge_index = batch.neighbors[0][:, :self.k].unsqueeze(0)
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i]((feats[-1], edge_index))[0])
        feats = torch.cat(feats, dim=1)

        fusion, _ = torch.max(self.fusion_block(feats), dim=0, keepdim=True)
        fusion = fusion.repeat(feats.shape[0], 1)
        return self.prediction(torch.cat((fusion, feats), dim=1))

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

