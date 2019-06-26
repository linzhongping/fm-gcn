import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution,BI_Intereaction
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, train_idx):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(2 * nhid, nclass)

        self.bi1 = BI_Intereaction(nfeat, nhid, train_idx)

        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(nhid)
        # self.bn2 = nn.BatchNorm1d(2 * nhid)

    def forward(self, x, adj):

        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)


        # first GCN-layer
        x_left = F.relu(self.gc1(x, adj))
        x_left = F.dropout(x_left, self.dropout, training=self.training)
        # first Bi-Pooling-layer
        x_right = F.relu(self.bi1(x))
        x_right = F.dropout(x_right, self.dropout)

        # concat
        x = torch.cat((x_left, x_right), 1)
        # x = 0.5 * ( x_left + x_right )
        # second GCN-layer
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)
