import math

from torch.nn import Embedding
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
from torch.autograd import Variable
torch.random.manual_seed(2070)
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BI_Intereaction(Module):

    def __init__(self, in_features, k_embedding, train_idx):
        '''
        :param in_features: 输入特征维数
        :param k:  单一特征embedding
        :param bias:
        '''

        super(BI_Intereaction, self).__init__()
        self.in_features = in_features
        self.k_embedding = k_embedding
        self.train_idx = train_idx
        self.embedding = Embedding(in_features, k_embedding)

        # self.weight = Parameter(torch.FloatTensor(in_features,k_embedding))
        # self.reset_parameters()
        self.init_embedding()

    def init_embedding(self):
        init.xavier_uniform_(self.embedding.weight)
        # print('embedding_init',self.embedding.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def bi_pooling(self, input, embeddings):
        output = torch.zeros(input.shape[0], self.k_embedding)
        rows, cols = input.shape[0], input.shape[1]

        # print(rows,cols)
        for _ in self.train_idx:
            left  = torch.zeros(self.k_embedding)
            right = torch.zeros(self.k_embedding)
            nonzero_index = torch.nonzero(input[_])
            # print(nonzero_index.squeeze(1))
            for i in nonzero_index.squeeze(1):

                left  += torch.mul(embeddings.weight[i] , input[_][i])
                right += torch.mul(embeddings.weight[i] , input[_][i]) ** 2

            vec = 0.5 * (left ** 2 - right)
            del left, right
            output[_] = vec
        return output


    def forward(self, input):
        return self.bi_pooling(input,self.embedding)


