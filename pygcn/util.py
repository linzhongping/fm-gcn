# coding=utf-8
from __future__ import print_function
import pickle
import numpy as np
import scipy.sparse as sp
from keras.utils import to_categorical
import scipy.io as sio
# import torch
import pickle as pkl
import sys
import random
import networkx as nx
import os
import torch
# seed = 123
# random.seed(seed)
# torch.random.manual_seed(seed)
label_set = ["Agents","AI","DB","IR","ML","HCI"]
# import networkx as nx

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def load_data(content_path,cites_path):
    label_set = ["Theory", "Neural_Networks", "Probabilistic_Methods", "Rule_Learning", "Case_Based",
               "Genetic_Algorithms", "Reinforcement_Learning"]  #cora
    # seed = random.randint(0, 10000)
    # np.random.seed(seed)
    # label_set = ["Agents","AI","DB","IR","ML","HCI"]  #citeseer
    with open(content_path,'r') as f:
        lines = f.readlines()
        ids = []
        features = []
        labels = []
        for line in lines:
            id = line.split('\t')[0]
            feature = line.split('\t')[1:-1]
            label = line.split('\t')[-1].replace('\n','')
            ids.append(id)
            features.append(feature)
            labels.append(label_set.index(label))
        features = np.array(features,dtype = np.float)
        # print(features.shape)
        features = sp.csr_matrix(features,shape=features.shape)
        # print(features)
    with open(cites_path,'r') as f:
        links = f.readlines()
        adj = np.zeros([len(ids),len(ids)],dtype = int)
        # print(ids)
        for l in links:
            u = l.split('\t')[0]
            v = l.split('\t')[1].replace('\n','')
            try:
                adj[ids.index(u)][ids.index(v)], adj[ids.index(v)][ids.index(u)]= 1.0 , 1.0
            except:
                pass

        for i in range(len(ids)):
            adj[i][i] = 1.0




        labels = to_categorical(labels)
        labels = torch.LongTensor(np.where(labels)[1])


        features = preprocess_features(features)
        features = torch.FloatTensor(features.todense())


        adj = sp.coo_matrix(adj)
        adj = preprocess_adj(adj)
        adj = torch.FloatTensor(adj)

        train_idx = range(140)
        test_idx = range(500,1000)
        val_idx = range(1500,2000)

        train_idx = torch.LongTensor(train_idx)
        test_idx = torch.LongTensor(test_idx)
        val_idx = torch.LongTensor(val_idx)

    return adj,features,labels,train_idx,test_idx,val_idx






# load_nell_data('nell.0.001')

def load_dataset():
    dict = sio.loadmat('DBLP.mat')
    #dict = sio.loadmat('flickr.mat')
    Y = dict['Y']
    A = dict['A']
    X = dict['X']
    X = X.astype('float64')

    Y = Y.astype('int64')
    A = A.astype('float64')
    X[ X > 1 ] = 1
    features = sp.csr_matrix(X, dtype=np.float64)
    features = preprocess_features(features)

    features = torch.FloatTensor(np.array(features.todense()))

    adj = sp.coo_matrix(A)
    adj = preprocess_adj(adj)
    adj = torch.FloatTensor(adj)

    labels = to_categorical(Y)
    labels = torch.LongTensor(np.where(labels)[1])

    # features = torch.FloatTensor(X)

    train_idx = range(80)
    test_idx = range(500,1500)
    val_idx = range(1500,2000)


    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)
    val_idx = torch.LongTensor(val_idx)
    return adj, features, labels, train_idx, test_idx, val_idx




def normalized_adj(adj):
    # print(type(adj))
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum,-0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj):
    adj_normalized = normalized_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    # return normalized_adj(adj + sp.eye(adj.shape[0]))

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1),dtype='float')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features
    # print(features)
    # rowsum = np.array(features.sum(1),dtype='float')
    #
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = np.diag(r_inv)
    # features = r_mat_inv.dot(features)
    # # return sparse_to_tuple(features)
    # return features

#
def accuracy(output,labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#
#
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


def load_data_pubmed(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    features[test_idx_reorder, :] = features[test_idx_range, :]


    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))



    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = preprocess_adj(adj)
    adj = torch.FloatTensor(adj)


    labels = np.vstack((ally, ty))
    print(labels.shape)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = torch.LongTensor(np.where(labels)[1])
    print(labels.shape)
    idx_test = torch.LongTensor(test_idx_range.tolist())

    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y)+500))


    return adj, features, labels, idx_train, idx_test[:-15], idx_val

