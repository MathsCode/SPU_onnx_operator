import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
from torch.onnx.symbolic_helper import parse_args
import torch.onnx
from torch_geometric.utils import spmm
from torch.nn import ReLU


class GNNNeighborSample(Function):
    @staticmethod
    def symbolic(g, data, input_nodes, replace = False, directed = False):
        return g.op("GNN_Neighbour_Sample", data, input_nodes, replace, directed)

    @staticmethod
    def forward(ctx, data, input_nodes, replace, directed):
        # return subgraph
        return replace


class GNNGEMM(Function):
    @staticmethod
    def symbolic(g, W, X, bias):
        return g.op("GNN_GEMM", W, X, bias)

    @staticmethod
    def forward(ctx, W, X, bias=None):
        if bias == None:
            result = torch.mul(W, X)
        else:
            result = torch.mul(W, X) + bias
        return result


class GNNSpMM(Function):
    @staticmethod
    def symbolic(g, Adj, X, reduce):
        return g.op("GNN_SpMM", Adj, X, reduce)

    @staticmethod
    def forward(ctx, Adj, X, reduce="sum"):
        return spmm(Adj, X, reduce=reduce)


class GNNReLU(Function):
    @staticmethod
    def symbolic(g, input, inplace):
        return g.op("GNN_ReLU", input, inplace)

    @staticmethod
    def forward(ctx, input, inplace=False):
        relu = ReLU(inplace)
        result = relu(input)
        return result


class GNNconcat(Function):
    @staticmethod
    def symbolic(g, tensors, dim):
        return g.op("GNN_concat", tensors, dim)

    @staticmethod
    def forward(ctx, tensors, dim):
        return torch.concat(tensors=tensors, dim=dim)


class GNNnormalize(Function):
    @staticmethod
    def symbolic(g, input, p, dim, eps, out):
        return g.op("GNN_normalize", input, p, dim, eps, out)

    @staticmethod
    def forward(ctx, input, p, dim, eps, out):
        return torch.nn.functional.normalize(input,p,dim,eps,out)
    
