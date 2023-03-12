import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
from torch.onnx.symbolic_helper import parse_args 
import torch.onnx

class GNNSample(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        mini_batch_id, 
        initial_batch_set,
        iteration_nums,
        samplingType ,
        max_sample_nums    
    ):
        # return output_graph
        return g.op(
            "GNN_sample", 
            mini_batch_id = 0, 
            initial_batch_set = 0,
            iteration_nums = 0,
            samplingType = 'individual',
            max_sample_nums = [2,2,2]     
        )

pcs = GNNSample.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(x)
        x = x.view(-1)
        x = pcs(x, 1, 1, 0)
        return x

net = TinyNet().cuda()
ipt = torch.ones(2,3,12,12).cuda()
torch.onnx.export(net, (ipt,), 'tinynet.onnx', opset_version=11, enable_onnx_checker=False)
print(onnx.load('tinynet.onnx'))