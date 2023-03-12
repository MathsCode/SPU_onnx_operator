import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
from torch.onnx.symbolic_helper import parse_args 
import torch.onnx

class PCSample(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        input_xyzs,
        input_xyz_num,
        sample_xyz_num,
        sample_method,
    ):
        # return sample_index
        return g.op(
            "point_cloud_sample",
            input_xyzs_i = 0,
            input_xyz_num_i = 0,
            sample_xyz_num_i = 0,
            sample_method_i = 0,
        )

class PCQuery(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        input_xyzs,
        input_xyz_num,
        query_xyz_index,
        query_xyz_num,
        query_method,
        query_num,
        radius,
    ):
        # return group_index
        return g.op(
            "point_cloud_sample",
            input_xyzs_i = input_xyzs,
            input_xyz_num_i = input_xyz_num,
            query_xyz_index_i = query_xyz_index,
            query_xyz_num_i = query_xyz_num,
            query_method_i = query_method,
            query_num_i = query_num,
            radius_i = radius,
        )

class PCFetch(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        input_xyzs,
        input_features,
        fetch_num,
        fetch_index,
    ):
        # return output_xyzs, output_features
        return g.op(
            "point_cloud_sample",
            input_xyzs_i = input_xyzs,
            input_features_i = input_features,
            fetch_num_i = fetch_num,
            fetch_index_i = fetch_index,
        )

class PCKernelMapping(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        kernel,
        stride,
        pad,
        indice_key,
        submconv_flag,

    ):
        # return mapping_table
        return g.op(
            "point_cloud_sample",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            kernel_i = kernel,
            stride_i = stride,
            pad_i = pad,
            indice_key_i = indice_key,
            submconv_flag_i = submconv_flag,
        )

class PCSubmConv3d(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        voxel_features,
        cin,
        cout,
        kernel,
        mapping_table,
        weight,
        bias_flag,
        bias,
        bn_flag,
        relu_flag,

    ):
        # return output_voxel_num, output_voxel_xyz, output_voxel_feature
        return g.op(
            "point_cloud_sample",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            voxel_features_i = voxel_features,
            cin_i = cin,
            cout_i = cout,
            kernel_i = kernel,
            mapping_table_i = mapping_table,
            weight_i = weight,
            bias_flag_i = bias_flag,
            bias_i = bias,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
        )

class PCSparseConv3d(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        voxel_features,
        cin,
        cout,
        kernel,
        mapping_table,
        weight,
        bias_flag,
        bias,
        bn_flag,
        relu_flag,

    ):
        # return output_voxel_num, output_voxel_xyz, output_voxel_feature
        return g.op(
            "point_cloud_sample",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            voxel_features_i = voxel_features,
            cin_i = cin,
            cout_i = cout,
            kernel_i = kernel,
            mapping_table_i = mapping_table,
            weight_i = weight,
            bias_flag_i = bias_flag,
            bias_i = bias,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
        )

class DenseConv2d(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        height,
        width,
        cin,
        cout,
        kernel,
        stride,
        pad,
        weight,
        bias_flag,
        bias,
        bn_flag,
        relu_flag,
        input_feature,
    ):
        # return output_feature
        return g.op(
            "point_cloud_sample",
            height_i = height,
            width_i = width,
            cin_i = cin,
            cout_i = cout,
            kernel_i = kernel,
            stride_i = stride,
            pad_i = pad,
            weight_i = weight,
            bias_flag_i = bias_flag,
            bias_i = bias,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
            input_feature_i = input_feature,
        )

pcs = PCSample.apply

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