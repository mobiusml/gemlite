# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************

import torch
from gemlite.core import GemLiteLinearTriton, DType, GEMLITE_ACC_DTYPE

####################################################################################################
#16-bit activations / 8-bit weigths
class A16W8:
    def __init__(self, device='cuda:0'):
        self.device = device

    def from_weights(self, weight, bias):
        #GEMLITE_ACC_DTYPE[DType.FP16] = DType.FP32

        scales = torch.abs(weight.float()).amax(axis=1, keepdim=True) / 127.0
        W_q    = torch.round(weight / scales).to(device=self.device, dtype=torch.int8)
        scales = scales.to(device=self.device, dtype=torch.float16)

        in_features, out_features = weight.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=DType.FP16, 
                        output_dtype=DType.FP16, 
                        )

        gemlite_linear.pack(W_q, scales, 
                            zeros=None, 
                            bias=bias.to(device=self.device, dtype=torch.float16) if bias is not None else None, 
                            contiguous=False)

        gemlite_linear.W_group_mode       = 2
        gemlite_linear.channel_scale_mode = 0
        gemlite_linear.default_gemv       = 'GEMV_SPLITK'
        return gemlite_linear

    def from_linear(self, linear_layer):
        return self.from_weights(linear_layer.weight.data, linear_layer.bias.data if linear_layer.bias is not None else None)

####################################################################################################
#8-bit dynamic activations / 8-bit weights
class A8W8_dynamic:
    def __init__(self, device='cuda:0', fp8=False, weight_scale=1.):
        self.device = device
        self.fp8 = fp8
        self.weight_scale = weight_scale

    def from_weights(self, weight, bias):
        if(self.fp8): #FP8
            w_dtype, input_dtype, max_val = torch.float8_e4m3fn, DType.FP8, 448
        else: #INT8
            w_dtype, input_dtype, max_val = torch.int8, DType.INT8, 127

        
        weight = weight.float() * self.weight_scale
        scales = torch.abs(weight).amax(axis=1, keepdim=True) / max_val
        W_q    = torch.round(weight / scales).to(device=self.device, dtype=w_dtype)
        scales = scales.to(device=self.device, dtype=torch.float16)#.float()

        in_features, out_features = weight.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=DType.FP16,
                        scaled_activations=True,
                        )

        def scale_fct(x):
            x_shape  = x.shape
            out_x    = x.view(-1, x.shape[-1]) 
            scaled_x = torch.abs(out_x).amax(axis=1, keepdim=True) / max_val
            out_x    = torch.round(out_x / scaled_x).to(dtype=w_dtype)
            return out_x.view(x_shape), scaled_x

        gemlite_linear.scale_activations = scale_fct

        gemlite_linear.pack(W_q, scales / self.weight_scale, 
                            zeros=None, 
                            bias=bias.to(device=self.device, dtype=torch.float16) if bias is not None else None, 
                            contiguous=False)
        
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 3 #activation[:,None] + weight[None,:]
        gemlite_linear.meta_dtype         = DType.FP32
        gemlite_linear.default_gemv       = 'GEMV_SPLITK'
        return gemlite_linear

    def from_linear(self, linear_layer):
        return self.from_weights(linear_layer.weight.data, linear_layer.bias.data if linear_layer.bias is not None else None)

class A8W8_int8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', weight_scale=1.):
        super().__init__()
        self.device = device
        self.weight_scale = weight_scale
        self.fp8 = False

class A8W8_fp8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', weight_scale=1.):
        super().__init__()
        self.device = device
        self.weight_scale = weight_scale
        self.fp8 = True

####################################################################################################
#FP16 activations / Wn packed weights
class A16Wn:
    def __init__(self, device='cuda:0', packing_bitwidth=32, post_scale=False):
        self.packing_bitwidth = 32
        self.post_scale       = post_scale
        self.device           = device

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias):
        in_features, out_features = W_q.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=DType.FP16, 
                        output_dtype=DType.FP16, 
                        scaled_activations=False,
                        )

        gemlite_linear.pack(W_q.to(self.device), 
                            scales.to(device=self.device, dtype=torch.float16), 
                            zeros.to(device=self.device, dtype=torch.float16), 
                            bias=bias.to(device=self.device, dtype=torch.float16) if bias is not None else None, 
                            contiguous=True, 
                            packing_bitwidth=self.packing_bitwidth) 

        gemlite_linear.default_gemv = 'GEMV_REVSPLITK' 

        if(group_size == in_features):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 1
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 0

        return gemlite_linear

    def from_hqqlinear(self, hqq_layer):
        assert hqq_layer.meta['axis'] == 1, 'Only axis==1 is supported.'

        self.device = hqq_layer.W_q.device

        W_nbits    = hqq_layer.meta['nbits']
        group_size = hqq_layer.meta["group_size"]
        if(group_size is None):
            group_size = hqq_layer.in_features

        W_q    = hqq_layer.unpack(dtype=torch.uint8).view(hqq_layer.meta['shape']) #Expects uint8 for Wn quantization!
        scales = hqq_layer.meta['scale'].clone()
        zeros  = hqq_layer.meta['zero'].clone()
        bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None  

        gemlite_linear = self.from_weights(W_q, scales, zeros, W_nbits, group_size, bias)

        del hqq_layer.W_q
        del hqq_layer.meta
        del hqq_layer
        torch.cuda.empty_cache()

        return gemlite_linear


####################################################################################################
#FP8 dynamic activations / W4 packed weights
class A8Wn_dynamic(A16Wn):
    def __init__(self, device='cuda:0', packing_bitwidth=32, post_scale=False):
        super().__init__()
        self.packing_bitwidth = 32
        self.post_scale       = post_scale
        self.device           = device

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias):
        w_dtype, input_dtype, max_val = torch.float8_e4m3fn, DType.FP8, 448

        in_features, out_features = W_q.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=DType.FP16,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q.to(self.device), 
                            scales.to(device=self.device, dtype=torch.float16), 
                            zeros.to(device=self.device, dtype=torch.float16), 
                            bias=bias.to(device=self.device, dtype=torch.float16) if bias is not None else None, 
                            contiguous=True, 
                            packing_bitwidth=self.packing_bitwidth) 

        def scale_fct(x):
            x_shape  = x.shape
            out_x    = x.view(-1, x.shape[-1]) 
            scaled_x = torch.abs(out_x).amax(axis=1, keepdim=True) / max_val
            out_x    = torch.round(out_x / scaled_x).to(dtype=w_dtype)
            return out_x.view(x_shape), scaled_x

        gemlite_linear.scale_activations = scale_fct

        gemlite_linear.default_gemv = 'GEMV_REVSPLITK' 

        if(group_size == in_features):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 3
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 2

        return gemlite_linear

