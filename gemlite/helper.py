# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************

import torch
import gc
import time
from tqdm import tqdm
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
from gemlite.triton_kernels.utils import M_MAPPING
############################################################################################################################################################
#16-bit activations / 8-bit weigths
class A16W8: #INT8 weights
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype

    def from_weights(self, weight, scales=None, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        if(scales is None):
            #Quantize
            dtype = weight.dtype if(self.dtype is None) else self.dtype
            weight = weight.to(dtype)
            assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should be floating point."
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            scales = weight.float().abs().amax(axis=1, keepdim=True) / 127.0
            W_q    = (weight / scales).round().to(device=self.device, dtype=torch.int8)
            scales = scales.to(device=self.device, dtype=dtype)
        else:
            #Pre-Quantized
            assert weight.dtype in [torch.int8], "Invalid weight.dtype, should be int8."
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device, dtype=torch.float32)
            dtype = torch.float16 if(self.dtype is None) else self.dtype
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

        #Bias
        bias = bias.clone().to(device=self.device, dtype=dtype) if (bias is not None) else None

        in_features, out_features = weight.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)
        gemlite_linear.W_group_mode       = 2
        gemlite_linear.channel_scale_mode = 0
        return gemlite_linear

    def from_linear(self, linear_layer):
        out_layer = self.from_weights(linear_layer.weight, linear_layer.bias)
        del linear_layer 
        torch.cuda.empty_cache()
        return out_layer

#FP16 activations / Wn packed weights
class A16Wn:
    def __init__(self, device='cuda:0', dtype=None, packing_bitwidth=None, post_scale=False):
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None):
        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        dtype = scales.dtype if(self.dtype is None) else self.dtype
        scales = scales.to(dtype)
        assert scales.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should floating point."
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        in_features, out_features = W_q.shape[::-1]

        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=dtype)
        zeros = zeros.to(device=self.device, dtype=dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size,  
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        gemlite_linear.pack(W_q, scales, zeros, bias=bias, packing_bitwidth=self.packing_bitwidth) 

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

        del hqq_layer.W_q
        del hqq_layer.meta
        del hqq_layer
        torch.cuda.empty_cache()

        return self.from_weights(W_q, scales, zeros, W_nbits, group_size, bias)

############################################################################################################################################################
#8-bit dynamic activations / 8-bit weights
class A8W8_dynamic:
    def __init__(self, device='cuda:0', dtype=None, fp8=False):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8

    def from_weights(self, weight, scales=None, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        if(self.fp8 is not False): #FP8
            if(self.fp8 in [torch.float8_e4m3fn]):
                w_dtype, input_dtype, max_val = torch.float8_e4m3fn, DType.FP8, 448
            if(self.fp8 in [torch.float8_e5m2]):
                w_dtype, input_dtype, max_val = torch.float8_e5m2, DType.FP8e5, 57344
        else: #INT8
            w_dtype, input_dtype, max_val = torch.int8, DType.INT8, 127

        if(scales is None):
            #Quantize
            dtype = weight.dtype if(self.dtype is None) else self.dtype
            weight = weight.to(dtype)
            assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should floating point."
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            weight = weight.float()
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            W_q    = weight / scales
            if(w_dtype.is_floating_point):
                W_q = W_q.to(device=self.device, dtype=w_dtype)
            else:
                W_q = W_q.round_().to(device=self.device, dtype=w_dtype)
            scales = scales.to(device=self.device, dtype=torch.float32)
        else:
            #Pre-Quantized
            assert weight.dtype.itemsize == 1, "Invalid weight.dtype, should be 8-bit."
            dtype = torch.float16 if(self.dtype is None) else self.dtype  
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device, dtype=torch.float32) 
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
        
        #Bias
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        in_features, out_features = weight.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)
        gemlite_linear.meta_dtype         = DType.FP32
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 3 #activation[:,None] + weight[None,:]
        return gemlite_linear

    def from_linear(self, linear_layer):
        out_layer = self.from_weights(linear_layer.weight, linear_layer.bias)
        del linear_layer 
        torch.cuda.empty_cache()
        return out_layer

class A8W8_int8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fp8 = False

class A8W8_fp8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None, use_fp8e5=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        if(use_fp8e5):
            self.fp8 = torch.float8_e5m2
        else:
            self.fp8 = torch.float8_e4m3fn

############################################################################################################################################################
#FP8 dynamic activations / W4 packed weights
class A8Wn_dynamic(A16Wn):
    def __init__(self, device='cuda:0', packing_bitwidth=None, dtype=None, post_scale=False, use_fp8e5=False):
        super().__init__()
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        if(use_fp8e5):
            self.fp8 = torch.float8_e5m2
        else:
            self.fp8 = torch.float8_e4m3fn

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None):
        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        dtype = scales.dtype if(self.dtype is None) else self.dtype
        scales = scales.to(dtype)
        assert scales.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should floating point."
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        input_dtype = TORCH_TO_DTYPE[self.fp8]
        fp32_scale = True
            
        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=dtype)
        zeros = zeros.to(device=self.device, dtype=dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        in_features, out_features = W_q.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros, bias=bias, packing_bitwidth=self.packing_bitwidth, fma_mode=False) 

        if(fp32_scale):
            gemlite_linear.meta_dtype = DType.FP32

        if(group_size == in_features):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 3
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 2

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

        del hqq_layer.W_q
        del hqq_layer.meta
        del hqq_layer
        torch.cuda.empty_cache()

        return self.from_weights(W_q=W_q, scales=scales, zeros=zeros, W_nbits=W_nbits, group_size=group_size, bias=bias)

############################################################################################################################################################
#BitNet
class A16W158:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = False

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        dtype = weight.dtype if(self.dtype is None) else self.dtype
        weight = weight.to(dtype)
        assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should floating point."
        W_q           = (weight + 1).to(torch.uint8)
        bias          = bias if bias is not None else None
        dtype         = weight.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32, device=self.device) * weight_scale.item() 

        gemlite_linear = GemLiteLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        if(self.fp32_scale is False):
            scales = scales.to(dtype=dtype)

        W_q = W_q.to(self.device)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)
        #post-scale
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 1 #weight-only
        return gemlite_linear

    def from_bitlinear(self, linear_layer):
        out_layer = self.from_weights(linear_layer.weight, linear_layer.weight_scale, linear_layer.bias)
        del linear_layer 
        torch.cuda.empty_cache()
        return out_layer

class A8W158:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = True

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data
        dtype = weight.dtype if(self.dtype is None) else self.dtype
        weight = weight.to(dtype)
        assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should floating point."
        W_q           = (weight + 1).to(torch.uint8)
        dtype         = weight.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32, device=self.device) * weight_scale.item() #[out_features, 1]

        input_dtype = DType.INT8

        gemlite_linear = GemLiteLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=True,
                        )

        if(self.fp32_scale is False):
            scales = scales.to(dtype=dtype)

        W_q = W_q.to(self.device)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)

        if(self.fp32_scale):
            gemlite_linear.meta_dtype = DType.FP32

        #post-scale 
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 3 #activations + weight

        return gemlite_linear

    def from_bitlinear(self, linear_layer):
        out_layer = self.from_weights(linear_layer.weight, linear_layer.weight_scale, linear_layer.bias)
        del linear_layer
        torch.cuda.empty_cache()
        return out_layer


############################################################################################################################################################
#Warm-up function:  
default_batch_sizes = sorted(list(set(M_MAPPING)))[::-1]
def warmup(shapes: list, batch_sizes: list = default_batch_sizes, W_nbits: list = [4], group_sizes: list = [64], mode: str = 'static', dtype = torch.float16):
    """
    * Warm-up for A16W4 with group_size=64
    warmup(shapes=[(4096, 4096)], W_nbits=[4], group_sizes=[64], mode='static')
    
    * warm-up for A8W8 int8 dynamic
    warmup(shapes=[(4096, 4096)], W_nbits=[8], mode='dynamic_int8')

    * Warm-up for A8W8 fp8 dynamic
    warmup(shapes=[(4096, 4096)], W_nbits=[8], mode='dynamic_fp8')
    """

    if min(W_nbits) < 8:
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError("the hqq package is missing. Please install via `pip install hqq`.")

    device = torch.device(torch.cuda.current_device())
    for W_nbit in W_nbits:
        for group_size in group_sizes:
            for shape in shapes:
                out_features, in_features = shape
                linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False, dtype=dtype, device=device)

                if(W_nbit == 8):
                    processor      = A16W8 if (mode == 'static') else (A8W8_fp8_dynamic if mode == 'dynamic_fp8' else A8W8_int8_dynamic)
                    gemlite_linear = processor(device=device).from_linear(linear)
                else:
                    processor      = A16Wn if (mode == 'static') else A8Wn_dynamic
                    quant_config   = BaseQuantizeConfig(nbits=W_nbit, group_size=group_size, axis=1)
                    linear         = HQQLinear(linear, quant_config=quant_config, compute_dtype=dtype, device=device)
                    gemlite_linear = processor(device=device).from_hqqlinear(linear)

                for batch_size in tqdm(batch_sizes):
                    _ = gemlite_linear(torch.randn((batch_size, in_features), dtype=dtype, device=device) / 100.)
                    torch.cuda.synchronize()

                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)

