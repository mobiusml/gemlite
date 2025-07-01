# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************

import torch
import gc
import time
from tqdm import tqdm
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
from gemlite.triton_kernels.utils import M_MAPPING
from .triton_kernels.utils import IS_HIP

if IS_HIP:
    default_fp8 = torch.float8_e4m3fnuz #AMD
    #default_fp8 = torch.float8_e5m2fnuz #AMD
    #default_fp8 = torch.float8_e5m2 #AMD - fp16 emulated
    default_post_scale = True
else:
    default_fp8 = torch.float8_e4m3fn #Nvidia 
    #default_fp8 = torch.float8_e5m2 #Nvidia
    default_post_scale = False
############################################################################################################################################################
#Clean-up
def cleanup_linear(linear_layer, del_orig=True):
    if(del_orig):
        for attr in ['weight', 'bias', 'weight_scale', 'W_q', 'meta']:
            if(hasattr(linear_layer, attr)):
                setattr(linear_layer, attr, None)
    torch.cuda.empty_cache()

#Replaces all linear layers with the corresponding processor
def patch_model(model, device, processor, skip_modules=[]):
    #Loadd HQQLinear when needed
    if(processor in [A16Wn, A8Wn_dynamic]):
        from hqq.core.quantize import HQQLinear
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ

    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules):
        layer = layer.to(device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(isinstance(layer, torch.nn.Linear)):
                return processor(device=device).from_linear(layer)
            elif(isinstance(layer, HQQLinear)):
                return processor(device=device).from_hqqlinear(layer)
            else:
                return layer

    #Replaces linear layers
    def _patch_linearlayers(model, fct, device, skip_modules):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                setattr(model, name, fct(layer, device, skip_modules))
            else:
                _patch_linearlayers(layer, fct, device, skip_modules)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules)

    #Clean-up
    torch.cuda.empty_cache()
    gc.collect()

#16-bit activations / 8-bit weigths
class A16W8: #INT8 weights
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=False):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            w_dtype, max_val = torch.int8, 127
            dtype = weight.dtype if(self.dtype is None) else self.dtype
            assert dtype in [torch.float16, torch.bfloat16, torch.float32], f"Invalid weight dtype, should be floating point, got {dtype}"
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            if(w_dtype.is_floating_point):
                W_q = (weight / scales).to(w_dtype)
            else:
                W_q = (weight / scales).round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.dtype in [torch.int8], f"Invalid weight.dtype, should be int8, got {weight.dtype}"
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device)
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)

        #Pre-scaling
        gemlite_linear.W_group_mode       = 2
        gemlite_linear.channel_scale_mode = 0
        return gemlite_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

#FP16 activations / Wn packed weights
class A16Wn:
    def __init__(self, device='cuda:0', dtype=None, packing_bitwidth=None, quant_type="INT", post_scale=default_post_scale):

        assert quant_type in ["INT", "MXFP"], f"Invalid quant_type. Got {quant_type}, valid values are INT, MXFP."

        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.quant_type = quant_type
        if(quant_type == "MXFP"):
            packing_bitwidth = 8
        self.packing_bitwidth = packing_bitwidth

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None):
        if(self.quant_type == "MXFP"):
            assert W_nbits in [8, 4], "Unsupported W_nbit for MXFP quant_dtype."
            assert group_size == 32, "group_size should 32 for MXFP."

        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.quant_type == "INT"):
            dtype = scales.dtype if(self.dtype is None) else self.dtype
            scales = scales.to(dtype)
            assert scales.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should floating point."
        else:
            dtype = torch.bfloat16 if (self.dtype is None) else self.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        in_features, out_features = W_q.shape[::-1]

        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=dtype) if (scales is not None) else None
        zeros = zeros.to(device=self.device, dtype=dtype) if (zeros is not None) else None
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        #MXFP4 / NVFP4 conversion to indices TODO
        if(self.quant_type == "MXFP" and W_q.is_floating_point()):
            #W_q -> to indices
            pass

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size,  
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        gemlite_linear.pack(W_q, scales, zeros, bias=bias, packing_bitwidth=self.packing_bitwidth)

        if(self.quant_type == "MXFP"): #[K//32, N]
            gemlite_linear.W_q.data    = gemlite_linear.W_q.data.contiguous() #.T.contiguous().T
            gemlite_linear.scales.data = gemlite_linear.scales.data.to(torch.float8_e8m0fnu).view(torch.uint8)
            gemlite_linear.scales.data = gemlite_linear.scales.data.T#.contiguous()
            gemlite_linear.W_group_mode = 2 #TODO - USE ANOTHER W_GROU)_MODE ? or another type_id for autotune
            gemlite_linear.channel_scale_mode = 0

        if(group_size == in_features and self.dtype == "INT"):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 1
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 0

        return gemlite_linear

    def from_hqqlinear(self, hqq_layer, del_orig=True):
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

        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights(W_q=W_q, scales=scales, zeros=zeros, W_nbits=W_nbits, group_size=group_size, bias=bias)
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

############################################################################################################################################################
#8-bit dynamic activations / 8-bit weights
class A8W8_dynamic:
    def __init__(self, device='cuda:0', dtype=None, fp8=False, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.fp8): #FP8
            w_dtype, input_dtype, max_val = self.fp8, TORCH_TO_DTYPE[self.fp8], torch.finfo(self.fp8).max
        else: #INT8
            w_dtype, input_dtype, max_val = torch.int8, DType.INT8, 127

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            dtype = weight.dtype if(self.dtype is None) else self.dtype
            assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight dtype, should be floating point."
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            if(w_dtype.is_floating_point):
                W_q = (weight / scales).to(w_dtype)
            else:
                W_q = (weight / scales).round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.dtype.itemsize == 1, "Invalid weight.dtype, should be 8-bit."
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype 
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device) 
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
        
        scales = scales.to(torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)

        #Post-scaling
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 3
        return gemlite_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W8_int8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fp8 = False

class A8W8_fp8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None, fp8=default_fp8):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8

############################################################################################################################################################
#FP8 dynamic activations / W4 packed weights
class A8Wn_dynamic(A16Wn):
    def __init__(self, device='cuda:0', packing_bitwidth=None, dtype=None, post_scale=default_post_scale, fp8=default_fp8, fp32_scale=False):
        super().__init__()
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.fp8 = fp8
        self.fp32_scale = fp32_scale

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None):
        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.dtype is None):
            dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
        else:
            dtype = self.dtype 

        assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should be floating point."
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        input_dtype = TORCH_TO_DTYPE[self.fp8]
            
        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=torch.float32 if self.fp32_scale else dtype) if (scales is not None) else None
        zeros = zeros.to(device=self.device, dtype=dtype) if (zeros is not None) else None
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
        
        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights(W_q=W_q, scales=scales, zeros=zeros, W_nbits=W_nbits, group_size=group_size, bias=bias)
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer
############################################################################################################################################################
#BitNet
class A16W158:
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype
        assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(dtype=dtype, device=self.device)
        W_q           = (weight + 1).to(torch.uint8)
        bias          = bias if bias is not None else None
        dtype         = weight.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32) * weight_scale.item() 

        gemlite_linear = GemLiteLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)
        #post-scale
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 1 #weight-only
        return gemlite_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, weight_scale=linear_layer.weight_scale, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W158:
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype
        assert dtypee in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(device=self.device, dtype=dtype)
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

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)

        #post-scale 
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 3 #activations + weight

        return gemlite_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, weight_scale=linear_layer.weight_scale, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
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

