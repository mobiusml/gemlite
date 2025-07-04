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

class QuantizerMXFP:
    def __init__(self, compute_dtype=torch.bfloat16, device="cuda:0"):
        self.compute_dtype = compute_dtype
        self.device = device
        self.fp4_values     = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=compute_dtype, device=device) 
        self.fp4_p_vals     = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=compute_dtype, device=device)
        self.fp4_thresholds = (self.fp4_p_vals[:-1] + self.fp4_p_vals[1:]) / 2
        self.max_val        = self.fp4_values.abs().max()

    def round_to_closest(self, tensor, W_nbits):
        if(W_nbits == 8):
            out = tensor.to(default_fp8).to(tensor.dtype)
        if(W_nbits ==4):
            out = self.fp4_p_vals[torch.searchsorted(self.fp4_thresholds, tensor.abs())]
            out *= tensor.sign()
        return out

    def to_index(self, W_q):
        assert W_q.is_floating_point(), "Input should be floating point fp4 values."
        return (W_q.view(-1, 1) == self.fp4_values.view(1 , -1)).to(torch.uint8).argmax(dim=1).to(torch.uint8).view(W_q.shape)

    @torch.compile(fullgraph=True)
    def quantize_mxfp(self, W: torch.Tensor, W_nbits: int, group_size: int = 32, window_size: int = 0, eps: float = 2 ** -30, index: bool = False) -> (torch.Tensor, torch.Tensor):

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= self.max_val

        if(window_size == 0):
            scales = (2 ** torch.round(torch.log2(ideal_scale)))
        else:
            initial_log2_scales = torch.round(torch.log2(ideal_scale))
            search_offsets = torch.arange(-window_size, window_size + 1, device=W.device, dtype=initial_log2_scales.dtype).view(1, -1)
            candidate_scales = torch.pow(2, initial_log2_scales + search_offsets)
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest(W_flat.unsqueeze(1) / candidate_scales.unsqueeze(-1), W_nbits=W_nbits)
            W_r_candidates = W_q_candidates * candidate_scales.unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True))

        W_q = self.round_to_closest(W_flat / scales, W_nbits=W_nbits)
        scales = scales.to(torch.float8_e8m0fnu)

        if(index):
            if(W_nbits == 8):
                W_q = W_q.to(default_fp8)
            if(W_nbits == 4):
                W_q = self.to_index(W_q)
        return W_q, scales

    @torch.compile(fullgraph=True)
    def quantize_nvfp4(self, W: torch.Tensor, group_size: int = 16, window_size: int = 0, eps: float = 1e-8, index: bool = False) -> (torch.Tensor, torch.Tensor):
        #Uses a greedy exhaustive search approach to look for the best scales in a neighborhood
        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= self.max_val

        if(window_size > 0):
            search_offsets = torch.arange(-window_size, window_size + 1, device=W.device, dtype=torch.int).view(1, -1)
            candidate_scales = (ideal_scale.to(torch.float8_e4m3fn).view(torch.int8) + search_offsets).clamp_(-128, 127).to(torch.int8)

            #Avoid nan in int8 range (-1, 127 as int8 as e4m3 nans)
            candidate_scales[candidate_scales==-1] = 1
            candidate_scales[candidate_scales==127] = 1
            candidate_scales = candidate_scales.view(torch.float8_e4m3fn).float()
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest(W_flat.unsqueeze(1) / candidate_scales.unsqueeze(-1))
            W_r_candidates = W_q_candidates * candidate_scales.unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True))
            scales = scales.to(torch.float8_e4m3fn)
        else:
            scales = ideal_scale.to(torch.float8_e4m3fn)

        W_q = self.round_to_closest(W_flat / scales.float())
        if(index):
            W_q = self.to_index(W_q)

        return W_q, scales

    def dequantize(self, W_q, scales, shape = None, dtype = None):
        if(W_q.dtype == torch.uint8): #from indices
            W_q = self.fp4_values[W_q.int()]

        group_size = W_q.numel() // scales.numel()
        out = (W_q.view([-1, group_size]).float() * scales.float())
        if(shape is not None):
            out = out.view(shape)
        return out.to(self.compute_dtype if dtype is None else dtype)

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
class A16W8: #INT8 weight-only channel-wise
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
class A16Wn: #8/4/2-bit weight-only as grouped "INT" / 8/4-bit as MXFP type 
    def __init__(self, device='cuda:0', dtype=None, packing_bitwidth=None, post_scale=default_post_scale):

        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.quantizer_mx = None

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT"):
        return self._from_weights(W_q, scales, zeros, W_nbits, group_size, bias, quant_type)

    def _from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT"):

        assert quant_type in ["INT", "MXFP"], f"Invalid quant_type. Got {quant_type}, valid values are INT, MXFP."

        if(quant_type == "MXFP"):
            assert W_nbits in [8, 4], "Unsupported W_nbit for MXFP quant_dtype."
            assert group_size == 32, "group_size should 32 for MXFP."

        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(quant_type == "INT"):
            dtype = scales.dtype if(self.dtype is None) else self.dtype
            scales = scales.to(dtype)
            assert scales.dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should floating point."
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

        elif(quant_type == "MXFP"):
            dtype = torch.bfloat16 if (self.dtype is None) else self.dtype
            if(dtype == torch.float16):
                gemlite_dtype = DType.MXFP16
            elif(dtype == torch.bfloat16):
                gemlite_dtype = DType.MXBF16
            else:
                raise Exception(f"Unsupported dtype for MXFP. Got {dtype}, supported [torch.float16, torch.bfloat16]")
            self.post_scale = False
            
            # #Handle indexing
            # if(W_q.is_floating_point()):
            #     W_q = to_fp4_indices(W_q)

            N, K = W_q.shape
            W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)#.float()

        in_features, out_features = W_q.shape[::-1] 

        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=dtype) if (scales is not None) else None
        zeros = zeros.to(device=self.device, dtype=dtype) if (zeros is not None) else None
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

        if(group_size == in_features and quant_type == "INT"):
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

        out_layer = self._from_weights(W_q=W_q, scales=scales, zeros=zeros, W_nbits=W_nbits, group_size=group_size, bias=bias, quant_type="INT")
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

    def mxfp_from_linear(self, linear_layer, W_nbits, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = QuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        W_q, scales = self.quantizer_mx.quantize_mxfp(W, W_nbits=W_nbits, window_size=0, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self._from_weights(W_q=W_q, scales=scales, zeros=None, W_nbits=W_nbits, group_size=group_size, bias=bias, quant_type="MXFP")

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer


#Alias
A16W8_CHANNELWISE_INT = A16W8

class A16Wn_HQQ_INT(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_weights(self, W_q, scales, zeros, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(W_q=W_q, scales=scales, zeros=zeros, W_nbits=self.W_nbits, group_size=group_size, bias=bias, quant_type="INT")

class A16W8_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype,  W_nbits=8)

class A16W4_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)

class A16W2_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=2)

class A16W1_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=1)

class A16Wn_MXFP(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_weights(self, W_q, scales, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(W_q=W_q, scales=scales, zeros=None, W_nbits=self.W_nbits, group_size=group_size, bias=bias, quant_type="MXFP")

    def from_linear(self, linear_layer, del_orig=True):
        return super().mxfp_from_linear(linear_layer=linear_layer, W_nbits=self.W_nbits, del_orig=del_orig)

class A16W8_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=8)

class A16W4_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)

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
        assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight.dtype, should be floating point."

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

