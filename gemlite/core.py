# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import torch
from torch import Tensor
import numpy as np
from enum import Enum
import math, json, os
import warnings, random
from typing import List, Union, Tuple, Callable
import logging
from functools import partial
    
#Dtypes
from .dtypes import *

# Triton
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.testing import do_bench, do_bench_cudagraph
from .triton_kernels import *
from .triton_kernels.utils import gpu_supports_float16_acc
from .triton_kernels import utils
from .bitpack import pack_weights_over_cols_triton, pack_weights_over_cols_torch

import threading
FILE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

###################################################################################################################################
# Triton backend
###################################################################################################################################
GEMLITE_ACC_DTYPE = {
    DType.FP16: DType.FP16 if gpu_supports_float16_acc() else DType.FP32,
    DType.BF16: DType.FP32,
    DType.FP32: DType.FP32,
    DType.FP8: DType.FP32,
    DType.FP8e5: DType.FP32,
    DType.INT8: DType.INT32,
}

GEMLITE_TRITON_KERNELS = [
    gemv_A16fWnO16f,
    gemv_revsplitK_A16fWnO16f,
    gemv_splitK_A16fWnO16f,
    gemm_splitK_A16fWnO16f, #gemm_splitK_A16fWnO16f / gemm_splitK_persistent_A16fWnO16f
    gemm_A16fWnO16f,
]

GEMLITE_TRITON_MAPPING       = {kernel.matmul_type : kernel for kernel in GEMLITE_TRITON_KERNELS}
GEMLITE_MATMUL_TYPES         = [kernel.matmul_type for kernel in GEMLITE_TRITON_KERNELS]
GEMLITE_MATMUL_TYPES_MAPPING = {GEMLITE_MATMUL_TYPES[i]: i for i in range(len(GEMLITE_MATMUL_TYPES))}
GEMLITE_TRITON_CONFIG_CACHE  = {} #Global config cache for all the kernels
_GROUP_SIZE_WARNED           = False

###################################################################################
#Utils

#Main function to cache kernel autotune config
def cache_kernel_config(kernel, num_keys):
    kernel_cache = kernel.cache
    k_config = {}
    if(len(kernel_cache) > 0):
        for k in kernel_cache:
            key    = list(k[:num_keys])
            key[0] = utils.get_closest_m(key[0]) #restrict batch-size
            key    = tuple(key)
            k_config[str(key)] = kernel_cache[k].all_kwargs()
    return k_config

#Set M autotune logic
def set_autotune_setting(fct): #fct = lambda M: M for max-autotune
    utils.get_closest_m = fct 

#set default packing format
def set_packing_bitwidth(packing_bitwidth : int):
    GemLiteLinearTriton.PACKING_BITWIDTH = packing_bitwidth

#Set accumulation dtype
def set_acc_dtype(dtype):
    global GEMLITE_ACC_DTYPE
    assert dtype in [DType.FP16, DType.FP32], "Invalid dtype (should be DType.FP16 or DType.FP32)."
    GEMLITE_ACC_DTYPE[DType.FP16] = dtype

#Return the default gemv kernel to use for M==1
def get_default_gemv(W_nbits: int) -> str:
    return 'GEMV_REVSPLITK' if (W_nbits < 8) else 'GEMV_SPLITK'

#matmul type selection logic
def get_matmul_type(batch_size: int, W_nbits: int):
    if batch_size > 64:
        return "GEMM"
    if batch_size > 1:
        return "GEMM_SPLITK"
    else:
        return get_default_gemv(W_nbits)

#######################################################################################################################
#Activation scaling

#Get max val based on compute type
def get_max_val(compute_dtype: torch.dtype) -> float:
    if(compute_dtype.is_floating_point):
        max_val = torch.finfo(compute_dtype).max
    else:
        max_val = torch.iinfo(compute_dtype).max
    return max_val

#Main activation scaling functions
@torch.compile(fullgraph=True)
def scale_activations_per_token_torch(x: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True) -> Tuple[Tensor, Tensor]:
    max_val = get_max_val(w_dtype)
    if(fp32_scale):
        x = x.to(torch.float32, copy=False)
    x_shape  = x.shape
    out_x    = x.view(-1, x.shape[-1]) 
    scales_x = torch.abs(out_x).amax(axis=1, keepdim=True)
    # if(fp32_scale):
    #     scales_x = scales_x.to(torch.float32)
    scales_x.div_(max_val)
    out_x = x / scales_x
    out_x.round_()    
    out_x = out_x.to(dtype=w_dtype)
    return out_x.view(x_shape), scales_x

@triton.jit
def scale_activations_per_token_kernel(
    x_ptr, scale_ptr, y_ptr, 
    M, K,
    stride_m, stride_k, stride_sm, 
    UNROLL: tl.constexpr,
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_K: tl.constexpr 
):
    pid_m = tl.program_id(0) * UNROLL
    pid_k = tl.program_id(1)

    offs_k  = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    for m in range(UNROLL):    
        mask    = ((offs_m < M)[:, None] & (offs_k < K)[None, :]).to(tl.int1)
        in_ptrs = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        x = tl.load(x_ptr + in_ptrs, mask=mask, other=0.0)
        if(fp32_scale):
            x = x.to(tl.float32)

        scales_x = tl.max(tl.abs(x), axis=1, keep_dims=True)
        scales_x /= max_val
        x /= scales_x
        x = libdevice.round(x)

        tl.store(scale_ptr + offs_m[:, None] * stride_sm, scales_x)
        tl.store(y_ptr + in_ptrs, x, mask=mask)
        offs_m += BLOCK_M 


def scale_activations_per_token_triton(x: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True) -> Tuple[Tensor, Tensor]:
    max_val = get_max_val(w_dtype)
    x_shape = x.shape
    x       = x.view(-1, x.shape[-1]) 
    M, K    = x.shape
    scales = torch.empty((M, 1), dtype=torch.float32 if fp32_scale else x.dtype, device=x.device)
    y      = torch.empty((M, K), dtype=w_dtype, device=x.device)

    UNROLL  = 1 #max(1, M // 128)
    BLOCK_M = 1 
    BLOCK_K = triton.next_power_of_2(K)
    grid    = (triton.cdiv(M, BLOCK_M * UNROLL), triton.cdiv(K, BLOCK_K)) 

    scale_activations_per_token_kernel[grid](
        x, scales, y,
        M, K,
        x.stride(0), 
        x.stride(1),
        scales.stride(0),
        max_val=max_val,
        fp32_scale=fp32_scale,
        UNROLL=UNROLL,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_stages=1,
        num_warps=4,
    )

    return y.view(x_shape), scales

scale_activations = scale_activations_per_token_triton
#######################################################################################################################

#Main functional forward call
@torch.library.custom_op("gemlite::forward_functional", mutates_args=())
def forward_functional(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1, #-1: auto, >=0: manual
) -> Tensor:
    
    scaled_activations = bool(meta_args[0])
    data_contiguous = bool(meta_args[1])
    W_nbits = meta_args[1]
    out_features = tensor_args[0].shape[1]

    if not x.is_contiguous():
        x = x.contiguous()
    
    #Dynamic activation quantization
    if(scaled_activations):
        compute_dtype = DTYPE_TO_TORCH[meta_args[5]] #input_dtype.value
        x, scales_x = scale_activations(x, w_dtype=compute_dtype)
    else:
        x, scales_x = x, None

    out_shape = x.shape[:-1] + (out_features,)
    x = x.view(-1, x.shape[-1])

    if(matmul_type >= 0):
        matmul_type_str = GEMLITE_MATMUL_TYPES[matmul_type]
    else:
        matmul_type_str = get_matmul_type(x.shape[0], W_nbits) #batch_size, W_nbits

    out = GEMLITE_TRITON_MAPPING[matmul_type_str].forward(x, *tensor_args, scales_x, *meta_args[1:-1], data_contiguous).view(out_shape)

    if bias is not None:
        out += bias

    return out

@torch.library.register_fake("gemlite::forward_functional")
def forward_functional_fake(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1,
) -> Tensor:
    out_features = tensor_args[0].shape[1]
    return torch.empty(x.shape[:-1] + (out_features,), device=x.device, dtype=x.dtype)

#######################################################################################################################
#Main class
class GemLiteLinearTriton(torch.nn.Module):
    SUPPORTED_BITS_TRITON = [1, 2, 4, 8, 16]
    SUPPORTED_DTYPES      = [DType.FP16, DType.BF16, DType.FP32, DType.FP8, DType.FP8e5, DType.INT8]
    MIN_SIZE              = 32
    PACKING_BITWIDTH      = 32 #Default packing bitwidth

    def __init__(
        self,
        W_nbits = 4,
        group_size = 64,
        in_features = None,
        out_features = None,
        input_dtype = DType.FP16,
        output_dtype = DType.FP16,
        acc_dtype = None,
        scaled_activations = False,
    ):
        global _GROUP_SIZE_WARNED

        super().__init__()

        if W_nbits not in GemLiteLinearTriton.SUPPORTED_BITS_TRITON:
            raise NotImplementedError("Only " + str(GemLiteLinearTriton.SUPPORTED_BITS_TRITON) + " W_nbits are supported.")

        if (in_features is not None and out_features is not None):
            if (in_features % GemLiteLinearTriton.MIN_SIZE != 0) or (in_features % group_size !=0 if (group_size is not None) else False):
                raise NotImplementedError("Invalid input shapes: " + str(in_features) + ' , ' + str(out_features) + '. in_features should be divisible by 32 or the group_size')

        #Warning: Input dtype should be the same as dequantize() weights dtype.
        if input_dtype not in GemLiteLinearTriton.SUPPORTED_DTYPES:
            raise NotImplementedError("Unsupport input dtype: " + str(input_dtype))

        if(group_size is not None):
            if(group_size < 32):
                raise NotImplementedError("Only group_size >= 32 is supported.")

        group_size = 1 if (group_size is None) else group_size

        self.in_features  = in_features
        self.out_features = out_features
        self.orig_shape   = (out_features, in_features)
        self.W_nbits      = W_nbits
        self.group_size   = group_size
        self.unpack_mask  = 2**self.W_nbits - 1
        self.elements_per_sample = None
        self.signature = (in_features, out_features, W_nbits, group_size)

        self.input_dtype   = input_dtype
        self.output_dtype  = output_dtype
        self.compute_dtype = DTYPE_TO_TORCH[self.input_dtype.value]
        self.meta_dtype    = input_dtype
        
        #Accumulation
        self.acc_dtype = GEMLITE_ACC_DTYPE[self.input_dtype] if(acc_dtype is None) else acc_dtype

        #Scales activations
        if(self.compute_dtype in [torch.float16, torch.bfloat16, torch.float32]):
            self.scaled_activations = False
        else:
            self.scaled_activations = scaled_activations

        #Default forward        
        self.forward = self.forward_auto_no_warmup

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.W_q        = state_dict.pop("W_q", None)
        self.bias       = state_dict.pop("bias", None)
        self.scales     = state_dict.pop("scales", None)
        self.zeros      = state_dict.pop("zeros", None)
        self.metadata   = state_dict.pop("metadata", None)
        self.orig_shape = state_dict.pop("orig_shape", None)

        self.metadata   = [v.item() for v in self.metadata]
        self.orig_shape = (v.item() for v in self.orig_shape)

        (self.scaled_activations,
        self.W_nbits,
        self.group_size,
        self.unpack_mask,
        self.elements_per_sample,
        self.input_dtype,
        self.output_dtype,
        self.acc_dtype,
        self.meta_dtype,
        self.channel_scale_mode,
        self.W_group_mode,
        self.data_contiguous) = self.metadata

        self.input_dtype  = DType(self.input_dtype)
        self.output_dtype = DType(self.output_dtype)
        self.acc_dtype    = DType(self.acc_dtype)
        self.meta_dtype   = DType(self.meta_dtype)

        self.out_features, self.in_features = self.orig_shape
        self.compute_dtype = DTYPE_TO_TORCH[self.input_dtype.value]
        self.scaled_activations = bool(self.scaled_activations)
        self.data_contiguous = bool(self.data_contiguous)

    #Make sure to feed UINT8 W_q for packing
    def pack(
        self,
        W_q: Tensor,
        scales: Tensor,
        zeros: Union[Tensor, int],
        bias: Union[Tensor, None] = None,
        fma_mode: bool = True,
        contiguous: Union[int, None] = None,
        packing_bitwidth: Union[int, None] = None,
    ):
        #Set packing bitwidth
        if(packing_bitwidth is None):
            packing_bitwidth = GemLiteLinearTriton.PACKING_BITWIDTH

        #Unpacked weights
        self.W_q = None
        if(W_q.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]):
            if(W_q.dtype in [torch.float16, torch.bfloat16]): 
                assert self.W_nbits == 16, "Invalid fp16 weights."
            else: 
                assert self.W_nbits == 8, "Invalid 8-bit weights."

            self.W_q = W_q.t() #row-major
            self.elements_per_sample = 1

            if(contiguous is None): contiguous = False

        if W_q.dtype == torch.uint8:  # Packed weigths
            _pack_weights_over_cols = pack_weights_over_cols_triton if (W_q.device.type == "cuda") else pack_weights_over_cols_torch

            self.W_q, self.elements_per_sample = _pack_weights_over_cols(
                W_q.view(self.orig_shape),
                W_nbits=self.W_nbits,
                packing_bitwidth=packing_bitwidth,
                transpose=True,
            )  # Over-K
            if contiguous is None:
                contiguous = True

        if(self.W_q is None):
            raise Exception('Weights were not packed, please check your W_q.dtype')

        #Bias / device
        self.bias   = None if (bias is None) else torch.nn.Parameter(bias.to(device=self.W_q.device, dtype=self.compute_dtype))
        self.device = self.W_q.device

        #initial values
        self.W_group_mode       = -1
        self.channel_scale_mode = 0

        #FP16 x FP16 / FP8 x FP8 / INT8 x INT8 - no meta-data case 
        if((scales is None) and (zeros is None)):
            self.zeros  = None
            self.scales = None
            self.W_group_mode = 0
            self.channel_scale_mode = 2 if self.scaled_activations else 0 
            
        #The rest of the use-cases require some kind of meta-data
        if(scales is not None):
            self.scales = scales.view((self.out_features, -1)).t()
        else:
            self.scales = None

        #channel-wise scaling 
        self.meta_is_channelwise = False if(self.scales is None) else self.scales.numel() == self.out_features 

        #Symmetric no shift
        if(zeros is None):  
            self.zeros = None
            self.W_group_mode = 2 if(self.scales is not None) else 0
        else:
            #Asymmetric or Symmetric with shift
            if(isinstance(zeros, torch.Tensor)):
                if(fma_mode and (self.meta_is_channelwise is False)): #W ~ Wq * scales + zeros
                    self.zeros = (-zeros.float()*scales.float()).to(zeros.dtype).view((self.out_features, -1)).t()
                    self.W_group_mode = 4
                else: #W ~ (Wq - zeros) * scales
                    self.zeros = zeros.view((self.out_features, -1)).t()
                    self.W_group_mode = 3 
            else: #Integer
                self.zeros = int(zeros) 
                if(self.scales is not None):
                    self.W_group_mode = 3 #Symmetric with shift
                else:
                    self.W_group_mode = 1 #Shift only with integer

        assert self.W_group_mode > -1, "Invalid scales/zeros settings."

        #weight-only
        if((self.scaled_activations == False) and (self.meta_is_channelwise == True)):
            self.channel_scale_mode = 1
            self.W_group_mode       = 1 if(self.zeros is not None) else 0 #only with fma_mode=False

        #activation-only
        if((self.scaled_activations == True) and (self.meta_is_channelwise == False)):
            self.channel_scale_mode = 2

        #weight + activation mode
        if((self.scaled_activations == True) and (self.meta_is_channelwise == True)):
             self.channel_scale_mode = 3
             self.W_group_mode       = 1 if(self.zeros is not None) else 0 #only with fma_mode=False

        if(self.channel_scale_mode in [1, 3]):
            assert self.W_group_mode not in [3, 4], "Can't use channel_scale_mode with W_group_mode == 3 or 4."

        # if(self.input_dtype == DType.INT8):
        #     assert self.W_group_mode in [1], "Only channel-wise symmetric quantization is supported for INT8 inputs."

        #Dummy values 
        if(isinstance(self.zeros, int)): #Union[Tensor, int] not supported by custom op
            self.zeros = torch.tensor(self.zeros, dtype=torch.int32, device=self.device)
        if(self.zeros is None):
            self.zeros = torch.tensor([[]], dtype=torch.int32, device=self.device)
        if(self.scales is None):
            self.scales = torch.tensor([[]], dtype=torch.int32, device=self.device)

        if(self.scales is not None):
            self.meta_dtype = TORCH_TO_DTYPE[self.scales.dtype]

        #Force contiguous
        if(contiguous):
            self.data_contiguous = True
            self.W_q = self.W_q.contiguous()
        else:
            self.data_contiguous = False

        if(isinstance(self.scales, torch.Tensor)):
            self.scales = self.scales.contiguous()
        if(isinstance(self.zeros, torch.Tensor)):
            self.zeros = self.zeros.contiguous()

        #Register buffers
        self.W_q        = torch.nn.Parameter(self.W_q,   requires_grad=False)
        self.scales     = torch.nn.Parameter(self.scales,requires_grad=False)
        self.zeros      = torch.nn.Parameter(self.zeros, requires_grad=False)
        self.metadata   = torch.nn.Parameter(torch.tensor(self.get_meta_args(), device=self.device, dtype=torch.int32), requires_grad=False)
        self.orig_shape = torch.nn.Parameter(torch.tensor([self.out_features, self.in_features], device=self.device, dtype=torch.int32), requires_grad=False)
        return self

    #Return the main arguments
    def get_tensor_args(self):
        return [self.W_q, self.scales, self.zeros]

    def get_meta_args(self):
        return [int(self.scaled_activations),
                self.W_nbits,
                self.group_size,
                self.unpack_mask,
                self.elements_per_sample,
                self.input_dtype.value,
                self.output_dtype.value,
                self.acc_dtype.value,
                self.meta_dtype.value,
                self.channel_scale_mode,
                self.W_group_mode,
                int(self.data_contiguous),
                ]

    # #Main manual call
    def forward_manual(self, x: Tensor, matmul_type: str="GEMM") -> Tensor:
        return forward_functional(
            x,
            self.bias,
            self.get_tensor_args(),
            self.get_meta_args(),
            GEMLITE_MATMUL_TYPES_MAPPING[matmul_type],
        )

    # #Main auto call without exhaustive search
    def forward_auto_no_warmup(self, x: Tensor) -> Tensor:
        return forward_functional(
            x,
            self.bias,
            self.get_tensor_args(),
            self.get_meta_args(),
        )

    @staticmethod
    def cache_config(filename: str):
        global GEMLITE_TRITON_CONFIG_CACHE
        #Load existing cache if available
        try:
            with FILE_LOCK, open(filename, 'r') as json_file:
                config = json.load(json_file)
        except:
            config = {}
    
        #Can't use GEMLITE_TRITON_MAPPING for some reason kernel.cache is empty
        _GEMLITE_TRITON_MAPPING = {}
        from .triton_kernels.gemv_A16fWnO16f_int32packing import gemv_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV'] = gemv_A16fWnO16f

        from .triton_kernels.gemv_revsplitK_A16fWnO16f_int32packing import gemv_revsplitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV_REVSPLITK'] = gemv_revsplitK_A16fWnO16f

        from .triton_kernels.gemv_splitK_A16fWnO16f_int32packing import gemv_splitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV_SPLITK'] = gemv_splitK_A16fWnO16f

        from .triton_kernels.gemm_splitK_A16fWnO16f_int32packing import gemm_splitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMM_SPLITK'] = gemm_splitK_A16fWnO16f

        from .triton_kernels.gemm_A16fWnO16f_int32packing import gemm_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMM'] = gemm_A16fWnO16f

        for name in _GEMLITE_TRITON_MAPPING:
            if(name not in config): 
                config[name] = {}
            
            if(name in GEMLITE_TRITON_CONFIG_CACHE):
                config[name].update(GEMLITE_TRITON_CONFIG_CACHE[name])

            config[name].update(cache_kernel_config(_GEMLITE_TRITON_MAPPING[name].kernel, 5)) #5: len(prune_keys)

        #Save combined cache
        with FILE_LOCK, open(filename, "w") as json_file: 
            json.dump(config, json_file)

    @staticmethod
    def load_config(filename: str, print_error: bool = True, overwrite: bool = False):
        global GEMLITE_TRITON_CONFIG_CACHE
        if(filename is None):
            return False
        try:
            with FILE_LOCK, open(filename, 'r') as json_file:
                config = json.load(json_file)
                if(overwrite):
                    GEMLITE_TRITON_CONFIG_CACHE = config
                else:
                    for name in config:
                        if(name not in GEMLITE_TRITON_CONFIG_CACHE): 
                            GEMLITE_TRITON_CONFIG_CACHE[name] = {}
                        GEMLITE_TRITON_CONFIG_CACHE[name].update(config[name])

        except Exception as e:
            if(print_error):
                logger.error(f"Failed to load the cache file '{filename}': {e}")
            return False
        return True 

    @staticmethod
    def reset_config():
        global GEMLITE_TRITON_CONFIG_CACHE
        GEMLITE_TRITON_CONFIG_CACHE = {}

###################################################################################################################################
###################################################################################################################################
GemLiteLinear = GemLiteLinearTriton  # Triton by default

#Setting default config
def get_default_cache_config():
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/")
    
    def get_tags(path):
        return [f.split('.')[0] for f in os.listdir(path)]

    name = torch.cuda.get_device_properties(0).name.lower().replace(' ', '_')
    tags = get_tags(root_path)
    tags.sort(key=len, reverse=True)

    selected_tag = None
    for tag in tags:
        if(tag in name):
            selected_tag = os.path.join(root_path, tag + '.json')
            break
    
    return selected_tag

selected_tag = get_default_cache_config()
if(GemLiteLinear.load_config(selected_tag)):
    logger.warning('Loaded ' + selected_tag + ' config.')
