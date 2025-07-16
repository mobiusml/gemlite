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
from .triton_kernels.utils import gpu_supports_float16_acc, IS_HIP, get_num_SMs, next_power_of_2
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
    DType.FP8e4: DType.FP32,
    DType.FP8e4nuz: DType.FP32,
    DType.FP8e5: DType.FP32, 
    DType.FP8e5nuz: DType.FP32,
    DType.INT8: DType.INT32,
    DType.MXFP16: DType.FP32,
    DType.MXBF16: DType.FP32,
    DType.MXFP8: DType.FP32,
    DType.MXFP4: DType.FP32,
    DType.NVFP4: DType.FP32,
}

GEMLITE_TRITON_KERNELS = [
    gemv,
    gemv_revsplitK,
    gemv_splitK,
    gemm_splitK, #gemm_splitK / gemm_splitK_persistent
    gemm,
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
    if IS_HIP:
        return 'GEMV_REVSPLITK' if (W_nbits < 8) else 'GEMV_SPLITK'
    else:
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
def scale_activations_per_token_torch(tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True) -> Tuple[Tensor, Tensor]:
    max_val = get_max_val(w_dtype)
    if fp32_scale:
        tensor = tensor.to(torch.float32, copy=False)
    out_shape = tensor.shape
    out = tensor.view(-1, tensor.shape[-1])
    scales = torch.abs(out).amax(axis=1, keepdim=True)
    # if(fp32_scale):
    #     scales = scales.to(torch.float32)
    scales.div_(max_val)
    out = tensor / scales

    if not w_dtype.is_floating_point:
        out.round_()

    out = out.to(dtype=w_dtype)
    return out.view(out_shape), scales


@triton.jit
def round_triton_nvidia(tensor):
    return libdevice.round(tensor)

@triton.jit
def round_triton_amd(tensor):
    return libdevice.floor(tensor + 0.50)

if IS_HIP:
    round_triton = round_triton_amd
else:
    round_triton = round_triton_nvidia

@triton.jit
def scale_activations_per_token_kernel(
    tensor_ptr, scale_ptr, y_ptr, 
    M, K,
    stride_m, stride_k, stride_sm,
    ROUND: tl.constexpr, 
    UNROLL: tl.constexpr,
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0) * UNROLL
    pid_k = tl.program_id(1)

    offs_k  = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    for m in range(UNROLL):
        mask = ((offs_m < M)[:, None] & (offs_k < K)[None, :]).to(tl.int1)
        in_ptrs = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        tensor = tl.load(tensor_ptr + in_ptrs, mask=mask, other=0.0)
        if fp32_scale:
            tensor = tensor.to(tl.float32)

        scales_x = tl.max(tl.abs(tensor), axis=1, keep_dims=True)
        scales_x /= max_val
        tensor /= scales_x

        if ROUND:
            tensor = round_triton(tensor)

        tl.store(scale_ptr + offs_m[:, None] * stride_sm, scales_x)
        tl.store(y_ptr + in_ptrs, tensor, mask=mask)
        offs_m += BLOCK_M

def scale_activations_per_token_triton(tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True) -> Tuple[Tensor, Tensor]:
    max_val = get_max_val(w_dtype)
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    scales = torch.empty(
        (M, 1), dtype=torch.float32 if fp32_scale else tensor.dtype, device=tensor.device
    )
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    UNROLL = 1  # max(1, M // 128)
    BLOCK_M = 1
    BLOCK_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_M * UNROLL), triton.cdiv(K, BLOCK_K))

    ROUND = not w_dtype.is_floating_point

    scale_activations_per_token_kernel[grid](
        tensor,
        scales,
        y,
        M,
        K,
        tensor.stride(0),
        tensor.stride(1),
        scales.stride(0),
        max_val=max_val,
        fp32_scale=fp32_scale,
        ROUND=ROUND,
        UNROLL=UNROLL,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_stages=1,
        num_warps=4,
    )

    return y.view(x_shape), scales

@triton.jit
def next_power_of_2_log_triton(val, eps: tl.constexpr):
    scales_log2 = tl.ceil(tl.log2(val)).to(tl.int32)
    scales = tl.where(scales_log2 >= 0, 1 << scales_log2, 1.0 / (1 << (-scales_log2)))
    scales = tl.maximum(scales, eps)
    return scales, scales_log2

@triton.jit
def next_power_of_2_logapprox_triton(val, eps: tl.constexpr):
    scales_log2 = tl.inline_asm_elementwise(
        """
        {
        lg2.approx.f32 $1, $1;
        cvt.rpi.f32.f32 $1, $1;
        cvt.rzi.s32.f32 $0, $1;
        }
        """,
        "=r,r",
        [val],
        dtype=tl.int32, 
        is_pure=True,
        pack=1
    )

    scales = tl.where(scales_log2 >= 0, 1 << scales_log2, 1.0 / (1 << (-scales_log2)))
    scales = tl.maximum(scales, eps)
    return scales, scales_log2

@triton.jit
def next_power_of_2_bitwise_triton(val, eps: tl.constexpr):
    xi = tl.cast(val, tl.uint32, bitcast=True)
    exp  = (xi >> 23) & 0xFF
    mant = xi & 0x7FFFFF
    exp += tl.where(mant != 0, 1, 0)
    scales_log2 = exp - 127
    exp = tl.minimum(exp, 254)
    yi = exp << 23
    scales = tl.cast(yi, tl.float32, bitcast=True)
    scales = tl.maximum(scales, eps)
    return scales, scales_log2

next_power_of_2_triton = next_power_of_2_bitwise_triton

@torch.compile(fullgraph=True)
def scale_activations_mxfp8_torch(tensor: Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn) -> Tuple[Tensor, Tensor]:
    group_size = 32
    eps: float = 2 ** -30
    max_val = get_max_val(w_dtype) #max_val == 6 if W_nbits == 4 else 448

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps) 

    W_q = (W_flat / scales).clamp_(-max_val, max_val).to(torch.float8_e4m3fn)
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    W_q = W_q.view(orig_shape)
    scales = scales.to(torch.float8_e8m0fnu).view(torch.uint8).view(post_pad_shape[0], post_pad_shape[1] // group_size)

    return W_q, scales

@triton.jit
def scale_activations_mxfp8_triton_kernel(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    E,
    eps: tl.constexpr,
    UNROLL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0) * UNROLL

    for m in range(UNROLL):
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = (offs < E).to(tl.int1)
        tensor = tl.load(tensor_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor)) / 6., eps)

        out = (tensor / scales).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, out)
        tl.store(scales_ptr + pid, scales_log2 + 127)

        pid += 1

def scale_activations_mxfp8_triton(tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size = 32
    eps: float = 2 ** -30
    max_val = get_max_val(w_dtype)
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    E = tensor.numel()

    UNROLL = min(triton.cdiv(triton.cdiv(E, group_size), get_num_SMs(tensor.device)), 1)

    out = torch.empty(inter_shape, device=tensor.device, dtype=w_dtype)
    scales = torch.empty((post_pad_shape[0], post_pad_shape[1] // group_size), device=tensor.device, dtype=torch.uint8)
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp8_triton_kernel[grid](
                tensor, 
                out, 
                scales, 
                E=E,
                eps=2 ** -30,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out.view(orig_shape), scales

fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device="cuda")
fp4_pos = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device='cuda')
thr_pos = (fp4_pos[:-1] + fp4_pos[1:]) * 0.5

@torch.compile(fullgraph=True)
def scale_activations_mxfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 32
    max_val = 6
    eps: float = 2 ** -30

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps)

    W_q = W_flat / scales
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    W_q = (W_q.view(-1, 1) - fp4_values.view(1,-1)).abs().argmin(dim=1).to(torch.uint8).view(inter_shape)
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = scales.to(torch.float8_e8m0fnu).view(torch.uint8).view(post_pad_shape[0], post_pad_shape[1] // group_size)
    return W_q, scales


@torch.compile(fullgraph=True)
def scale_activations_nvfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 16
    max_val = 6
    eps = 2 ** -30
    fp8_dtype = torch.float8_e4m3fn
    max_fp8 = torch.finfo(fp8_dtype).max #448

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = scales.clamp_(max=max_fp8).to(fp8_dtype).to(W_flat.dtype).clamp_(eps)

    W_q = W_flat / scales
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    W_q = (W_q.view(-1, 1) - fp4_values.view(1,-1)).abs().argmin(dim=1).to(torch.uint8).view(inter_shape)
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = scales.to(fp8_dtype).view(post_pad_shape[0], post_pad_shape[1] // group_size)
    return W_q, scales


@triton.jit
def scale_activations_mxfp4_triton_kernel_v1(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    E,
    eps: tl.constexpr,
    UNROLL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0) * UNROLL

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    for m in range(UNROLL):
        #Load
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = (offs < E).to(tl.int1)
        tensor = tl.load(tensor_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor)) / 6., eps)

        #Map to index
        wq = tensor / scales
        idx_pos = tl.sum(wq[:, None] > thr_pos, axis=1) - 1
        idx_neg = tl.sum(wq[:, None] < -thr_pos, axis=1) + 7
        out = tl.where(wq >= 0, idx_pos, idx_neg).to(out_dtype)

        #Pack
        lo, hi = tl.split(out.reshape((HALF_GROUP_SIZE, 2), can_reorder=False))
        out = lo | (hi << 4)

        #Store
        offs_out = pid * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
        tl.store(out_ptr + offs_out, out)
        tl.store(scales_ptr + pid, scales_log2 + 127)

        pid += 1


def scale_activations_mxfp4_triton_v1(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size = 32
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = (tensor.shape[0], tensor.shape[1] // 2)
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    E = tensor.numel()

    UNROLL = min(triton.cdiv(triton.cdiv(E, group_size), get_num_SMs(tensor.device)), 1)

    out = torch.empty(inter_shape, device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((post_pad_shape[0], post_pad_shape[1] * 2 // group_size), device=tensor.device, dtype=torch.uint8)
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp4_triton_kernel_v1[grid](
                tensor, 
                out, 
                scales,
                thr_pos,
                E,
                eps=2 ** -30,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out, scales


@triton.jit
def scale_activations_mxfp4_triton_kernel_v2(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K,
    stride_m_t, stride_k_t,
    stride_m_s, stride_k_s,
    stride_m_o, stride_k_o,
    #########################
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = (offs_m[:, None] < M).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / 6., eps)

    #Map to index
    wq = tensor / scales
    idx_pos = tl.sum(wq[:, :, None] > thr_pos[None, :, :], axis=2) - 1
    idx_neg = tl.sum(wq[:, :, None] < -thr_pos[None, :, :], axis=2) + 7
    out = tl.where(wq >= 0, idx_pos, idx_neg).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2 + 127)

def scale_activations_mxfp4_triton_v2(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size = 32
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = (tensor.shape[0], tensor.shape[1] // 2)
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    M, K = tensor.shape

    out = torch.empty(inter_shape, device=tensor.device, dtype=torch.uint8) 
    scales = torch.empty((post_pad_shape[0], post_pad_shape[1] * 2 // group_size), device=tensor.device, dtype=torch.uint8)

    BLOCK_SIZE_M = min(next_power_of_2(M), 4) 
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))

    scale_activations_mxfp4_triton_kernel_v2[grid](
                tensor,
                out,
                scales,
                thr_pos,
                M, K,
                tensor.stride(0), tensor.stride(1),
                scales.stride(0), scales.stride(1),
                out.stride(0), out.stride(1),
                #########################
                eps=2 ** -30,
                GROUP_SIZE=group_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                num_stages=1,
                num_warps=4,
                )

    return out, scales

scale_activations_per_token = scale_activations_per_token_triton
scale_activations_mxfp8 = scale_activations_mxfp8_triton
scale_activations_mxfp4 = scale_activations_mxfp4_triton_v2
scale_activations_nvfp4 = scale_activations_nvfp4_torch
#######################################################################################################################
def enable_activation_scaling(batch_size):
    """
    This functions enables adaptive activation quantization based on the batch-size. 
    For example, it would only do dynamic activation in more compute-bound settings (batch_size >=32 / 64). 
    Only works with the MXFP format - use with A8W4_MXFP/A4W4_MXFP.
    """
    return True
    #return batch_size >= 32


#Main functional forward call
@torch.library.custom_op("gemlite::forward_functional", mutates_args=())
def forward_functional(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1, #-1: auto, >=0: manual
) -> Tensor:
    
    data_contiguous = bool(meta_args[1])
    W_nbits = meta_args[1]
    out_features = tensor_args[0].shape[1]

    #Get type_id for autotune: use same autotune for float16/bfloat16
    input_dtype = meta_args[5]
    input_dtype = DType.FP16.value if (input_dtype == DType.BF16.value) else input_dtype
    input_dtype = DType.MXFP16.value if (input_dtype == DType.MXBF16.value) else input_dtype
    type_id = input_dtype * 100 + W_nbits

    if not x.is_contiguous():
        x = x.contiguous()

    batch_size = x.numel() // x.shape[-1]
    orig_shape = x.shape
    out_shape = x.shape[:-1] + (out_features,)
    x_dtype = x.dtype

    scaled_activations = bool(meta_args[0]) and enable_activation_scaling(batch_size)
    #Dynamic activation quantization
    scales_x = None
    if(scaled_activations):
        input_dtype = DType(meta_args[5])
        channel_scale_mode = meta_args[9]

        if(input_dtype in FP8_INT8_DTYPES): #INT8 / FP8
            x, scales_x = scale_activations_per_token(x, w_dtype=DTYPE_TO_TORCH[input_dtype.value])

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 4): #MXPF8
            x, scales_x = scale_activations_mxfp8(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 2): #MXPF8 with post-scale for the activations
            x, scales_x = scale_activations_per_token(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP4] and channel_scale_mode == 4): #MXPF4
            x, scales_x = scale_activations_mxfp4(x)

        elif(input_dtype in [DType.NVFP4] and channel_scale_mode == 4): #NVPF4: TODO
            x, scales_x = scale_activations_nvfp4(x)
    
    x = x.view(-1, x.shape[-1])

    if(matmul_type >= 0):
        matmul_type_str = GEMLITE_MATMUL_TYPES[matmul_type]
    else:
        matmul_type_str = get_matmul_type(x.shape[0], W_nbits) #batch_size, W_nbits

    out = GEMLITE_TRITON_MAPPING[matmul_type_str].forward(x, *tensor_args, scales_x, *meta_args[1:-1], data_contiguous, type_id).view(out_shape)

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
    SUPPORTED_DTYPES      = [DType.FP16, DType.BF16, DType.FP32, 
                            DType.FP8, DType.FP8e4, DType.FP8e4nuz, DType.FP8e5, DType.FP8e5nuz, DType.INT8, 
                            DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4, DType.NVFP4]
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
            if(group_size < 16):
                raise NotImplementedError("Only group_size >= 16 is supported.")

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

    def using_mx_dtype(self):
        return self.input_dtype in [DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4, DType.NVFP4]

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

        #Only 8-bit packing is supported for microscaling
        if(self.using_mx_dtype()):
            packing_bitwidth = 8

        #Non-packed weights
        self.W_q = None
        if(W_q.dtype in [torch.int8] or W_q.is_floating_point()):
            if(W_q.dtype in [torch.float32]):
                assert self.W_nbits == 32, "Invalid fp32 weights."
            elif(W_q.dtype in [torch.float16, torch.bfloat16]): 
                assert self.W_nbits == 16, "Invalid fp16 weights."
            else: 
                assert self.W_nbits == 8, "Invalid 8-bit weights."

            self.W_q = W_q.t() #row-major
            self.elements_per_sample = 1

            if(contiguous is None): 
                contiguous = False

        # Packed weigths
        if W_q.dtype == torch.uint8:  
            _pack_weights_over_cols = pack_weights_over_cols_triton if (W_q.device.type == "cuda") else pack_weights_over_cols_torch

            self.W_q, self.elements_per_sample = _pack_weights_over_cols(
                W_q.view(self.orig_shape),
                W_nbits=self.W_nbits,
                packing_bitwidth=packing_bitwidth,
                transpose=True,
            )  # Over-K
            
            if contiguous is None:
                contiguous = True
                #TODO: check this for MX dtypes

        if(self.W_q is None):
            raise Exception('Weights were not packed, please check your W_q.dtype')

        #Bias / device
        self.device = self.W_q.device
        self.bias   = None if (bias is None) else bias.to(device=self.device)
        
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

        #MX dtypes scaling
        if(self.input_dtype in [DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4]):
            self.scales = self.scales.to(torch.float8_e8m0fnu).view(torch.uint8)
        if(self.input_dtype in [DType.NVFP4]):
            self.scales = self.scales.to(torch.float8_e4m3fn)
        if(self.using_mx_dtype()):
            self.scales = self.scales.T
            self.W_group_mode = 2
            self.channel_scale_mode = 0

        if(self.scales is not None):
            self.meta_dtype = TORCH_TO_DTYPE[self.scales.dtype]

        #Register buffers
        self.W_q        = torch.nn.Parameter(self.W_q, requires_grad=False)
        self.bias       = torch.nn.Parameter(self.bias, requires_grad=False) if self.bias is not None else None
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
        from .triton_kernels.gemv_kernels import gemv
        _GEMLITE_TRITON_MAPPING['GEMV'] = gemv

        from .triton_kernels.gemv_revsplitK_kernels import gemv_revsplitK
        _GEMLITE_TRITON_MAPPING['GEMV_REVSPLITK'] = gemv_revsplitK

        from .triton_kernels.gemv_splitK_kernels import gemv_splitK
        _GEMLITE_TRITON_MAPPING['GEMV_SPLITK'] = gemv_splitK

        from .triton_kernels.gemm_splitK_kernels import gemm_splitK
        _GEMLITE_TRITON_MAPPING['GEMM_SPLITK'] = gemm_splitK

        from .triton_kernels.gemm_kernels import gemm
        _GEMLITE_TRITON_MAPPING['GEMM'] = gemm

        for name in _GEMLITE_TRITON_MAPPING:
            if(name not in config): 
                config[name] = {}
            
            if(name in GEMLITE_TRITON_CONFIG_CACHE):
                config[name].update(GEMLITE_TRITON_CONFIG_CACHE[name])

            config[name].update(cache_kernel_config(_GEMLITE_TRITON_MAPPING[name].kernel, 6)) #5: len(prune_keys)

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
