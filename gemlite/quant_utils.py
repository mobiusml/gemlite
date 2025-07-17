from typing import Tuple
import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .triton_kernels.utils import IS_HIP, get_num_SMs, next_power_of_2
from .dtypes import *

#Cache workspace for multiple gpus
fp4_values, thr_pos = [], []
for g_id in range(torch.cuda.device_count()):
    current_device = "cuda:" + str(g_id)
    fp4_values.append(torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device=current_device))
    fp4_pos = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=current_device)
    thr_pos.append((fp4_pos[:-1] + fp4_pos[1:]) * 0.5)


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
def scale_activations_mxfp8_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 32
    eps = 2 ** -30
    w_dtype = torch.float8_e4m3fn
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

def scale_activations_mxfp8_triton(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size = 32
    eps = 2 ** -30
    w_dtype = torch.float8_e4m3fn
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
                eps=eps,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out.view(orig_shape), scales

@torch.compile(fullgraph=True)
def scale_activations_mxfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 32
    eps = 2 ** -30
    max_val = 6

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
    device_index = W_q.device.index
    W_q = (W_q.view(-1, 1) - fp4_values[device_index].view(1,-1)).abs().argmin(dim=1).to(torch.uint8).view(inter_shape)
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = scales.to(torch.float8_e8m0fnu).view(torch.uint8).view(post_pad_shape[0], post_pad_shape[1] // group_size)
    return W_q, scales


@torch.compile(fullgraph=True)
def scale_activations_nvfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 16
    eps = 2 ** -30
    max_val = 6
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
    device_index = W_q.device.index
    W_q = (W_q.view(-1, 1) - fp4_values[device_index].view(1,-1)).abs().argmin(dim=1).to(torch.uint8).view(inter_shape)
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


def scale_activations_mxfp4_triton_v1(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 32
    eps = 2 ** -30
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
    device_index = tensor.device.index
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp4_triton_kernel_v1[grid](
                tensor, 
                out, 
                scales,
                thr_pos[device_index],
                E,
                eps=eps,
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

def scale_activations_mxfp4_triton_v2(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size = 32
    eps = 2 ** -30
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = (tensor.shape[0], tensor.shape[1] // 2)
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    M, K = tensor.shape

    out = torch.empty(inter_shape, device=tensor.device, dtype=torch.uint8) 
    scales = torch.empty((post_pad_shape[0], post_pad_shape[1] * 2 // group_size), device=tensor.device, dtype=torch.uint8)
    device_index = tensor.device.index

    BLOCK_SIZE_M = min(next_power_of_2(M), 4) 
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))

    scale_activations_mxfp4_triton_kernel_v2[grid](
                tensor,
                out,
                scales,
                thr_pos[device_index],
                M, K,
                tensor.stride(0), tensor.stride(1),
                scales.stride(0), scales.stride(1),
                out.stride(0), out.stride(1),
                #########################
                eps=eps,
                GROUP_SIZE=group_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                num_stages=1,
                num_warps=4,
                )

    return out, scales

@triton.jit
def scale_activations_nvfp4_triton_kernel_v2(
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

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = (offs_m[:, None] < M).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #FP8 scales
    scales = tl.max(tl.abs(tensor), axis=1, keep_dims=True) / 6.
    scales = tl.minimum(scales, max_fp8).to(fp8_dtype).to(tl.float32)
    scales = tl.maximum(scales, eps)

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
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales)

def scale_activations_nvfp4_triton_v2(tensor: torch.Tensor, eps: float = 2**-30) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size = 16
    eps = 2 ** -30
    fp8_dtype = torch.float8_e4m3fn
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = (tensor.shape[0], tensor.shape[1] // 2)
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    M, K = tensor.shape

    out = torch.empty(inter_shape, device=tensor.device, dtype=torch.uint8) 
    scales = torch.empty((post_pad_shape[0], post_pad_shape[1] * 2 // group_size), device=tensor.device, dtype=fp8_dtype)
    device_index = tensor.device.index

    BLOCK_SIZE_M = min(next_power_of_2(M), 8) 
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))

    scale_activations_nvfp4_triton_kernel_v2[grid](
                tensor,
                out,
                scales,
                thr_pos[device_index],
                M, K,
                tensor.stride(0), tensor.stride(1),
                scales.stride(0), scales.stride(1),
                out.stride(0), out.stride(1),
                #########################
                eps=eps,
                GROUP_SIZE=group_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                num_stages=1,
                num_warps=4,
                )

    return out, scales

scale_activations_per_token = scale_activations_per_token_triton
scale_activations_mxfp8 = scale_activations_mxfp8_triton
scale_activations_mxfp4 = scale_activations_mxfp4_triton_v2
scale_activations_nvfp4 = scale_activations_nvfp4_triton_v2