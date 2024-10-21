# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from .utils import *

def kernel_config_pruner(configs, nargs, **kwargs):
    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K']
    g = nargs['group_size']

    used = set()
    for config in configs:
        block_size_m      = 1 #Only 1 allowed
        block_size_n      = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k      = min(k, config.kwargs['BLOCK_SIZE_K'])
        block_size_k      = min(g, block_size_k) #Makes BLOCK_SIZE_K compatible with the group_size
        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']

        _key  = (block_size_m, block_size_n, block_size_k, 
                A_load_order, meta_evict_policy, atomic_mode, 
                config.num_stages, config.num_warps
                )

        if _key in used:
            continue

        used.add(_key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                
                'A_load_order': A_load_order,
                'meta_evict_policy': meta_evict_policy,
                'atomic_mode': atomic_mode,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            pre_hook=config.pre_hook,
        )

def get_autotune_config():
    #Tuned on 4090 RTX
    _configs = []
    for _M in [1]: #ONLY 1 allowed here
        for _N in [128, 256]:
            for _K in [32, 64]: #block_size >=32 
                for _w in [4]: #Changing this makes autotune act weird
                    for _s in [2, 4]: 
                        for _A_load_order in [1, 2]: #2 - default 4090: [1, 2, 3]
                            for _meta_evict_policy in ['']: #[', 'evict_last'] - ['']: default 4090
                                for _atomic_mode in ['relaxed']:  #['release', 'relaxed'] - 'relaxed' default 4090
                                    _configs.append(
                                            triton.Config(
                                                {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 
                                                'A_load_order': _A_load_order, 'meta_evict_policy': _meta_evict_policy, 'atomic_mode': _atomic_mode}, 
                                                num_stages=_s, num_warps=_w, 
                                                pre_hook=init_to_zero("c_ptr"),
                                                )
                                            )

    return _configs


compute_capability = torch.cuda.get_device_capability(0)

def get_default_config():
    #4090: default
    config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'A_load_order':2, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=2, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (8, 0)): #A100
        config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'A_load_order':1, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=2, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (9, 0)): #H100
        config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'A_load_order':1, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=4, pre_hook=init_to_zero("c_ptr"))

    return [config]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMV

@triton.autotune(
    configs = get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
    key=['M', 'N', 'K', 'group_size', 'elements_per_sample'],
    prune_configs_by={'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup=50, 
    rep=50,
    use_cuda_graph=True,
)

@triton.jit
def gemv_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, 
    W_nbits, group_size, unpack_mask, elements_per_sample, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy : tl.constexpr, atomic_mode: tl.constexpr,
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros)). This is optimized for M==1
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K // elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16
    """    

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1) 

    #Swizzle?
    #pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, 8)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offs_k  = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #Vectorized coalesced load
    offs_k  = tl.max_contiguous(tl.multiple_of(offs_k,  BLOCK_SIZE_K), BLOCK_SIZE_K)    
    a_ptrs  = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak  
    b_ptrs  = b_ptr + (offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn

    ####################################################################
    #Load
    if(A_load_order == 0):
        a = tl.load(a_ptrs, eviction_policy='evict_last').to(acc_dtype)

    b = tl.load(b_ptrs, eviction_policy='evict_first')

    if(A_load_order == 1):
        a = tl.load(a_ptrs, eviction_policy='evict_last').to(acc_dtype)
    
    k_m    = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)
    scales = tl.load(scales_ptr + offs_bn[None, :] + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
    zeros  = tl.load(zeros_ptr  + offs_bn[None, :] + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
    
    if(A_load_order == 2):
        a = tl.load(a_ptrs, eviction_policy='evict_last').to(acc_dtype)

    ####################################################################
    # Unpack and dequantize
    b = (b >> ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]) & unpack_mask
    b = ((b.to(meta_dtype) - zeros) * scales).to(acc_dtype) 

    if(A_load_order == 3):
        a = tl.load(a_ptrs, eviction_policy='evict_last').to(acc_dtype)

    #Dot product
    acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b, axis=0, keep_dims=True) #Don't set this to True

    ##################################################################
    #Channel-wise scaling
    if(channel_scale_mode == 1): #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc      = acc.to(meta_dtype)  * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    ####################################################################
    #Output: tl.atomic_add only supports 1D fp16 arrays, bfp16 would crash 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem=atomic_mode) 


_costum_op_id = '_' + str(int(random.random()*10000))

@torch.library.custom_op("gemlite::gemv_A16fWnO16f_int32packing_forward" + _costum_op_id, mutates_args=())
def gemv_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                         W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                         acc_dtype: int,
                                         ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=scales.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(K, meta['BLOCK_SIZE_K']))

    gemv_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, 
        W_nbits, group_size, unpack_mask, elements_per_sample,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        acc_dtype=tl.float16 if (acc_dtype == 1) else tl.float32, 
        meta_dtype=tl.float16,
        channel_scale_mode=0, #1 + zeros-only for channel-wise
    )

    return output

@torch.library.register_fake("gemlite::gemv_A16fWnO16f_int32packing_forward" + _costum_op_id)
def gemv_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                              W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                              acc_dtype: int,
                                              ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=scales.dtype)


class gemv_A16fWnO16f:
    kernel      = gemv_A16fWnO16f_int32packing_kernel
    forward     = gemv_A16fWnO16f_int32packing_forward
    matmul_type = "GEMV"

__all__ = ["gemv_A16fWnO16f"]