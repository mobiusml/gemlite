# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def is_divisible(dividend, divisor):
    return dividend % divisor == 0

def kernel_config_pruner(configs, nargs, **kwargs):
    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M'] #min(m, config.kwargs['BLOCK_SIZE_M'])
        block_size_n = config.kwargs['BLOCK_SIZE_N'] #min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = config.kwargs['BLOCK_SIZE_K'] #min(k, config.kwargs['BLOCK_SIZE_K'])
        split_k      = config.kwargs['SPLIT_K']

        #Constraints
        #BLOCK_SIZE_K >= group_size
        block_size_k = min(block_size_k, g)
        #K needs to be devisible by BLOCK_SIZE_K * SPLIT_K 
        if(not is_divisible(k, block_size_k * split_k)):
            continue

        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, 
                A_load_order, meta_evict_policy, atomic_mode,
                config.num_stages, config.num_warps,
                )
        
        if _key in used:
            continue

        used.add(_key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m,
                'SPLIT_K'     : split_k,

                'A_load_order'      : A_load_order,
                'meta_evict_policy' : meta_evict_policy,
                'atomic_mode'       : atomic_mode,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            pre_hook=config.pre_hook,
        )


def get_exhaustive_config():
    #Tuned on 4090 RTX
    _configs = []
    for _M in [16]: #This is fixed to 16 for skinny matrices
        for _N in [32, 64]:
            for _K in [32, 64, 128]: #[128], group_size >= 128
                for _w in [4]: #[4] 
                    for _s in [2, 3]: #[2, 3] #
                        for _sK in [2, 4, 8]: #[2, 4, 8]
                            for _a_load_order in [1, 2, 3]: #[1, 2, 3] - [1]: default 4090
                                for _meta_evict_policy in ['']: #[', 'evict_last'] - ['']: default 4090
                                    for _atomic_mode in ['release', 'relaxed']: #['release', 'relaxed']:
                                        _configs.append(
                                                triton.Config(
                                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 
                                                    'GROUP_SIZE_M': 8, 'SPLIT_K': _sK,
                                                    'A_load_order': _a_load_order, 'meta_evict_policy': _meta_evict_policy, 'atomic_mode': _atomic_mode,
                                                    }, 
                                                    num_stages=_s, num_warps=_w,
                                                    pre_hook=init_to_zero("c_ptr"),
                                                    )
                                                )
    return _configs


#4090 RTX
#Optimized for low-batch size decoding: K needs to be divisible by BLOCK_SIZE_K * SPLIT_K = 256 !!!
def get_default_config():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':32, 'SPLIT_K':8, 'GROUP_SIZE_M':8, 
                           'A_load_order':2, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=3, pre_hook=init_to_zero("c_ptr")),]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMM_SPLITK

@triton.autotune(
    configs=get_exhaustive_config() if ENABLE_AUTOTUNE else get_default_config(),
    key=['M', 'N', 'K', 'group_size', 'elements_per_sample'],
    prune_configs_by={'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup=200, 
    rep=50, 
)

@triton.jit
def gemm_splitK_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K, 
    W_nbits: tl.constexpr, group_size: tl.constexpr, unpack_mask: tl.constexpr, elements_per_sample: tl.constexpr, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    acc_dtype: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr, atomic_mode: tl.constexpr,
):
    """
    Based on https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
 
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    #Swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m            = first_pid_m + (pid % group_size_m)
    pid_n            = (pid % num_pid_in_group) // group_size_m

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) 

    #Vectorized coalesced load
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    a_mask  = offs_am[:, None] < M
    b_ptrs  = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    #Meta data stuff
    q_shifts    = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr     = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_P: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
    ####################################################################################
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in tl.range(0, num_pid_k, 1, num_stages=1):

        b = tl.load(b_ptrs) #, eviction_policy='evict_first'

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 
        
        #Meta-data loading policy
        k_m    = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 
        scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
        zeros  = tl.load(zeros_ptrs  + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
        
        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        # Unpack and dequantize
        b = ((b >> q_shifts) & unpack_mask)
        b = (b.to(scales.dtype) - zeros) * scales

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        #Dot
        acc = tl.dot(a, b.to(a.dtype), acc=acc, out_dtype=acc_dtype, input_precision="ieee") 

        #Advance
        a_ptrs += BLOCK_SIZE_K   * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * SPLIT_K * stride_bk

    #Output
    #acc = acc.to(tl.float16) 
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), sem=atomic_mode) #release / relaxed


@torch.library.custom_op("gemlite::gemm_splitK_A16fWnO16f_int32packing_forward", mutates_args=())
def gemm_splitK_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, 
                                                W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                                                acc_dtype: int,
                                                ) -> Tensor: 
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    #assert group_size >= 128, "Only group_size >= 128 is currently supported"
    output = torch.empty((M, N), device=W_q.device, dtype=scales.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    gemm_splitK_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, 
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        tl.float16 if (acc_dtype == 1) else tl.float32,
    )

    return output

@torch.library.register_fake("gemlite::gemm_splitK_A16fWnO16f_int32packing_forward")
def gemm_splitK_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, 
                                              W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                              acc_dtype: int,
                                              ) -> Tensor:
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=scales.dtype)


class gemm_splitK_A16fWnO16f:
    kernel = gemm_splitK_A16fWnO16f_int32packing_kernel
    forward = gemm_splitK_A16fWnO16f_int32packing_forward
    matmul_type = "GEMM_SPLITK"

__all__ = ["gemm_splitK_A16fWnO16f"]
