# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math
from torch import Tensor
import triton
import triton.language as tl

# code based https://github.com/fpgaminer/GPTQ-triton
def kernel_config_pruner(configs, nargs, **kwargs):
    m = max(2 ** int(math.ceil(math.log2(nargs['M']))), 16) #Need at least 16 here for tl.dot
    n = max(2 ** int(math.ceil(math.log2(nargs['N']))), 16)
    k = max(2 ** int(math.ceil(math.log2(nargs['K']))), 16)
    g = nargs['group_size']

    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = min(m, config.kwargs['BLOCK_SIZE_M'])
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        block_size_k = min(block_size_k, g) #Makes BLOCK_SIZE_K compatible with the group_size
        A_load_order = config.kwargs['A_load_order']
        cache_meta   = config.kwargs['cache_meta']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, A_load_order, cache_meta, config.num_stages, config.num_warps)
        
        if _key in used:
            continue

        used.add(_key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m,
                'A_load_order': A_load_order,
                'cache_meta'  : cache_meta,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )


def get_gemm_config():
    #Tuned on 4090 RTX
    _configs = []
    for _M in [16, 32, 64, 128]: #+ [256] #might need higher values for larger batch-sizes
        for _N in [32, 64, 128]: 
            for _K in [32, 64, 128]: #[32, 64, 128], 32 <= block_size
                for _w in [2, 4]: 
                    for _s in [2, 4]:
                        for _a_load_order in [2]: #[1, 2, 3] - [2]: default 4090
                            for _cache_meta in [0]: #[0, 1] - [0]: default 4090
                                _configs.append(
                                        triton.Config(
                                            {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8, 
                                            'A_load_order': _a_load_order, 'cache_meta': _cache_meta}, 
                                            num_stages=_s, num_warps=_w)
                                        )
    return _configs

@triton.heuristics(values={'CLOSEST_M': lambda args: 2 ** int(math.ceil(math.log2(args['M'])))})
@triton.autotune(
    configs = get_gemm_config(),
    key=['CLOSEST_M', 'N', 'K', 'group_size', 'elements_per_sample'],
    prune_configs_by={
        'early_config_prune': kernel_config_pruner,
    },
    warmup=200, 
    rep=50, #20 for faster tuning 
)

@triton.jit
def gemm_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K, 
    W_nbits: tl.constexpr, group_size: tl.constexpr, unpack_mask: tl.constexpr, elements_per_sample: tl.constexpr, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta, 
    acc_dtype: tl.constexpr,
    ######### tuning params #########
    CLOSEST_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_load_order: tl.constexpr, cache_meta: tl.constexpr,
):
    """
    Based on https://github.com/fpgaminer/GPTQ-triton
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K <= group_size
    """

    pid       = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m            = first_pid_m + (pid % group_size_m)
    pid_n            = (pid % num_pid_in_group) // group_size_m

    #Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = tl.arange(0, BLOCK_SIZE_K)

    #Vectorized coalesced load
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    a_mask  = (offs_am[:, None] < M)
    b_ptrs  = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    #Output
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]

    #Meta data stuff
    q_shifts    = ((offs_k  % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :]
    stride_mul  = BLOCK_SIZE_K / group_size 

    ####################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 

    for k in tl.range(0, num_pid_k, 1, num_stages=1):
        b = tl.load(b_ptrs, eviction_policy='evict_first') #(BLOCK_SIZE_K, BLOCK_SIZE_N) - repeated over K dim

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') #(BLOCK_SIZE_M, BLOCK_SIZE_K)

        #k_m = (k + pid_k) // (group_size // BLOCK_SIZE_K)
        k_m = (k * stride_mul).to(tl.int32)
        if(cache_meta == 0):
            scales = tl.load(scales_ptrs + k_m * stride_meta) 
            zeros  = tl.load(zeros_ptrs  + k_m * stride_meta) 

        if(cache_meta == 1):
            scales = tl.load(scales_ptrs + k_m * stride_meta, eviction_policy='evict_last')
            zeros  = tl.load(zeros_ptrs  + k_m * stride_meta, eviction_policy='evict_last')

        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') #Best for 4090! (BLOCK_SIZE_M, BLOCK_SIZE_K)

        # Unpack and dequantize
        b = ((b >> q_shifts) & unpack_mask)
        b = (b - zeros) * scales

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') #Best for 3090! (BLOCK_SIZE_M, BLOCK_SIZE_K)
        
        #Dot
        acc = tl.dot(a, b.to(a.dtype), acc=acc, out_dtype=acc_dtype, input_precision="ieee") #(BLOCK_SIZE_M, BLOCK_SIZE_N)

        #Advance
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    tl.store(c_ptrs, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))



@torch.library.custom_op("gemlite::gemm_A16fWnO16f_int32packing_forward", mutates_args=())
def gemm_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, 
                                         W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, acc_dtype: str = 'fp16') -> Tensor:
    

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=scales.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    gemm_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, 
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0),
        tl.dtype(acc_dtype),
    )

    return output

@torch.library.register_fake("gemlite::gemm_A16fWnO16f_int32packing_forward")
def gemm_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, 
                                              W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, acc_dtype: str = 'fp16') -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=scales.dtype)


class gemm_A16fWnO16f_int32packing:
    kernel = gemm_A16fWnO16f_int32packing_kernel
    forward = gemm_A16fWnO16f_int32packing_forward
    matmul_type = "GEMM"

__all__ = ["gemm_A16fWnO16f_int32packing"]