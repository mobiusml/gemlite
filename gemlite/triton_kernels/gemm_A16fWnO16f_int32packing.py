# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE
from . import utils
from .utils import (
    DTYPE_TO_TORCH,
    DTYPE_TO_TRITON,
    swizzle_tile,
    linear_tile,
    dequantize,
    is_divisible,
    get_gpu_shared_memory,
    next_power_of_2,
)

KEYS        = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'a_sizeof', 'b_sizeof'] 
MATMUL_TYPE = "GEMM"

def kernel_config_pruner(configs, nargs, **kwargs):
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    a_sizeof = nargs['a_sizeof']
    b_sizeof = nargs['b_sizeof']

    #Check cache
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        signature = str(tuple([utils.get_closest_m(m), n, k, g, e]))
        if(signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps  = config.pop('num_warps')
            num_ctas   = config.pop('num_ctas')

            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)

            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps)
            return
    
    gpu_shared_memory = get_gpu_shared_memory() 
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])

        A_load_order = config.kwargs['A_load_order']
        num_stages = config.num_stages
        num_warps  = config.num_warps

        #Autotune prune the batch_size
        if m <= 16:    block_size_m = 16
        elif m <= 32:  block_size_m = min(max(block_size_m, 16), 32)  #m: [16...32]
        elif m <= 64:  block_size_m = min(max(block_size_m, 32), 64)  #m: [32...64]
        elif m <= 128: block_size_m = min(max(block_size_m, 64), 128) #m: [64...128]
        elif m <= 256: block_size_m = min(max(block_size_m, 64), 256) #m: [128...256]
        elif m > 256:  block_size_m = min(max(block_size_m, 64), 256) #m > 256
    
        #Constraint: BLOCK_SIZE_K >= group_size
        block_size_k = min(block_size_k, g)
        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        #Hint: skip block_size_n > block_size_k for col-major non-packed data.

        #Nvidia
        if(e > 1): num_stages = min(num_stages, 4) #Limit num stages when data is packed
        if(e == 1 and num_stages == 1): continue #skip num_stages=1 for non-packed weights

        #Avoid OOM
        while num_stages > 0:
            shared_mem = (block_size_m * block_size_k * a_sizeof + block_size_k * block_size_n * b_sizeof) 
            if(e > 1): 
                shared_mem += block_size_k * block_size_n * a_sizeof
            shared_mem *= num_stages
            if int(shared_mem) <= gpu_shared_memory:
                break
            num_stages -= 1

        if(num_stages == 0): continue #config too large

        key = (block_size_m, block_size_n, block_size_k, group_size_m, A_load_order, num_stages, num_warps)
        
        if key in used:
            continue

        used.add(key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m,
                'A_load_order': A_load_order,
            },
            num_stages=num_stages,
            num_warps=num_warps
        )

def get_max_autotune_config():
    stages  = [1, 4, 5] if utils.gpu_has_more_shared_memory() else [1, 2, 4]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64, 128, 256]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128, 256, 512]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, "GROUP_SIZE_M": 8, "A_load_order": A},
                                    num_warps=w, num_stages=s,
                                )
                            )

    return configs

def get_fast_autotune_config():
    configs = [] #BLOCK_SIZE_M is automatically adapted in the config pruning.
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=3))
    return configs

def get_default_config():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4),]

AUTOTUNE_SETTING = AUTOTUNE.GEMM
if(AUTOTUNE_SETTING == 'max'):
    get_autotune_config = get_max_autotune_config
elif(AUTOTUNE_SETTING == 'fast'):
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config

@triton.autotune(
    configs = get_autotune_config(),
    key = KEYS, 
    prune_configs_by = {'early_config_prune': kernel_config_pruner},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)

@triton.jit
def gemm_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, M_CLOSEST,
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    #################################
    a_sizeof: tl.constexpr,
    b_sizeof: tl.constexpr,
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    #CLOSEST_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_load_order: tl.constexpr, 
    data_contiguous: tl.constexpr,
    #################################
    meta_evict_policy: tl.constexpr = '',
    a_evict: tl.constexpr = '',
    b_evict: tl.constexpr = '',
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
    
    pid = tl.program_id(axis=0)
    
    #Swizzle?
    if(elements_per_sample > 1):
        pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    else:
        pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    #Offsets
    offs_m  = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n  = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = tl.arange(0, BLOCK_SIZE_K)

    #Offsets
    #############################################################################################################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k
    

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < K)).to(tl.int1)
    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    #Meta data stuff
    q_shift     = ((offs_k  % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n
    stride_mul  = BLOCK_SIZE_K / group_size 

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    #############################################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 
    for k in range(num_pid_k):

        if(A_load_order == 0): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        b = tl.load(b_ptrs, eviction_policy=b_evict)

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        #Load meta-data  
        if(W_group_mode > 0):          
            k_m = (k * stride_mul).to(tl.int32)

        if(W_group_mode >= 2): #[2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
            if(zero_is_scalar):
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs  + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None

        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        # Unpack and dequantize
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        #Dot
        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32")

        #Advance
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    #############################################################################################################
    #Channel-wise scaling
    if(channel_scale_mode == 1): #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr   + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    acc = acc.to(output_dtype)
    
    #############################################################################################################
    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)) 


def gemm_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                         W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                         input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                         channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                        ) -> Tensor:
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    M_CLOSEST = utils.get_closest_m(M)

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    gemm_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.dtype.itemsize, W_q.dtype.itemsize,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        ########################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = DTYPE_TO_TRITON[output_dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ########################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
    )

    return output

    
class gemm_A16fWnO16f:
    kernel = gemm_A16fWnO16f_int32packing_kernel
    forward = gemm_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm_A16fWnO16f"]
