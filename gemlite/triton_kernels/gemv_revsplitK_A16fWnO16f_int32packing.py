# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from .utils import *

KEYS        = ['M', 'N', 'K', 'group_size', 'elements_per_sample']
MATMUL_TYPE = "GEMV_REVSPLITK"

def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    #Check cache
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        _signature = str(tuple([nargs[i] for i in KEYS]))
        if(_signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            _config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][_signature])
            _num_stages = _config.pop('num_stages')
            _num_warps  = _config.pop('num_warps')
            _num_ctas   = _config.pop('num_ctas')

            _config.pop('num_buffers_warp_spec', None)
            _config.pop('num_consumer_groups', None)
            _config.pop('reg_dec_producer', None)
            _config.pop('reg_inc_consumer', None)

            yield triton.Config(_config,
                num_stages=_num_stages,
                num_warps=_num_warps,
                pre_hook=init_to_zero("c_ptr"),
            )

            return

    used = set()
    for config in configs:
        block_size_m      = 1 #Only 1 allowed here
        block_size_n      = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k      = min(k, config.kwargs['BLOCK_SIZE_K'])

        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']
        dot_prod_mode     = config.kwargs['dot_prod_mode']
        split_k           = 2

        #Faster autotune
        block_area = block_size_k * block_size_n
        if(block_area < 1024 or block_area > 4096 * 8): 
            continue

        #Constraints
        #BLOCK_SIZE_K <= group_size
        block_size_k = min(block_size_k, g)

        #Since we load the scales / zeros once per split_k pass, we need this
        if(not (block_size_k * split_k <= g)):
            continue

        #Block size should be compatible with minimum-packing
        if(block_size_k < e):
            continue
        
        #K needs to be divisible by BLOCK_SIZE_K * SPLIT_K 
        if(not is_divisible(k, block_size_k * split_k)):
            continue

        _key  = (block_size_m, block_size_n, block_size_k, 
                A_load_order, meta_evict_policy, atomic_mode, dot_prod_mode,
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
                'dot_prod_mode': dot_prod_mode,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            pre_hook=config.pre_hook,
        )

#contiguous = True
def get_autotune_config():
    #Tuned on 4090 RTX / A100 SXM4
    _configs = []
    for _M in [1]: #ONLY 1 allowed here
        for _N in [128, 256, 512]:
            for _K in [8, 16, 32, 64]: 
                for _w in [2, 4]:
                    for _s in [1, 2]:
                        for _A_load_order in [0, 1]: 
                            for _meta_evict_policy in ['']: #[', 'evict_last']
                                for _atomic_mode in ['relaxed']:  #['release', 'relaxed']
                                    for _dot_prod_mode in [0]: #[0, 1]
                                        _configs.append(
                                                triton.Config(
                                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 
                                                    'A_load_order': _A_load_order, 'meta_evict_policy': _meta_evict_policy, 
                                                    'atomic_mode': _atomic_mode, 'dot_prod_mode': _dot_prod_mode}, 
                                                    num_stages=_s, num_warps=_w, 
                                                    pre_hook=init_to_zero("c_ptr"),
                                                    )
                                                )

    return _configs


compute_capability = torch.cuda.get_device_capability(0)

def get_default_config():
    # #4090: default
    config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'A_load_order':1, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                            num_warps=4, num_stages=2, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (8, 0)): #A100
        config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':16, 'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                            num_warps=2, num_stages=2, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (9, 0)): #H100
        config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':16, 'A_load_order':1, 'meta_evict_policy':'', 'atomic_mode':'relaxed', 'dot_prod_mode':0}, 
                            num_warps=2, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    return [config]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMV_REVSPLITK

@triton.autotune(
    configs = get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
    key = KEYS,
    prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup = 50, 
    rep = 50,
    use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH,
)

@triton.jit
def gemv_revsplitK_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy : tl.constexpr, atomic_mode: tl.constexpr, dot_prod_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0, #Improve accuracy mainly for A16W8 with post looop scaling
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros)). This is optimized for M==1
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K // elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16
    """    
    ##############################
    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1) * 2

    #Swizzle?
    pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    #pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, 8)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #Vectorized coalesced load
    ##############################
    offs_am = offs_m
    offs_ak = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    if(data_contiguous):
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
        offs_bk = offs_k
    else:
        offs_bn = offs_n
        offs_bk = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    ###############################

    a_ptrs  = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak  
    b_ptrs  = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

    #Load meta data first, for two passes
    if(W_group_mode > 0):
        k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)

    if(W_group_mode >= 2): #[2, 3, 4]
        scales = tl.load(scales_ptr + offs_bn[None, :] * stride_meta_n + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
    else:
        scales = None
    
    if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
        if(zero_is_scalar):
            zeros = tl.load(zeros_ptr, eviction_policy=meta_evict_policy)
        else:
            zeros = tl.load(zeros_ptr  + offs_bn[None, :] * stride_meta_n + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
    else:
        zeros = None

    ##################################################################
    #Load
    if(A_load_order == 0):
        a = tl.load(a_ptrs, eviction_policy='evict_last').reshape((BLOCK_SIZE_K, 1), can_reorder=False)
    
    b = tl.load(b_ptrs, eviction_policy='evict_first') 

    if(A_load_order == 1):
        a = tl.load(a_ptrs, eviction_policy='evict_last').reshape((BLOCK_SIZE_K, 1), can_reorder=False)

    # Unpack and dequantize    
    b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

    #Dot product
    if(dump_b_val > 0): b = b.to(tl.float32) * dump_b_val
    if(dot_prod_mode == 0):
        acc = tl.sum(a.to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True)
    if(dot_prod_mode == 1):
        acc = tl.sum(a * b.to(input_dtype), axis=0, keep_dims=True).to(acc_dtype)
    
    #Advance and load next chunk
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    if(A_load_order == 0):
        a = tl.load(a_ptrs, eviction_policy='evict_last').reshape((BLOCK_SIZE_K, 1), can_reorder=False)
    
    b = tl.load(b_ptrs, eviction_policy='evict_first') 

    if(A_load_order == 1):
        a = tl.load(a_ptrs, eviction_policy='evict_last').reshape((BLOCK_SIZE_K, 1), can_reorder=False)

    # Unpack and dequantize    
    b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)
    
    #Dot product
    if(dump_b_val > 0): b = b.to(tl.float32) * dump_b_val
    if(dot_prod_mode == 0):
        acc += tl.sum(a.to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True) 
    if(dot_prod_mode == 1):
        acc += tl.sum(a * b.to(input_dtype), axis=0, keep_dims=True).to(acc_dtype)

    if(dump_b_val > 0): acc /= dump_b_val
    ##################################################################
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
    ##################################################################

    #Output: tl.atomic_add only supports 1D fp16 arrays, bfp16 would crash 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem=atomic_mode) 


_costum_op_id = '_' + str(int(random.random()*10000))

@torch.library.custom_op("gemlite::gemv_revsplitK_A16fWnO16f_int32packing_forward" + _costum_op_id, mutates_args=())
def gemv_revsplitK_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                                   W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                                   input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                                   channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                                   ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(K, meta['BLOCK_SIZE_K'] * 2))

    gemv_revsplitK_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, 
        W_nbits, group_size, unpack_mask, elements_per_sample,
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
        dump_b_val         = 0.001 if(W_group_mode in [0, 1] and acc_dtype == DType.FP16.value and W_nbits == 8) else 0, #Warning: Only use with INT8
    )

    return output

@torch.library.register_fake("gemlite::gemv_revsplitK_A16fWnO16f_int32packing_forward" + _costum_op_id)
def gemv_revsplitK_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                                        W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                                        input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                                        channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                                        ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])


class gemv_revsplitK_A16fWnO16f:
    kernel      = gemv_revsplitK_A16fWnO16f_int32packing_kernel
    forward     = gemv_revsplitK_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemv_revsplitK_A16fWnO16f"]
