# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from . import utils
from .utils import DTYPE_TO_TORCH, DTYPE_TO_TRITON, swizzle_tile, linear_tile, dequantize

KEYS        = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample'] 
MATMUL_TYPE = "GEMM"

def kernel_config_pruner(configs, nargs, **kwargs):
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    #Check cache
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        _signature = str(tuple([utils.get_closest_m(m), n, k, g, e]))
        if(_signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            _config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][_signature])
            _num_stages = _config.pop('num_stages')
            _num_warps  = _config.pop('num_warps')
            _num_ctas   = _config.pop('num_ctas')

            _config.pop('num_buffers_warp_spec', None)
            _config.pop('num_consumer_groups', None)
            _config.pop('reg_dec_producer', None)
            _config.pop('reg_inc_consumer', None)

            yield triton.Config(_config, num_stages=_num_stages, num_warps=_num_warps)
            return
            
    used = set()
    for config in configs:
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = config.kwargs['BLOCK_SIZE_N']
        block_size_k = config.kwargs['BLOCK_SIZE_K']
        block_size_k = min(block_size_k, g) #Makes BLOCK_SIZE_K compatible with the group_size

        #Makes autotune a bit faster: NOT optimal at larger batch-sizes > 128
        if(m <= 16) : block_size_m = 16
        if(m >= 32) : block_size_m = min(max(block_size_m, 16), 32)   #[16, 32]
        if(m >= 64) : block_size_m = min(max(block_size_m, 32), 64)   #[32, 64]
        if(m >= 128): block_size_m = min(max(block_size_m, 64), 128)  #[64, 128]
        if(m >= 256): block_size_m = min(max(block_size_m, 64), 256)  #[64, 256]
        if(m >= 512): block_size_m = min(max(block_size_m, 64), 256)  #[64, 256]

        #Filter 
        block_area = block_size_k * block_size_n
        if(block_area > 4096 * 4): #Limit area for faster autotuning. Use for more 4096 * 8
            continue

        num_warps  = config.num_warps
        num_stages = config.num_stages

        #if(e > 1): num_stages = 1 #TODO: Remove this after fix
        if(e == 1 and num_stages == 1): continue #skip num_stages=1 for non-packed weights

        group_size_m      = config.kwargs['GROUP_SIZE_M']
        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, 
                A_load_order, meta_evict_policy, 
                num_stages, num_warps)
        
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
                'meta_evict_policy': meta_evict_policy,
            },
            num_stages=num_stages,
            num_warps=num_warps
        )


def get_autotune_config():
    _stages  = [1, 4, 5] if utils.gpu_has_more_shared_memory() else [1, 2, 4]
    _configs = []
    for _M in [16, 32, 64, 128, 256]: #might need higher values for larger batch-sizes
        for _N in [32, 64, 128, 256]: 
            for _K in [32, 64, 128, 256]:
                for _w in [4, 8]:
                    for _s in _stages:
                        for _A_load_order in [0, 2]: #[0, 1, 2, 3] - [2] for 4090, [0]: for A100 
                            for _meta_evict_policy in ['']: #[', 'evict_last'] - ['']: default 4090
                                _configs.append(
                                        triton.Config(
                                            {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8, 
                                            'A_load_order': _A_load_order, 'meta_evict_policy': _meta_evict_policy}, 
                                            num_stages=_s, num_warps=_w)
                                        )

    return _configs


#4090 RTX
def get_default_config():
    #small batch, not sure what is the right default cnnfig here.
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2, 'meta_evict_policy':''}, 
                            num_warps=4, num_stages=1),]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMM

@triton.autotune(
    configs = get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
    key = KEYS, 
    prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup = 50, 
    rep = 50,
    use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH,
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
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr,
    data_contiguous: tl.constexpr,
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
    #pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    #Offsets
    offs_m  = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n  = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = tl.arange(0, BLOCK_SIZE_K)

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

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    a_mask  = (offs_am[:, None] < M)
    b_ptrs  = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    #Meta data stuff
    q_shift     = ((offs_k  % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n
    stride_mul  = BLOCK_SIZE_K / group_size 

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    ####################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 

    #for k in tl.range(0, num_pid_k, 1): #num_stages=1
    for k in range(num_pid_k):

        if(A_load_order == 0): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        b = tl.load(b_ptrs, eviction_policy='evict_first')

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
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
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        # Unpack and dequantize
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        #Dot
        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32")

        #Advance
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

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

    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)) 


_costum_op_id = '_' + str(int(random.random()*10000))

@torch.library.custom_op("gemlite::gemm_A16fWnO16f_int32packing_forward" + _costum_op_id, mutates_args=())
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

@torch.library.register_fake("gemlite::gemm_A16fWnO16f_int32packing_forward" + _costum_op_id)
def gemm_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                              W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                              input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                              channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                              ) -> Tensor:

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])


class gemm_A16fWnO16f:
    kernel = gemm_A16fWnO16f_int32packing_kernel
    forward = gemm_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm_A16fWnO16f"]
