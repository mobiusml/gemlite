# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from . import utils
from .utils import (
    DType,
    DTYPE_TO_TORCH,
    DTYPE_TO_TRITON,
    init_to_zero,
    is_divisible,
    swizzle_tile,
    linear_tile,
    dequantize,
    gpu_supports_bfloat16_atomicadd,
    swizzle_tile_persistent,
)

KEYS          = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample'] 
MATMUL_TYPE   = "GEMM_SPLITK"
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()

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

            yield triton.Config(_config,
                num_stages=_num_stages,
                num_warps=_num_warps,
                pre_hook=init_to_zero("c_ptr") if (_config['SPLIT_K'] > 1) else None,
            )

            return

    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min((2 ** int(math.ceil(math.log2(n)))), config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min((2 ** int(math.ceil(math.log2(k)))), config.kwargs['BLOCK_SIZE_K'])
        split_k      = config.kwargs['SPLIT_K']

        #Makes autotuning faster: mainly for batch-size 1 .. 64
        if(m <= 16): block_size_m = 16
        if(m >= 32): block_size_m = min(max(block_size_m, 16), 32) #[16, 32]
        if(m >= 64): block_size_m = min(max(block_size_m, 32), 64) #[32, 64]

        #Only use higher split_k values for smaller m
        if(m >= 32): split_k = min(split_k, 8)
        #Only use lower split_k values for larger m
        if(m <= 16): split_k = max(split_k, 1)

        #Filter 
        block_area = block_size_k * block_size_n
        if(block_area > 4096 * 8): #Limit area for faster autotuning. Use for more 4096 * 8
            continue

        #Constraints
        #BLOCK_SIZE_K >= group_size
        block_size_k = min(block_size_k, g)

        #K needs to be devisible by BLOCK_SIZE_K * SPLIT_K 
        if(not is_divisible(k, block_size_k * split_k)):
            continue

        num_stages = config.num_stages
        num_warps  = config.num_warps
        
        #Nvidia
        if(e > 1): num_stages = 1 #TODO: Remove this after fix?
        if(e == 1 and num_stages == 1): continue #skip num_stages=1 for non-packed weights

        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, 
                A_load_order, meta_evict_policy, atomic_mode,
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
                'SPLIT_K'     : split_k,

                'A_load_order'      : A_load_order,
                'meta_evict_policy' : meta_evict_policy,
                'atomic_mode'       : atomic_mode,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            pre_hook=init_to_zero("c_ptr") if (split_k > 1) else None, 
        )

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_autotune_config():
    _stages  = [1, 2, 4, 5] if utils.gpu_has_more_shared_memory() else [1, 2, 4]
    _configs = []
    for _M in [16, 32, 64]: #for better performance at batch-sizes [4-64]
        for _N in [32, 64, 128, 256]:
            for _K in [32, 64, 128, 256]:
                for _w in [4, 8]:
                    for _s in _stages:
                        for _sK in [1, 2, 4, 8]: 
                            for _a_load_order in [0, 2]: 
                                for _meta_evict_policy in ['']: 
                                    for _atomic_mode in ['relaxed']: 
                                        _configs.append(
                                                triton.Config(
                                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8, 'SPLIT_K': _sK,
                                                    'A_load_order': _a_load_order, 'meta_evict_policy': _meta_evict_policy, 'atomic_mode': _atomic_mode,
                                                    }, 
                                                    num_stages=_s, num_warps=_w,
                                                    pre_hook=init_to_zero("c_ptr") if (_sK > 1) else None,
                                                    )
                                                )
    return _configs


compute_capability = torch.cuda.get_device_capability(0)

#Optimized for low-batch size decoding: K needs to be divisible by BLOCK_SIZE_K * SPLIT_K = 256 !!!
def get_default_config():
    #4090: default
    config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                           'A_load_order':2, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (8, 0)): #A100
        config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                             'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                             num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (9, 0)): #H100
        config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                             'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                             num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    return [config]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMM_SPLITK

@triton.autotune(
    configs=get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
    key = KEYS,
    prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
    warmup = 50, 
    rep = 50,
    use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH,
)

@triton.jit
def gemm_splitK_persistent_A16fWnO16f_int32packing_kernel(
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr, atomic_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Based on https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """

    pid       = tl.program_id(axis=0)
    pid_k     = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    num_tiles = num_pid_m * num_pid_n

    num_sms = NUM_SMS #V1
    #num_sms = tl.cdiv(NUM_SMS, SPLIT_K) #V2

    tile_id_c = pid - num_sms
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in range(pid, num_tiles, num_sms):
        pid_m, pid_n = swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m  = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n  = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        if data_contiguous:
            offs_bn = offs_n  
        else:
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 

        offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)

        #Meta data stuff
        scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
        zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n
        stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
        if(zero_is_scalar):
            zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')
        
        a_ptrs = a_ptr + offs_am[:, None] * stride_am
        b_ptrs = b_ptr + offs_bn[None, :] * stride_bn
        a_mask = offs_am[:, None] < M
        offs_k_base = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, num_pid_k):
            offs_ak = offs_k_base + k * (BLOCK_SIZE_K * SPLIT_K)
            offs_bk = offs_ak // elements_per_sample

            if(A_load_order == 0):
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy='evict_last')

            b = tl.load(b_ptrs + offs_bk[:, None] * stride_bk, eviction_policy='evict_first')

            if(A_load_order == 1):
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy='evict_last')
    
            #Meta-data loading policy
            if(W_group_mode > 0):
                k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 

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

            if(A_load_order == 2):
                a = tl.load(a_ptrs + offs_ak[None, :] * stride_ak, mask=a_mask, other=0., eviction_policy='evict_last')

            # Unpack and dequantize
            if(elements_per_sample > 1):
                q_shift = ((offs_ak % elements_per_sample) * W_nbits).to(tl.int32)[:, None] 
                b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

            acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32") 

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

        ##################################################################
        #Output
        tile_id_c += num_sms
        pid_m, pid_n = swizzle_tile_persistent(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask   = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if(SPLIT_K > 1):
            tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode) 
        else:
            tl.store(c_ptrs, acc, mask=mask)


def gemm_splitK_persistent_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                                W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                                                input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                                channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                                ) -> Tensor: 
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    M_CLOSEST = utils.get_closest_m(M)

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or NATIVE_ATOMIC
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32)
    
    NUM_SMS =  torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), META['SPLIT_K']) #V1
    #grid = lambda META: (min(triton.cdiv(NUM_SMS, META['SPLIT_K']), triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), META['SPLIT_K']) #V2

    gemm_splitK_persistent_A16fWnO16f_int32packing_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        ################################################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = DTYPE_TO_TRITON[output_dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ################################################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
        NUM_SMS            = NUM_SMS,
    )

    if(not native_atomic):
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output


class gemm_splitK_persistent_A16fWnO16f:
    kernel = gemm_splitK_persistent_A16fWnO16f_int32packing_kernel
    forward = gemm_splitK_persistent_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm_splitK_persistent_A16fWnO16f"]
