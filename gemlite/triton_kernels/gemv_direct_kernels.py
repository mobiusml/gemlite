# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025
# Direct Accumulation GEMV (M=1) specialization for 2-bit Ampere/Ada

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
from ..dtypes import is_mx_dtype
from .config import AUTOTUNE, KERNEL_CACHE, BLOCK_QUANT_SIZE
from . import config as _gemlite_config
from .utils import *

KEYS        = ['M', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'channel_scale_mode']
MATMUL_TYPE = "GEMV_DIRECT"

def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = nargs['M']
    n = nargs['N']
    k = nargs['K']
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = nargs['type_id']

    pre_hook = init_to_zero("c_ptr") if nargs['use_prehook'] else None

    # Check cache
    channel_scale_mode = kwargs.get('channel_scale_mode', 0)
    if MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE:
        signature = str(tuple([get_closest_m(m), n, k, g, e, t]))
        if signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]:
            config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps  = config.pop('num_warps')
            num_ctas   = config.pop('num_ctas', 1)

            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)
            config['NUM_STAGES'] = num_stages

            config['EVEN_K'] = (k % config['BLOCK_SIZE_K'] == 0)
            config['EVEN_N'] = (n % config['BLOCK_SIZE_N'] == 0)

            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)
            return

    used = set()
    for config in configs:
        block_size_m = 1 # Only M=1 allowed for GEMV
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])

        # Constraints: BLOCK_SIZE_K <= group_size
        block_size_k = min(g, block_size_k)
        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        # Block size should be compatible with minimum-packing
        if block_size_k < e:
            continue

        A_load_order  = config.kwargs.get('A_load_order', 0)
        dot_prod_mode = config.kwargs.get('dot_prod_mode', 0)
        num_stages    = config.num_stages
        num_warps     = config.num_warps

        even_k = (k % block_size_k == 0)
        even_n = (n % block_size_n == 0)

        key = (block_size_m, block_size_n, block_size_k, A_load_order, dot_prod_mode, num_stages, num_warps)

        new_config = {
            'BLOCK_SIZE_M': block_size_m,
            'BLOCK_SIZE_N': block_size_n,
            'BLOCK_SIZE_K': block_size_k,
            'A_load_order': A_load_order,
            'dot_prod_mode': dot_prod_mode,
            'NUM_STAGES': num_stages,
            'EVEN_K': even_k,
            'EVEN_N': even_n,
        }

        if IS_HIP:
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            key = key + (new_config['waves_per_eu'],)

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)

########################################################################################################################################
# NVIDIA configurations

def get_max_autotune_config_nvidia():
    configs = []
    for w in [1, 2, 4]:
        for s in [1, 2]:
            for N in [32, 64, 128, 256]:
                for K in [16, 32, 64, 128]:
                    configs.append(
                        triton.Config(
                            {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K, 'A_load_order': 0, 'dot_prod_mode': 0},
                            num_stages=s, num_warps=w,
                        )
                    )
    return configs

def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'A_load_order': 0, 'dot_prod_mode': 0}, num_warps=2, num_stages=1))
    return configs

def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'NUM_STAGES': 1}, num_warps=2, num_stages=1)]

########################################################################################################################################
# AMD configurations

def get_max_autotune_config_amd():
    configs = []
    for w in [1, 2, 4]:
        for s in [1, 2]:
            for v in [0, 2, 4]:
                for N in [32, 64, 128, 256]:
                    for K in [16, 32, 64, 128]:
                        configs.append(
                            triton.Config(
                                {'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': v},
                                num_stages=s, num_warps=w,
                            )
                        )
    return configs

def get_fast_autotune_config_amd():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'waves_per_eu': 2}, num_warps=2, num_stages=1))
    return configs

def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'A_load_order': 0, 'dot_prod_mode': 0, 'NUM_STAGES': 1, 'waves_per_eu': 0}, num_warps=2, num_stages=1)]

########################################################################################################################################

if IS_HIP:
    get_max_autotune_config  = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config       = get_default_config_amd
else:
    get_max_autotune_config  = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config       = get_default_config_nvidia

AUTOTUNE_SETTING = getattr(AUTOTUNE, 'GEMV_DIRECT', 'fast')
if AUTOTUNE_SETTING == 'max':
    get_autotune_config = get_max_autotune_config
elif AUTOTUNE_SETTING == 'fast':
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config

@triton.autotune(
    configs=get_autotune_config(),
    key=KEYS,
    restore_value=['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by={'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemv_direct_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N: tl.constexpr, K: tl.constexpr, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    ######### Strides #########
    stride_am: tl.constexpr, 
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, 
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, 
    stride_cn: tl.constexpr,
    stride_meta_a_m: tl.constexpr, 
    stride_meta_a_g: tl.constexpr,
    stride_meta_g: tl.constexpr, 
    stride_meta_n: tl.constexpr,
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
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, 
    NUM_STAGES: tl.constexpr,
    dot_prod_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0,
    #################################
    meta_evict_policy: tl.constexpr = 'evict_first',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    EVEN_K: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    block_quant_size: tl.constexpr = BLOCK_QUANT_SIZE,
):
    """
    Direct Accumulation GEMV for C = matmul(A, dequantize(B, scales, zeros)).
    Accumulates across the full K dimension in thread registers with no split-K atomic additions.
    Optimized for M=1 decode with column-major packed weights.
    """
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if data_contiguous:
        offs_bn = offs_n
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    mask_n = offs_bn < N
    acc = tl.zeros([BLOCK_SIZE_N], dtype=acc_dtype)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load grouped metadata (scales & zeros) once per BLOCK_SIZE_K tile
        if W_group_mode > 0:
            k_m = (k_start // group_size)

            if W_group_mode >= 2:
                scale_ptrs = scales_ptr + k_m * stride_meta_g + offs_bn * stride_meta_n
                if EVEN_N:
                    scale = tl.load(scale_ptrs, eviction_policy=meta_evict_policy).to(acc_dtype)
                else:
                    scale = tl.load(scale_ptrs, mask=mask_n, other=1.0, eviction_policy=meta_evict_policy).to(acc_dtype)
            else:
                scale = 1.0

            if (W_group_mode == 1) or (W_group_mode >= 3):
                if zero_is_scalar:
                    zero = tl.load(zeros_ptr, eviction_policy=meta_evict_policy).to(acc_dtype)
                else:
                    zero_ptrs = zeros_ptr + k_m * stride_meta_g + offs_bn * stride_meta_n
                    if EVEN_N:
                        zero = tl.load(zero_ptrs, eviction_policy=meta_evict_policy).to(acc_dtype)
                    else:
                        zero = tl.load(zero_ptrs, mask=mask_n, other=0.0, eviction_policy=meta_evict_policy).to(acc_dtype)
            else:
                zero = 0.0
        else:
            scale = 1.0
            zero = 0.0

        # Iterate over packed words along K inside this block
        for sub_w in tl.static_range(BLOCK_SIZE_K // elements_per_sample):
            word_k = (k_start // elements_per_sample) + sub_w
            w_ptrs = b_ptr + word_k * stride_bk + offs_bn * stride_bn

            if EVEN_N:
                packed = tl.load(w_ptrs, eviction_policy=b_evict)
            else:
                packed = tl.load(w_ptrs, mask=mask_n, other=0, eviction_policy=b_evict)

            x_base = sub_w * elements_per_sample

            # Statically unroll extraction and multiply-accumulate for all elements packed in word
            for i in tl.static_range(elements_per_sample):
                k_idx = k_start + x_base + i

                if EVEN_K:
                    xi = tl.load(a_ptr + k_idx * stride_ak, eviction_policy=a_evict).to(acc_dtype)
                else:
                    xi = tl.load(a_ptr + k_idx * stride_ak, mask=k_idx < K, other=0.0, eviction_policy=a_evict).to(acc_dtype)

                qi = ((packed >> (i * W_nbits)) & unpack_mask).to(acc_dtype)

                if W_group_mode == 4:
                    wi = tl.fma(qi, scale, zero)
                elif W_group_mode == 2:
                    wi = qi * scale
                elif W_group_mode == 3:
                    wi = (qi - zero) * scale
                elif W_group_mode == 1:
                    wi = qi - zero
                else:
                    wi = qi

                acc += wi * xi

    # Post-reduction channel-wise scaling if requested
    if channel_scale_mode == 1:
        if EVEN_N:
            scales_b = tl.load(scales_ptr + offs_bn * stride_meta_n, eviction_policy=meta_evict_policy)
        else:
            scales_b = tl.load(scales_ptr + offs_bn * stride_meta_n, mask=mask_n, other=1.0, eviction_policy=meta_evict_policy)
        acc = acc * scales_b.to(acc_dtype)

    # Direct global store (no atomics, zeroing, or intermediate buffers)
    c_ptrs = c_ptr + offs_bn * stride_cn
    if EVEN_N:
        tl.store(c_ptrs, acc.to(output_dtype))
    else:
        tl.store(c_ptrs, acc.to(output_dtype), mask=mask_n)


def gemv_direct_forward(
    x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
    W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
    input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype: int, 
    channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id: int,
    meta_scale: float = 0.0,
) -> Tensor:
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    target_dtype = DTYPE_TO_TORCH[output_dtype]
    output = torch.empty((M, N), device=W_q.device, dtype=target_dtype)
    use_prehook = False

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    dtype = DTYPE_TO_TRITON[input_dtype]
    if dtype in [tl.float16, tl.bfloat16, tl.float32] and _gemlite_config.GEMLITE_FAST_GEMV_ACC:
        acc_dtype = dtype
    else:
        acc_dtype = DTYPE_TO_TRITON[acc_dtype]

    if scales_x is not None:
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = 0, 0

    if is_mx_dtype(input_dtype):
        raise NotImplementedError('MX dtypes are not supported for gemv_direct.')

    gemv_direct_INT_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample, type_id, use_prehook,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        ################################################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype    = acc_dtype,
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ################################################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
        dump_b_val         = 0,
    )

    return output


class gemv_direct:
    kernel      = [gemv_direct_INT_kernel]
    forward     = gemv_direct_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemv_direct"]
