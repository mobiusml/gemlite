import torch, math
import triton
import triton.language as tl

##############################################################################################
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
        
        if (block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps) in used:
            continue

        used.add((block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps))
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )

def get_gemm_config():
    #Tuned on 4090 RTX
    _configs = []
    for _M in [16, 32, 64, 128]: #might need higher values for larger batch-sizes
        for _N in [32, 64, 128]: 
            for _K in [32, 64, 128]: #[32, 64, 128], 32 <= block_size
                for _w in [2, 4]: 
                    for _s in [2, 4]:
                        _configs.append(
                                triton.Config(
                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8}, 
                                    num_stages=_s, num_warps=_w)
                                )
    return _configs

@triton.autotune(
    configs = get_gemm_config(),
    key=['M', 'N', 'K', 'group_size', 'W_nbits_1', 'W_nbits_2'],
    prune_configs_by={
        'early_config_prune': kernel_config_pruner,
    },
    warmup=200, 
    rep=50, #20 for faster tuning 
)

@triton.jit
def gemm_A16fWnO16f_int32packing_kernel(
    a_ptr, b_1_ptr, b_2_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K, 
    W_nbits_1: tl.constexpr, W_nbits_2: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask_1: tl.constexpr, unpack_mask_2: tl.constexpr,
    elements_per_sample_1: tl.constexpr, elements_per_sample_2: tl.constexpr,
    stride_am, stride_ak,
    stride_bk_1, stride_bn_1,
    stride_bk_2, stride_bn_2,
    stride_cm, stride_cn,
    stride_meta, 
    acc_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_stages: tl.constexpr
):
    """
    Based on https://github.com/fpgaminer/GPTQ-triton
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//8, N): int32 as a packed matrix
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

    #Inputs
    a_ptrs   = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    a_mask   = (offs_am[:, None] < M)
    b_ptrs_1 = b_1_ptr + ((offs_k[:, None] // elements_per_sample_1) * stride_bk_1 + offs_bn[None, :] * stride_bn_1)
    b_ptrs_2 = b_2_ptr + ((offs_k[:, None] // elements_per_sample_2) * stride_bk_2 + offs_bn[None, :] * stride_bn_2) 

    #Output
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]

    #Meta data stuff
    q_shifts1   = ((offs_k  % elements_per_sample_1) * W_nbits_1).to(tl.int32)[:, None]
    q_shifts2   = ((offs_k  % elements_per_sample_2) * W_nbits_2).to(tl.int32)[:, None]
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :]
    stride_mul  = BLOCK_SIZE_K / group_size 

    ####################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 

    for k in tl.range(0, num_pid_k, 1, num_stages=1):
        b1 = ((tl.load(b_ptrs_1, eviction_policy='evict_first') >> q_shifts1) & unpack_mask_1).to(tl.uint8) #(BLOCK_SIZE_K, BLOCK_SIZE_N) - repeated over K dim
        b2 = ((tl.load(b_ptrs_2, eviction_policy='evict_first') >> q_shifts2) & unpack_mask_2).to(tl.uint8) #(BLOCK_SIZE_K, BLOCK_SIZE_N) - repeated over K dim

        k_m    = (k * stride_mul).to(tl.int32)
        scales = tl.load(scales_ptrs + k_m * stride_meta)
        zeros  = tl.load(zeros_ptrs  + k_m * stride_meta)

        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') #(BLOCK_SIZE_M, BLOCK_SIZE_K)

        # Unpack and dequantize
        b = (((b2 << 1) | b1).to(a.dtype) - zeros) * scales
        
        #Dot
        acc = tl.dot(a, b, acc=acc, out_dtype=acc_dtype, input_precision="ieee") #(BLOCK_SIZE_M, BLOCK_SIZE_N)

        #Advance
        a_ptrs   += BLOCK_SIZE_K
        b_ptrs_1 += (BLOCK_SIZE_K // elements_per_sample_1) * stride_bk_1
        b_ptrs_2 += (BLOCK_SIZE_K // elements_per_sample_2) * stride_bk_2


    tl.store(c_ptrs, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


def gemm_A16fWnO16f_int32packing_forward(x, W_q_1bit, W_q_2bit, scales, zeros, W_nbits_1, W_nbits_2, group_size, unpack_mask_1, unpack_mask_2, elements_per_sample_1, elements_per_sample_2, acc_dtype=tl.float16):
    output = torch.empty((x.shape[0], W_q_1bit.shape[1]), device=W_q_1bit.device, dtype=scales.dtype)

    #assert x.shape[1] == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    grid = lambda META: (
        triton.cdiv(x.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(W_q_1bit.shape[1], META['BLOCK_SIZE_N']),
    )

    gemm_A16fWnO16f_int32packing_kernel[grid](
        x, W_q_1bit, W_q_2bit, output,
        scales, zeros, 
        x.shape[0], W_q_1bit.shape[1], x.shape[1], 
        W_nbits_1, W_nbits_2, group_size, 
        unpack_mask_1, unpack_mask_2, 
        elements_per_sample_1, elements_per_sample_2,  
        x.stride(0), x.stride(1),
        W_q_1bit.stride(0), W_q_1bit.stride(1),
        W_q_2bit.stride(0), W_q_2bit.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0),
        acc_dtype,
    )

    return output


class GemliteLinear(torch.nn.Module):
    def __init__(
        self,
        W_nbits,
        group_size,
        in_features,
        out_features,
        bias
    ):  

        self.SUPPORTED_BITS = [3] #+ [3]

        super().__init__()
        if W_nbits not in self.SUPPORTED_BITS:
            raise NotImplementedError("Only 2,4,8 W_nbits are supported.")
        # if in_features % 256 != 0 or out_features % 256 != 0:
        #     raise NotImplementedError("in_feature or out_feature must be divisible by 256.")
        self.in_features  = in_features
        self.out_features = out_features
        self.orig_shape   = (out_features, in_features)
        self.group_size   = group_size if group_size != -1 else in_features

        self.W_nbits_1     = 1
        self.unpack_mask_1 = 2 ** self.W_nbits_1 - 1
        self.elements_per_sample_1 = 32 // self.W_nbits_1

        self.W_nbits_2     = 2
        self.unpack_mask_2 = 2 ** self.W_nbits_2 - 1
        self.elements_per_sample_2 = 32 // self.W_nbits_2
        
        self.bias         = None

    def pack_base(self, W_q, n_bits):
        W_q      = W_q.reshape(self.orig_shape).t().contiguous().to(torch.int32)
        W_q_out = torch.zeros((W_q.shape[0] // 32 * n_bits, W_q.shape[1]), dtype=torch.int32, device=W_q.device) #Packed

        step = 32 // n_bits
        i, row = 0, 0
        while row <  W_q_out.shape[0]:
            shift = 0
            for j in range(i, i + step):
                W_q_out[row] |= (W_q[j] << shift)
                shift += n_bits
            i += step
            row += 1

        return W_q_out.contiguous()

    def pack(self, W_q, scales, zeros):
        self.W_q_1bit = self.pack_base(W_q.int() & 0b1, self.W_nbits_1)
        self.W_q_2bit = self.pack_base((W_q.int() >> 1) & 0b11, self.W_nbits_2)
        self.scales   = scales.reshape((self.out_features, -1)).t().contiguous()
        self.zeros    = zeros.reshape((self.out_features, -1)).t().contiguous()
        return self

    
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        out = gemm_A16fWnO16f_int32packing_forward(
            x.reshape(-1, x.shape[-1]),
            self.W_q_1bit, self.W_q_2bit,
            self.scales,
            self.zeros,
            self.W_nbits_1, self.W_nbits_2,
            self.group_size,
            self.unpack_mask_1, self.unpack_mask_2,
            self.elements_per_sample_1, self.elements_per_sample_2
        ).reshape(out_shape)

        if(self.bias is not None):
            out += self.bias
        return out


#################################################################################################################################
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.coordinate_descent_tuning = True

@torch.compile()
def matmul_torch_A16W8SYM(x, W_q, scales, out_features):
    out_shape = x.shape[:-1] + (out_features,)
    out = ((x.view((-1, x.shape[-1])) @ W_q.T.to(x.dtype)) / scales.view(1, -1)).view(out_shape)
    return out

class Torch_A16W8SYM(torch.nn.Module):
    def __init__(self, in_features, out_features, W_q, scales, bias=None):
        super().__init__() 
        self.W_q           = W_q
        self.in_features   = in_features
        self.out_features  = out_features
        self.scales        = scales 
        self.bias          = bias 
        self.device        = W_q.device
        self.compute_dtype = scales.dtype
     
    def forward(self, x):
        out = matmul_torch_A16W8SYM(x.to(self.device), self.W_q, self.scales, self.out_features)
        if(self.bias is not None):
            out += self.bias
        return out

from hqq.core.quantize import *
from hqq.backends.bitblas import patch_hqq_to_bitblas, HQQLinearBitBlas
from hqq.backends.torchao import patch_hqq_to_aoint4


import numpy as np
from triton.testing import do_bench
def eval_time(fct, params): 
    return do_bench(lambda: fct(**params), warmup=25, rep=200, fast_flush=True, return_mode='min') 

torch.random.manual_seed(100)
class empty_linear(torch.nn.Module):
    def __init__(self, in_features, out_features, compute_dtype, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.compute_dtype = compute_dtype

def gen_data(in_features, out_features, W_nbits, group_size, device='cuda:0'):
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False, device='cpu')

    quant_config = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size, quant_zero=False, quant_scale=False, axis=1)
    hqq_layer    = HQQLinear(linear, quant_config=quant_config, compute_dtype=torch.float16, device=device, del_orig=False) #bfloat16

    orig_shape   = (out_features, in_features)
    W            = hqq_layer.dequantize().reshape(orig_shape)

    triton_linear, torchao_linear, bitblas_linear, marlin_linear = [None]*4

    # W_q    = torch.randint(0, 2**W_nbits, (out_features, in_features), dtype=compute_dtype, device=device).to(torch.uint8)

    # orig_shape = (out_features, in_features)
    # N          = in_features * out_features // group_size

    # #scales = torch.ones((N,), dtype=compute_dtype, device='cuda:0')
    # scales  = torch.randn((N,), dtype=compute_dtype, device=device).abs()/500.

    # #zeros  = torch.zeros((N,), dtype=compute_dtype, device='cuda:0')
    # zeros  = torch.randint(0, 2**W_nbits - 1, (N,), dtype=compute_dtype, device=device)


    # W      = ((W_q.reshape([-1, group_size]) - zeros.view((N, 1))) * scales.view((N, 1))).reshape(orig_shape)


    #Triton
    if(W_nbits in [3]):
        Z             = W.numel() // group_size
        W_q           = hqq_layer.unpack()[:Z,:].reshape(orig_shape).clone()
        scales        = hqq_layer.meta['scale'].clone()
        zeros         = hqq_layer.meta['zero'].clone()
        triton_linear = GemliteLinear(W_nbits=W_nbits, group_size=group_size, in_features=in_features, out_features=out_features, bias=False).pack(W_q, scales, zeros)
        triton_linear.compute_dtype = scales.dtype

    # #TorchAO
    # if(W_nbits==8):
    #     torchao_linear = Torch_A16W8SYM(in_features, out_features, (W_q.int() - 127).to(torch.int8), scales, bias=None)
    # if(W_nbits==4):
    #     torchao_linear = patch_hqq_to_aoint4(hqq_layer, None)

    # # Bitblas
    # if(W_nbits in [8, 4, 2]):
    #     bitblas_linear = patch_hqq_to_bitblas(HQQLinear(linear, quant_config=quant_config, compute_dtype=torch.float16, device=device, del_orig=False), None)

    #################################################################
    # #Marlin
    # from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinLinearMethod as MarlinLinearMethod
    # from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig as MarlinConfig

    # if(W_nbits==4):
    #     _marlin_linear = MarlinLinearMethod(MarlinConfig(weight_bits=W_nbits, group_size=group_size, has_zp=True, lm_head_quantized=False))

    #     marlin_linear = empty_linear(in_features, out_features, compute_dtype=torch.float16, device='cuda:0')
    #     _marlin_linear.create_weights(layer=marlin_linear,
    #             input_size_per_partition=in_features,
    #             output_partition_sizes=[out_features],
    #             input_size=in_features,
    #             output_size=out_features,
    #             params_dtype=torch.float16)

    #     marlin_linear = marlin_linear.cuda()
    #     _marlin_linear.process_weights_after_loading(marlin_linear)

    #     marlin_linear.scales.data = torch.zeros_like(marlin_linear.scales.data) + 1
    #     marlin_linear.bias = None
    #     marlin_linear.forward = lambda x: _marlin_linear.apply(layer=marlin_linear, x=x, bias=marlin_linear.bias)
    # #################################################################


    del linear
    torch.cuda.empty_cache()

    return W, triton_linear, torchao_linear, bitblas_linear, marlin_linear


def check_valid(x, W, quant_linear):

    y_ref = torch.matmul(x, W.T)
    y_q   = quant_linear(x)

    # print("ref", y_ref)
    # print("quantized", y_q)
    # print("max error", (y_ref - y_q).abs().max())

    try:
        assert (y_ref - y_q).abs().mean() < 1e-3
    except:
        raise Error('assertion failed')


#############################################################################################################
#in_features, out_features = 4096, 4096
#in_features, out_features = 4096*2, 4096*2
#in_features, out_features = 4096*4, 4096*4 
in_features, out_features = 4096*8, 4096*8 

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#BATCH_SIZES = [128, 256, 512, 1024]

HQQLinearBitBlas.DEFAULT_BATCHSIZE = [1] # [1] / [8], [16] 
#HQQLinearBitBlas.DEFAULT_BATCHSIZE = [16]
HQQLinearBitBlas.check = lambda hqq_layer: True
HQQLinearBitBlas.BIT_TO_DTYPE = {8:"uint8", 4: "uint4", 2: "uint2", 1: "uint1"}

#W_nbits, group_size = 8, in_features 
#W_nbits, group_size = 4, 128 
#W_nbits, group_size = 2, 128
W_nbits, group_size = 3, 128 

W, triton_linear, torchao_linear, bitblas_linear, marlin_linear = gen_data(in_features, out_features, W_nbits, group_size)

print("W_nbits", W_nbits, "group_size", group_size)
for batch_size in BATCH_SIZES:

    x = torch.randn((batch_size, in_features), dtype=triton_linear.compute_dtype, device='cuda:0')/10.
    check_valid(x, W, triton_linear)

    ref_time = eval_time(lambda x: torch.matmul(x, W.T),  {'x':x.to(W.dtype)}) 
    
    if(triton_linear is not None):
        triton_time  = eval_time(lambda x: triton_linear(x),      {'x':x.to(triton_linear.compute_dtype)}) 
        print((batch_size, in_features, out_features), 'Triton  Speed-up vs. torch.matmul', np.round(ref_time/triton_time, 2), 'time', triton_time)

    if(torchao_linear is not None):
        torchao_time = eval_time(lambda x: torchao_linear(x),     {'x':x.to(torchao_linear.compute_dtype)}) 
        print((batch_size, in_features, out_features), 'Torchao Speed-up vs. torch.matmul', np.round(ref_time/torchao_time, 2))

    if(bitblas_linear is not None):
        bitblas_time = eval_time(lambda x: bitblas_linear(x),     {'x':x.to(bitblas_linear.compute_dtype)}) 
        print((batch_size, in_features, out_features), 'Bitblas Speed-up vs. torch.matmul', np.round(ref_time/bitblas_time, 2))

    if(marlin_linear is not None):
        marlin_time = eval_time(lambda x: marlin_linear.forward(x),     {'x':x.to(marlin_linear.compute_dtype)}) 
        print((batch_size, in_features, out_features), 'Marlin Speed-up vs. torch.matmul', np.round(ref_time/marlin_time, 2))

    print('----------------------------------------------')



