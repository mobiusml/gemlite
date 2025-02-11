import torch, triton
import triton.language as tl
from ..dtypes import *

@triton.jit
def swizzle_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n

@triton.jit
def dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample: tl.constexpr, W_group_mode: tl.constexpr, zero_is_scalar: tl.constexpr):
    #Unpack
    if(elements_per_sample > 1):
        b = (b >> q_shift) & unpack_mask # int32 -> int32

    if(W_group_mode == 1): #Shift
        b = b.to(meta_dtype) - zeros 

    if(W_group_mode == 2):
        b = b.to(meta_dtype) * scales #Symmetric no shift (Grouped)

    if(W_group_mode == 3): #Asymmetric / Symmetric with shift(Grouped - (b - zeros) * scales)
        #b = (b - zeros) * scales
        if(zero_is_scalar):
            b = (b - zeros).to(meta_dtype) * scales
        else:
            b = (b.to(meta_dtype) - zeros) * scales

    if(W_group_mode == 4):
        b = tl.fma(b.to(meta_dtype), scales, zeros) #Asymmetric (Grouped - b*scales + zeros)

    return b
    
def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def is_divisible(dividend, divisor):
    return dividend % divisor == 0

def gpu_has_more_shared_memory(ref_gpus = ['a100', 'h100', 'h200', 'h800']): 
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]