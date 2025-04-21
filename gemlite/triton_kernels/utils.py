import torch, triton, math
import triton.language as tl
from ..dtypes import *

@triton.jit
def swizzle_tile_v1(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v2(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_m
    group_id   = pid // width
    group_size = tl.minimum(grid_n - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_n      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_m      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v3(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m  = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n  = pid // tl.cdiv(M, BLOCK_SIZE_M)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    return tl.swizzle2d(pid_m, pid_n, grid_m, grid_n, GROUP_SIZE_M)

@triton.jit
def swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr): 
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

swizzle_tile = swizzle_tile_v1

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    return pid_m, pid_n

#################################################################################################################
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

@triton.jit
def atomic_add_cas(ptr, value, Lock, mask=None, sem: tl.constexpr = "release"):    
    while tl.atomic_cas(Lock, 0, 1, sem=sem) == 1:
        pass
    tl.store(ptr, tl.load(ptr, mask=mask) + value, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def is_divisible(dividend, divisor):
    return dividend % divisor == 0

def gpu_has_more_shared_memory(ref_gpus = ["a100", "h100", "h200", "h20", "h800", "b100", "b200"]): 
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]

def gpu_supports_float16_acc(
    ref_gpus=["5090", "5080", "5070", "5060", 
              "4090", "4080", "4070", "4060", 
              "3090", "3080", "3070", "3060",
              "4000", "5000", '6000',
              '2080', 'titan rtx',
              "a40",  "a10",  "l40"]
):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]


def gpu_supports_bfloat16_atomicadd():
    #Triton tl.atomic_add doens't support bfloat16 even for Hopper and above. 
    #return torch.cuda.get_device_capability()[0] >= 9 #Hopper and above
    return False

#Next power of 2
M_MAXVAL  = 1024
M_MAPPING = {M:min(2 ** int(math.ceil(math.log2(M))), M_MAXVAL) if (M > 0) else 0 for M in range(M_MAXVAL + 1)}
def get_closest_m_fast_autotune(M):
    return M_MAPPING[M] if M <= M_MAXVAL else M_MAXVAL

get_closest_m = get_closest_m_fast_autotune
