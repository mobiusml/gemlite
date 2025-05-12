# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025
# ********************************************************
import torch
import triton
import triton.language as tl
from .dtypes import TORCH_DTYPE_TO_TRITON, PACKING_BITWIDTH_TO_TORCH_DTYPE

# Pack data, adapted from: following the same logic as: https://github.com/LeiWang1999/AutoGPTQ.tvm/blob/dcd135b9784b9f98235fc91467fe3c3c8afa34fc/auto_gptq/nn_modules/qlinear_triton.py#L413-L419
def pack_weights_over_rows_torch(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    assert packing_bitwidth in [8, 16, 32, 64], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits

    W_q     = W_q.to(torch.int32)
    W_q_out = torch.zeros((W_q.shape[0] // elements_per_sample, W_q.shape[1]), dtype=torch.int32 if packing_bitwidth <=32 else torch.int64, device=W_q.device) 

    for j in range(W_q.shape[0]):
        row = j // elements_per_sample
        offset = j % elements_per_sample
        W_q_out[row] |= W_q[j] << (W_nbits * offset)

    W_q_out = W_q_out.to(dtype=PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth])

    if(transpose): W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample

def pack_weights_over_cols_torch(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    assert packing_bitwidth in [8, 16, 32, 64], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits

    W_q     = W_q.to(torch.int32)
    W_q_out = torch.zeros((W_q.shape[0], W_q.shape[1] // elements_per_sample), dtype=torch.int32 if packing_bitwidth <=32 else torch.int64, device=W_q.device) 

    for j in range(W_q.shape[1]):
        col = j // elements_per_sample
        shift = (j % elements_per_sample) * W_nbits
        W_q_out[:, col] |= W_q[:, j] << shift

    W_q_out = W_q_out.to(dtype=PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth])

    if(transpose): W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample

########################################################################################################################################################
# Triton Bitpacking
########################################################################################################################################################
_powers_of_2 = [2**n for n in range(10)][::-1]
def highest_divisor(n: int, max_val: int) -> int:
    if(max_val == 1): 
        return 1
    
    for d in _powers_of_2:
        if n % d == 0 and d <= max_val:
            return d

@triton.jit
def or_fn(a, b): return a | b

@triton.jit
def pack_weights_over_cols_kernel(
    W_q_ptr,
    W_q_out_ptr,
    num_input_cols,
    num_cols,
    unroll: tl.constexpr,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    out_dtype: tl.constexpr,
):  

    pid     = tl.program_id(0)
    pid_row = (pid // num_cols) * unroll
    pid_col = (pid % num_cols)

    for r in range(unroll):  
        start_col = pid_col * elements_per_sample
        cols      = tl.arange(0, elements_per_sample)
        shifts    = (cols * W_nbits).to(out_dtype)

        #Load  
        offset = pid_row * num_input_cols + start_col + cols
        offset = tl.max_contiguous(tl.multiple_of(offset, elements_per_sample), elements_per_sample)
        values = tl.load(W_q_ptr + offset).to(out_dtype)

        #Pack
        result = tl.reduce(values << shifts, axis=0, combine_fn=or_fn)

        #Store
        output_offset = pid_row * num_cols + pid_col
        tl.store(W_q_out_ptr + output_offset, result)
        pid_row += 1

def pack_weights_over_cols_triton(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    assert packing_bitwidth in [8, 16, 32, 64], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits
    num_rows, num_input_cols = W_q.shape
    num_cols = num_input_cols // elements_per_sample

    dtype = PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth]
    out_dtype = TORCH_DTYPE_TO_TRITON[dtype]

    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)
    unroll  = highest_divisor(num_rows, max_val=64) 
    grid    = (triton.cdiv(num_rows * num_cols, unroll), )

    pack_weights_over_cols_kernel[grid](
        W_q.contiguous(),
        W_q_out,
        num_input_cols,
        num_cols,
        unroll,
        elements_per_sample,
        W_nbits,
        out_dtype,
        num_stages=2,
        num_warps=1,
    )
    
    if(transpose): W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample

@torch.library.custom_op("gemlite::unpack_over_cols_torch", mutates_args=())
def unpack_over_cols_torch(W_q_packed: torch.Tensor, W_nbits: int, num_output_cols: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols = W_q_packed.shape
    elements_per_sample = num_output_cols // num_cols

    shifts       = torch.arange(elements_per_sample, device=W_q_packed.device, dtype=W_q_packed.dtype) * W_nbits 
    mask         = (1 << W_nbits) - 1
    W_q_unpacked = ((W_q_packed.unsqueeze(-1) >> shifts) & mask).to(dtype)
    W_q_unpacked = W_q_unpacked.view(num_rows, num_output_cols)

    return W_q_unpacked

@torch.library.register_fake("gemlite::unpack_over_cols_torch")
def unpack_over_cols_torch_fake(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols  = W_q_packed.shape
    return torch.empty((num_rows, num_output_cols), dtype=dtype, device=W_q_packed.device)

@triton.jit
def unpack_over_cols_kernel(
    W_q_packed_ptr,
    W_q_unpacked_ptr,
    num_rows,
    num_cols,
    num_output_cols,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    unroll: tl.constexpr,
    output_dtype: tl.constexpr,
):
    pid           = tl.program_id(0)
    num_blocks    = tl.cdiv(num_output_cols, unroll)
    pid_row       = pid // num_blocks
    pid_col_block = pid % num_blocks
    
    # Load
    cols          = pid_col_block * unroll + tl.arange(0, unroll)
    packed_cols   = cols // elements_per_sample
    offset        = pid_row * num_cols + packed_cols
    packed_values = tl.load(W_q_packed_ptr + offset)
    
    # Unpack
    shifts   = (cols % elements_per_sample) * W_nbits
    mask_val = (1 << W_nbits) - 1
    unpacked_values = ((packed_values >> shifts) & mask_val).to(output_dtype)
    
    # Store the unpacked values
    unpacked_offsets = pid_row * num_output_cols + cols
    tl.store(W_q_unpacked_ptr + unpacked_offsets, unpacked_values)

@torch.library.custom_op("gemlite::unpack_over_cols_triton", mutates_args=())
def unpack_over_cols_triton(W_q_packed: torch.Tensor, W_nbits: int, num_output_cols: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:

    # Get input dimensions
    num_rows, num_cols  = W_q_packed.shape
    elements_per_sample = num_output_cols // num_cols

    # Allocate output tensor
    W_q_unpacked = torch.empty((num_rows, num_output_cols), dtype=dtype, device=W_q_packed.device)
    output_dtype = TORCH_DTYPE_TO_TRITON[dtype]

    unroll = highest_divisor(num_cols, max_val=256) 
    grid = (num_rows * triton.cdiv(num_output_cols, unroll),)

    # Launch the kernel
    unpack_over_cols_kernel[grid](
        W_q_packed.contiguous(),
        W_q_unpacked,
        num_rows,
        num_cols,
        num_output_cols,
        elements_per_sample,
        W_nbits,
        unroll,
        output_dtype,
        num_stages=2,
        num_warps=1
    )

    return W_q_unpacked

@torch.library.register_fake("gemlite::unpack_over_cols_triton")
def unpack_over_cols_triton_fake(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols  = W_q_packed.shape
    return torch.empty((num_rows, num_output_cols), dtype=dtype, device=W_q_packed.device)

@triton.jit
def pack_weights_over_rows_kernel(
    W_q_ptr,
    W_q_out_ptr,
    num_rows,
    num_cols,
    unroll: tl.constexpr,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    out_dtype: tl.constexpr,
):  

    pid        = tl.program_id(0)
    num_blocks = tl.cdiv(num_cols, unroll)
    pid_row    = pid // num_blocks
    pid_col    = pid % num_blocks
    col        = pid_col * unroll

    for r in range(unroll):  
        start_row = pid_row * elements_per_sample
        rows = tl.arange(0, elements_per_sample)
        
        #Load
        offset = (start_row + rows) * num_cols + col
        values = tl.load(W_q_ptr + offset).to(out_dtype)
        
        #Pack
        shifts = (rows * W_nbits).to(out_dtype)
        result = tl.reduce(values << shifts, axis=0, combine_fn=or_fn)
        
        #Store
        output_offset = pid_row * num_cols + col
        tl.store(W_q_out_ptr + output_offset, result)
        col += 1


def pack_weights_over_rows_triton(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    elements_per_sample = packing_bitwidth // W_nbits
    num_input_rows, num_cols = W_q.shape
    num_rows = num_input_rows // elements_per_sample

    dtype = PACKING_BITWIDTH_TO_TORCH_DTYPE[packing_bitwidth]
    out_dtype = TORCH_DTYPE_TO_TRITON[dtype]

    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)
    unroll  = highest_divisor(num_cols, max_val=64) 
    grid    = (triton.cdiv(num_rows * num_cols, unroll), )

    pack_weights_over_rows_kernel[grid](
        W_q.contiguous(),
        W_q_out,
        num_rows,
        num_cols,
        unroll,
        elements_per_sample,
        W_nbits,
        out_dtype,
        num_stages=2,
        num_warps=1,
    )

    if(transpose): W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample

@torch.library.custom_op("gemlite::unpack_over_rows_torch", mutates_args=())
def unpack_over_rows_torch(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols = W_q_packed.shape
    elements_per_sample = num_output_rows // num_rows

    shifts = torch.arange(elements_per_sample, device=W_q_packed.device, dtype=W_q_packed.dtype) * W_nbits  
    mask   = ((1 << W_nbits) - 1)
    W_q_unpacked = ((W_q_packed.unsqueeze(-1) >> shifts) & mask).to(dtype) 
    W_q_unpacked = W_q_unpacked.permute(0, 2, 1).reshape(num_output_rows, num_cols)
    
    return W_q_unpacked

@torch.library.register_fake("gemlite::unpack_over_rows_torch")
def unpack_over_rows_torch_fake(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols = W_q_packed.shape
    return torch.empty((num_output_rows, num_cols), dtype=dtype, device=W_q_packed.device)

@triton.jit
def unpack_weights_over_rows_kernel(
    W_q_packed_ptr,
    W_q_unpacked_ptr,
    num_input_rows,
    num_cols,
    num_packed_rows,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    unroll: tl.constexpr,
    output_dtype: tl.constexpr, 
):

    pid        = tl.program_id(0)
    num_blocks = tl.cdiv(num_cols, unroll)
    pid_row    = pid // num_blocks
    pid_block  = pid % num_blocks
    start_col  = pid_block * unroll
    packed_row = pid_row // elements_per_sample

    #Load
    cols   = start_col + tl.arange(0, unroll)
    offset = packed_row * num_cols + cols
    packed_values = tl.load(W_q_packed_ptr + offset)
    
    # Unpack
    shift           = ((pid_row % elements_per_sample) * W_nbits)
    mask            = (1 << W_nbits) - 1
    unpacked_values = ((packed_values >> shift) & mask).to(output_dtype)
    
    # Store
    unpacked_offsets = pid_row * num_cols + cols
    tl.store(W_q_unpacked_ptr + unpacked_offsets, unpacked_values)

@torch.library.custom_op("gemlite::unpack_over_rows_triton", mutates_args=())
def unpack_over_rows_triton(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_packed_rows, num_cols = W_q_packed.shape
    elements_per_sample = num_output_rows // num_packed_rows

    # Allocate output
    W_q_unpacked = torch.empty((num_output_rows, num_cols), dtype=dtype, device=W_q_packed.device)
    output_dtype = TORCH_DTYPE_TO_TRITON[dtype]

    # Define grid
    unroll = highest_divisor(num_cols, max_val=256) 
    grid = (triton.cdiv(num_output_rows * num_cols, unroll),)
    
    # Launch kernel
    unpack_weights_over_rows_kernel[grid](
        W_q_packed.contiguous(),
        W_q_unpacked,
        num_output_rows,
        num_cols,
        num_packed_rows,
        elements_per_sample,
        W_nbits,
        unroll,
        output_dtype,
        num_stages=2,
        num_warps=1
    )
    
    return W_q_unpacked

@torch.library.register_fake("gemlite::unpack_over_rows_triton")
def unpack_over_rows_triton_fake(W_q_packed: torch.Tensor, W_nbits: int, num_output_rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    num_rows, num_cols = W_q_packed.shape
    return torch.empty((num_output_rows, num_cols), dtype=dtype, device=W_q_packed.device)
########################################################################################################################################################
pack_weights_over_rows = pack_weights_over_rows_triton
pack_weights_over_cols = pack_weights_over_cols_triton
unpack_over_rows       = unpack_over_rows_triton
unpack_over_cols       = unpack_over_cols_triton