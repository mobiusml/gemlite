# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025
# ********************************************************
import torch
import triton
import triton.language as tl

# Pack data, adapted from: following the same logic as: https://github.com/LeiWang1999/AutoGPTQ.tvm/blob/dcd135b9784b9f98235fc91467fe3c3c8afa34fc/auto_gptq/nn_modules/qlinear_triton.py#L413-L419
def pack_weights_over_rows_torch(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    assert packing_bitwidth in [8, 16, 32], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits

    W_q     = W_q.to(torch.int32)
    W_q_out = torch.zeros((W_q.shape[0] // elements_per_sample, W_q.shape[1]), dtype=torch.int32, device=W_q.device) 

    i, row = 0, 0
    while row < W_q_out.shape[0]:
        for j in range(i, i + (packing_bitwidth // W_nbits)):
            W_q_out[row] |= W_q[j] << (W_nbits * (j - i))
        i += elements_per_sample
        row += 1

    if(packing_bitwidth == 8) : W_q_out = W_q_out.to(torch.uint8)
    if(packing_bitwidth == 16): W_q_out = W_q_out.to(torch.int16)
    if(packing_bitwidth == 32): W_q_out = W_q_out.to(torch.int32)

    if(transpose): W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample

def pack_weights_over_cols_torch(W_q: torch.Tensor, W_nbits: int, packing_bitwidth: int, transpose: bool)-> tuple[torch.Tensor, int]:
    assert packing_bitwidth in [8, 16, 32], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits

    W_q     = W_q.to(torch.int32)
    W_q_out = torch.zeros((W_q.shape[0], W_q.shape[1] // elements_per_sample), dtype=torch.int32, device=W_q.device) 

    i, col = 0, 0
    while col <  W_q_out.shape[1]: 
        shift = 0
        for j in range(i, i + elements_per_sample):
            W_q_out[:, col] |= (W_q[:, j] << shift)
            shift += W_nbits
        i += elements_per_sample
        col += 1

    if(packing_bitwidth == 8) : W_q_out = W_q_out.to(torch.uint8)
    if(packing_bitwidth == 16): W_q_out = W_q_out.to(torch.int16)
    if(packing_bitwidth == 32): W_q_out = W_q_out.to(torch.int32)

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
    assert packing_bitwidth in [8, 16, 32], "Unsuported bitpacking width"
    assert W_nbits in [8, 4, 2, 1], "Unsuported nbits"
    elements_per_sample = packing_bitwidth // W_nbits
    num_rows, num_input_cols = W_q.shape
    num_cols = num_input_cols // elements_per_sample

    if(packing_bitwidth == 8) : dtype, out_dtype = torch.uint8, tl.uint8 
    if(packing_bitwidth == 16): dtype, out_dtype = torch.int16, tl.int16
    if(packing_bitwidth == 32): dtype, out_dtype = torch.int32, tl.int32

    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)
    unroll  = highest_divisor(num_rows, max_val=64) 
    grid    = (num_rows * num_cols // unroll, )

    pack_weights_over_cols_kernel[grid](
        W_q,
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
    num_blocks = num_cols // unroll
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

    if(packing_bitwidth == 8) : dtype, out_dtype = torch.uint8, tl.uint8 
    if(packing_bitwidth == 16): dtype, out_dtype = torch.int16, tl.int16
    if(packing_bitwidth == 32): dtype, out_dtype = torch.int32, tl.int32

    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)
    unroll  = highest_divisor(num_cols, max_val=64) 
    grid    = (num_rows * num_cols // unroll, )

    pack_weights_over_rows_kernel[grid](
        W_q,
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

########################################################################################################################################################
pack_weights_over_rows = pack_weights_over_rows_triton
pack_weights_over_cols = pack_weights_over_cols_triton
