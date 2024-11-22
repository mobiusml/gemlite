import torch
import triton.language as tl
from enum import Enum

class DType(Enum):
    FP32   = 0
    FP16   = 1
    BF16   = 2
    FP8    = 3
    INT8   = 4
    UINT8  = 5
    INT32  = 6
    UINT32 = 7
    FP8e5  = 8

DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.float8_e4m3fn,
    4: torch.int8,
    5: torch.uint8,
    6: torch.int32,
    7: torch.uint32,
    8: torch.float8_e5m2,
}

TORCH_DTYPE_TO_TRITON = {
    torch.float16:       tl.float16,
    torch.float32:       tl.float32,
    torch.bfloat16:      tl.bfloat16,
    torch.int8:          tl.int8,
    torch.uint8:         tl.uint8,
    torch.int16:         tl.int16,
    torch.uint16:        tl.uint16,
    torch.int32:         tl.int32,
    torch.uint32:        tl.uint32,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2:   tl.float8e5,
}

DTYPE_TO_TRITON = {k:TORCH_DTYPE_TO_TRITON[d] for k,d in DTYPE_TO_TORCH.items()}
