import torch
import triton.language as tl
from enum import Enum

#torch_fp8_e4, triton_fp8_e4 = torch.float8_e4m3fn, tl.float8e4nv #Nvidia
torch_fp8_e4, triton_fp8_e4 = torch.float8_e4m3fnuz, tl.float8e4nv #AMD

class DType(Enum):
    FP32   = 0
    FP16   = 1
    BF16   = 2
    FP8    = 3
    FP8e4  = 3 #alias for FP8
    INT8   = 4
    UINT8  = 5
    INT32  = 6
    UINT32 = 7
    FP8e5  = 8
    FP8e4nuz = 9
    FP8e5nuz = 10

DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    4: torch.int8,
    5: torch.uint8,
    6: torch.int32,
    7: torch.uint32,
    3: torch.float8_e4m3fn,
    9: torch.float8_e4m3fnuz,
    8: torch.float8_e5m2,
    10: torch.float8_e5m2fnuz,
}

TORCH_TO_DTYPE = {
    torch.float32:  DType.FP32,
    torch.float16:  DType.FP16,
    torch.bfloat16: DType.BF16,
    torch.int8:     DType.INT8,
    torch.uint8:    DType.UINT8,
    torch.int32:    DType.INT32,
    torch.uint32:   DType.UINT32,
    torch.float8_e4m3fn: DType.FP8e4,
    torch.float8_e4m3fnuz: DType.FP8e4nuz,
    torch.float8_e5m2: DType.FP8e5,
    torch.float8_e5m2fnuz: DType.FP8e5nuz, 
}

TORCH_DTYPE_TO_TRITON = {
    torch.float16:     tl.float16,
    torch.float32:     tl.float32,
    torch.bfloat16:    tl.bfloat16,
    torch.int8:        tl.int8,
    torch.uint8:       tl.uint8,
    torch.int16:       tl.int16,
    torch.uint16:      tl.uint16,
    torch.int32:       tl.int32,
    torch.uint32:      tl.uint32,
    torch.float8_e4m3fn: tl.float8e4nv, #NVIDIA
    torch.float8_e4m3fnuz: tl.float8e4b8, #AMD
    torch.float8_e5m2: tl.float8e5,#NVIDIA
    torch.float8_e5m2fnuz: tl.float8e5b16 #AMD
}

DTYPE_TO_TRITON = {k:TORCH_DTYPE_TO_TRITON[d] for k,d in DTYPE_TO_TORCH.items()}
