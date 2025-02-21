import triton
from .utils import init_to_zero

# You can add your own configs here
GEMM = {}
GEMM_SPLITK = {}
GEMV_REVSPLITK = {}
GEMV = {}
GEMV_SPLITK = {}

class AUTOTUNE_CONFIG:
	GEMV           = GEMV
	GEMV_REVSPLITK = GEMV_REVSPLITK
	GEMV_SPLITK    = GEMV_SPLITK
	GEMM_SPLITK    = GEMM_SPLITK
	GEMM           = GEMM