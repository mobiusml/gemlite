# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import sys
if sys.version_info < (3, 12): import imp
else: import importlib as imp

class AUTOTUNE_ENABLE:
	GEMV           = True
	GEMV_REVSPLITK = True
	GEMV_SPLITK    = True
	GEMM_SPLITK    = True
	GEMM           = True
	EXHAUSTIVE     = False
	USE_CUDA_GRAPH = False

def reload_all_modules():
	#Avoid circular imports
	from . import gemv_A16fWnO16f_int32packing
	from . import gemm_A16fWnO16f_int32packing
	from . import gemm_splitK_A16fWnO16f_int32packing
	from . import gemv_revsplitK_A16fWnO16f_int32packing

	MODULES = {'GEMV':[gemv_A16fWnO16f_int32packing], 
			   'GEMM': [gemm_A16fWnO16f_int32packing], 
			   'GEMM_SPLITK':[gemm_splitK_A16fWnO16f_int32packing],
			   'GEMV_REVSPLITK':[gemv_revsplitK_A16fWnO16f_int32packing]
			   }

	for matmul_dtype in MODULES:
		for module in MODULES[matmul_dtype]:
			imp.reload(module)

def set_autotune(matmul_dtypes: dict, **kwargs):
	for key in matmul_dtypes:
		setattr(AUTOTUNE_ENABLE, key, matmul_dtypes[key])
	reload_all_modules()

	if('exhaustive' in kwargs):
		AUTOTUNE_ENABLE.EXHAUSTIVE: bool = kwargs['exhaustive']

	if('use_cuda_graph' in kwargs):
		AUTOTUNE_ENABLE.CUDA_GRAPH: bool = kwargs['use_cuda_graph']
