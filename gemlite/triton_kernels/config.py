# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import imp 

class AUTOTUNE_ENABLE:
	GEMV        = True
	GEMM        = False
	GEMM_SPLITK = True

def reload_all_modules():
	#Avoid circular imports
	from . import gemv_A16fWnO16f_int32packing
	from . import gemm_A16fWnO16f_int32packing
	from . import gemm_splitK_A16fWnO16f_int32packing
	MODULES = {'GEMV':[gemv_A16fWnO16f_int32packing], 
			   'GEMM': [gemm_A16fWnO16f_int32packing], 
			   'GEMM_SPLITK':[gemm_splitK_A16fWnO16f_int32packing]
			   }

	for matmul_dtype in MODULES:
		for module in MODULES[matmul_dtype]:
			imp.reload(module)

def set_autotune(matmul_dtypes: dict):
	for key in matmul_dtypes:
		setattr(AUTOTUNE_ENABLE, key, matmul_dtypes[key])
	reload_all_modules()