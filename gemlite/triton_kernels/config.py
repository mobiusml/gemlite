# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import sys
if sys.version_info < (3, 12): import imp
else: import importlib as imp
from typing import Union

class AUTOTUNE_ENABLE:
	GEMV           = "fast" #"max", "fast", "default" 
	GEMV_REVSPLITK = "fast"
	GEMV_SPLITK    = "fast"
	GEMM_SPLITK    = "fast"
	GEMM           = "fast"
	USE_CUDA_GRAPH = False

def reload_all_modules():
	#Avoid circular imports
	from . import gemm_A16fWnO16f_int32packing
	from . import gemm_splitK_A16fWnO16f_int32packing
	from . import gemm_splitK_persistent_A16fWnO16f_int32packing
	from . import gemv_A16fWnO16f_int32packing
	from . import gemv_splitK_A16fWnO16f_int32packing
	from . import gemv_revsplitK_A16fWnO16f_int32packing

	imp.reload(gemm_A16fWnO16f_int32packing)
	imp.reload(gemm_splitK_A16fWnO16f_int32packing)
	imp.reload(gemm_splitK_persistent_A16fWnO16f_int32packing)
	imp.reload(gemv_A16fWnO16f_int32packing)
	imp.reload(gemv_splitK_A16fWnO16f_int32packing)
	imp.reload(gemv_revsplitK_A16fWnO16f_int32packing)

def set_autotune(config: Union[dict, str, bool], **kwargs):
	if(type(config) == str):
		for key in ['GEMV', 'GEMV_REVSPLITK', 'GEMV_SPLITK', 'GEMM_SPLITK', 'GEMM']:
			setattr(AUTOTUNE_ENABLE, key, config.lower())

	if(type(config) == bool):
		for key in ['GEMV', 'GEMV_REVSPLITK', 'GEMV_SPLITK', 'GEMM_SPLITK', 'GEMM']:
			setattr(AUTOTUNE_ENABLE, key, "max" if config else "default")

	if(type(config) == dict):
		for key in config:
			setattr(AUTOTUNE_ENABLE, key, config[key])

	reload_all_modules()

	if('use_cuda_graph' in kwargs):
		AUTOTUNE_ENABLE.CUDA_GRAPH: bool = kwargs['use_cuda_graph']
