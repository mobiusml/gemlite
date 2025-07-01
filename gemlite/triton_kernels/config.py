# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import sys
if sys.version_info < (3, 12): import imp
else: import importlib as imp
from typing import Union

MATMUL_DTYPES = ['GEMV', 'GEMV_REVSPLITK', 'GEMV_SPLITK', 'GEMM_SPLITK', 'GEMM']

class AUTOTUNE:
	GEMV           = "fast" #"max", "fast", "default" 
	GEMV_REVSPLITK = "fast"
	GEMV_SPLITK    = "fast"
	GEMM_SPLITK    = "fast"
	GEMM           = "fast"
	USE_CUDA_GRAPH = False

class KERNEL:
	ENABLE_CACHING = False
	CACHE_SIZE     = 256 

def reload_all_modules():
	#Avoid circular imports
	from . import gemm
	from . import gemm_splitK
	from . import gemm_splitK_persistent
	from . import gemv
	from . import gemv_splitK
	from . import gemv_revsplitK

	imp.reload(gemm)
	imp.reload(gemm_splitK)
	imp.reload(gemm_splitK_persistent)
	imp.reload(gemv)
	imp.reload(gemv_splitK)
	imp.reload(gemv_revsplitK)

def set_kernel_caching(enable: bool):
	KERNEL.ENABLE_CACHING = enable

def set_autotune(config: Union[dict, str, bool], **kwargs):
	if(type(config) == str):
		for key in MATMUL_DTYPES:
			setattr(AUTOTUNE, key, config.lower())

	if(type(config) == bool):
		for key in MATMUL_DTYPES:
			setattr(AUTOTUNE, key, "max" if config else "default")

	if(type(config) == dict):
		for key in config:
			setattr(AUTOTUNE, key, config[key])

	if('use_cuda_graph' in kwargs):
		setattr(AUTOTUNE, 'USE_CUDA_GRAPH', bool(kwargs['use_cuda_graph']))

	reload_all_modules()