__version__ = "0.4.2"
__author__  = 'Dr. Hicham Badri'
__credits__ = 'Mobius Labs GmbH'

from .core import GemLiteLinearTriton, GemLiteLinear, DType, GEMLITE_ACC_DTYPE, GEMLITE_TRITON_CONFIG_CACHE, GEMLITE_TRITON_RESTRICT_M
load_config  = GemLiteLinear.load_config
cache_config = GemLiteLinear.cache_config
reset_config = GemLiteLinear.reset_config