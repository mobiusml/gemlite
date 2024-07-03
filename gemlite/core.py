# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************

import torch
import gemlite_lib
import numpy as np 
from enum import Enum

class MatmulType(Enum):
    GEMV = "GEMV"
    GEMM = "GEMM"

class DType(Enum):
    FP16   = "FP16"
    INT8   = "INT8"
    INT32  = "INT32"
    FP16D8 = "FP16D8i" #dynamic quantization

GEMLITE_GEMV_FP16_INPUT_FP16_OUTPUT_INT32_WEIGHTS_MAPPING = {
	8: gemlite_lib.gemv_A16fW8iO16f, #(x, W, w_shift, w_scale)
	4: gemlite_lib.gemv_A16fW4iO16f,
	2: gemlite_lib.gemv_A16fW2iO16f,
}

GEMLITE_GEMV_INT8_INPUT_INT32_OUTPUT_INT32_WEIGHTS_MAPPING = {
	8: gemlite_lib.gemv_A8iW8iO32i, #(x, W, w_shift)
	4: gemlite_lib.gemv_A8iW4iO32i,
	2: gemlite_lib.gemv_A8iW2iO32i,
}

#input_dtype, output_dtype
GEMLITE_GEMV_MAPPING = {
	'FP16': {'FP16': GEMLITE_GEMV_FP16_INPUT_FP16_OUTPUT_INT32_WEIGHTS_MAPPING},
	'INT8': {'INT32': GEMLITE_GEMV_INT8_INPUT_INT32_OUTPUT_INT32_WEIGHTS_MAPPING},
}

GEMLITE_MAPPING ={
	'GEMV': GEMLITE_GEMV_MAPPING,
}

class GemLiteMatmul(torch.nn.Module):
	warp_size		   = 32
	warps_per_block    = 32
	cols_per_warp	   = 1 
	threads_per_group  = warp_size // cols_per_warp

	#Input weights W_uint should be uint8 [0, ...]
	def __init__(self, W_nbits=8, input_shape=None, input_dtype=DType.FP16, output_dtype=DType.FP16): 
		super().__init__()
		self.input_shape  = input_shape
		self.W_nbits      = W_nbits
		self.input_dtype  = input_dtype
		self.output_dtype = output_dtype
		self.matmul_type  = MatmulType.GEMV
		#Todo in the future: use input_shape to determin GEMV/GEMM algo 
		try:
			self.forward = GEMLITE_MAPPING[self.matmul_type.value][self.input_dtype.value][self.output_dtype.value][self.W_nbits]
		except Exception as exp:
			self.forward = None
			print("Unsupported type", exp)

	@classmethod
	#Universal bitpacking with int32
	def pack_warped_int32(self, W_q, nbits):
		tile_size = self.threads_per_group

		step = 32 // nbits
		pad  =  int(step*np.ceil(W_q.shape[1]/step) - W_q.shape[1])
		#pad  += int(tile_size*np.ceil(W_q.shape[1]/tile_size) - W_q.shape[1])  
		if(pad > 0):
			W_q = torch.nn.functional.pad(W_q, pad=(0, pad), value=0)

		W_shape = W_q.shape
		W_q	 = W_q.to(torch.int32)
		W_q	 = W_q.reshape(-1, tile_size)

		i, shift = 0, 32
		shift -= nbits
		W_q_packed = (W_q[i::step, :] << shift)
		for i in range(1, step):
			shift -= nbits
			W_q_packed |= (W_q[i::step, :] << shift)

		W_q_packed = W_q_packed.reshape(W_shape[0], W_shape[1] // step)
		return W_q_packed

	@classmethod
	def unpack_warped_int32(self, W_q_packed, nbits, dtype=torch.uint8):
		tile_size  = self.threads_per_group
		
		step       = 32 // nbits
		W_shape	   = [W_q_packed.shape[0], W_q_packed.shape[1]*step]
		W_q_packed = W_q_packed.reshape((-1, tile_size))
		W_r		   = torch.empty([step * W_q_packed.numel() //tile_size, tile_size], dtype=dtype, device=W_q_packed.device)
		mask	    = 2**nbits - 1

		shift = 32
		for i in range(0, step):
			shift -= nbits
			W_r[i::step,:]  = ((W_q_packed >> shift) & mask)

		return W_r.reshape(W_shape)
