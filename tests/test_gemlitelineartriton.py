#python -m unittest test_gemlitelineartriton.py

import unittest
import torch
from gemlite.core import GemLiteLinearTriton, DType, set_autotune

device = 'cuda:0'
matmul_types = ['GEMV', 'GEMV_SPLITK', 'GEMV_REVSPLITK'] + ['GEMM_SPLITK', 'GEMM']
set_autotune(dict([(m, False) for m in matmul_types]), exhaustive=False, use_cuda_graph=False)

def gen_data(in_features, out_features, W_nbits, group_size, dtype=torch.float16):

	W_q = torch.randint(0, 2**W_nbits - 1, (out_features, in_features), device=device).to(torch.uint8)

	shape  = (out_features, in_features)
	gs     = W_q.numel() // group_size
	scales = torch.ones((gs, 1), device=device, dtype=dtype) * 0.001
	zeros  = torch.zeros((gs, 1), device=device, dtype=dtype) * ((2**W_nbits - 1)//2)
	W      = ((W_q.reshape([-1, group_size]) - zeros) * scales).to(torch.float8_e4m3fn).to(dtype)

	zeros = torch.mean(W_q.reshape([-1, group_size]).float() - (W / scales).float(), axis=1, keepdim=True).to(dtype)
	W     = ((W_q.reshape([-1, group_size]).to(dtype) - zeros) * scales)
	W     = W.reshape(shape)

	return W, W_q, scales, zeros


in_features, out_features = 4096, 4096
batch_sizes               = [1, 4]
W_nbits, group_size       = 4, 128 #128 / in_features
W, W_q, scales, zeros     = gen_data(in_features, out_features, W_nbits=W_nbits, group_size=group_size)

def scale_fct(x, max_val, w_dtype):
    x_shape  = x.shape
    out_x    = x.view(-1, x.shape[-1]) 
    scaled_x = torch.abs(out_x).amax(axis=1, keepdim=True) / max_val
    out_x    = torch.round(out_x / scaled_x).to(dtype=w_dtype)
    return out_x.view(x_shape), scaled_x

class TestGemLiteLinearTriton(unittest.TestCase):

	def test_fp16xfp16(self):
		gemlite_linear = GemLiteLinearTriton(W_nbits=16, 
						group_size=None, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP16, 
						output_dtype=DType.FP16,
						scaled_activations=False)

		gemlite_linear.pack(W, None, None, None);

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0 and gemlite_linear.channel_scale_mode == 0)
		#Use non-contiguous when data is not packed
		self.assertTrue(gemlite_linear.data_contiguous == False)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.)
			y_ref = torch.matmul(x.half(), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_fp16xWn_asymmetric(self):
		#FP16 x Wn / asymmetric 
		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP16, 
						output_dtype=DType.FP16)


		gemlite_linear.pack(W_q, scales, zeros, None);

		if(group_size == in_features):
			#Weights are unpacked() then shift only if group_size == in_features (1) otherwise (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1 and gemlite_linear.channel_scale_mode == 1) or 
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 0)) 
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 0)

		#Use-contiguous when data is packed
		self.assertTrue(gemlite_linear.data_contiguous == True)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.
			y_ref = torch.matmul(x.half(), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_symmetric_no_activation_scaling(self):
		#INT8 x Wn - symmetric / no scaling activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, #only channelwise is supported 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=False) 


		_scales = torch.randn((out_features, 1), dtype=torch.float16, device=device) * 1e-4
		gemlite_linear.pack(W_q, scales=_scales, zeros=7, bias=None);

		#Weights are unpacked() then shifted by 7
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Since the scales are channel-wise, we perform scaling post K-sum
		self.assertTrue(gemlite_linear.channel_scale_mode == 1)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = (torch.randint(-10, 10, (batch_size, in_features), device=device)).to(torch.int8)
			y_ref = torch.matmul(x.half(), ((W_q.half() - 7) * _scales).T) 
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales=None, zeros=7, bias=None);

		def scaled_activations(x):
			return scale_fct(x, max_val=127, w_dtype=torch.int8)

		gemlite_linear.scale_activations = scaled_activations
		gemlite_linear.meta_dtype        = DType.FP32

		#Weights are unpacked() then shifted by 7
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Activations only are scaled
		self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		tol = 5e-3

		for batch_size in batch_sizes:
			x = torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 20.
			
			_x, _x_scaled = scaled_activations(x)
			y_ref = torch.matmul(_x.half(), (W_q.half() - 7).T) * _x_scaled

			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	def test_int8Wn_scaled_weights_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=in_features,  #only channel-wise supported
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)

		_scales = torch.randn((out_features, 1), dtype=torch.float16, device=device) * 1e-4
		gemlite_linear.pack(W_q, scales=_scales, zeros=7, bias=None);

		#Scaling activations
		def scaled_activations(x):
			return scale_fct(x, max_val=127, w_dtype=torch.int8)
		gemlite_linear.scale_activations = scaled_activations

		#Weights are unpacked() then shifted by 7 if group_size == in_features (1), otherwise (3)
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Activations only are scaled if group_size != in_features (2) otherwise bot are scales merged (3)
		self.assertTrue(gemlite_linear.channel_scale_mode == 3)

		tol = 1e-3

		for batch_size in batch_sizes:
			shape = W_q.shape
			x = torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.
			_x, _x_scaled = scaled_activations(x)
			y_ref = torch.matmul(_x.half(), ((W_q.half() - 7) * _scales).T) * _x_scaled
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))



	def test_fp8xfp8(self):
		#FP8 x FP8 - no scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=None, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP8, 
						output_dtype=DType.FP16,
						scaled_activations=False)


		gemlite_linear.pack(W.to(torch.float8_e4m3fn), None, None, None)

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#No channel-wise scaling
		self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
			y_ref = torch.matmul(x.half(), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_fp8xfp8_scaled_weights_scaled_activations(self):
		#FP8 x FP8 - both activations and weights are scaled

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=in_features, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP8, 
						output_dtype=DType.FP16,
						scaled_activations=True)


		_scales = torch.randn((1, out_features), dtype=torch.float16, device=device) * 1e-4
		gemlite_linear.pack(W.to(torch.float8_e4m3fn), scales=_scales, zeros=None, bias=None);

		#Scaling activations
		scales_x = torch.ones((1, 1), dtype=torch.float16, device='cuda:0') * 0.1
		def scaled_activations(x):
			return x, scales_x
		gemlite_linear.scale_activations = scaled_activations

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#Both activations and weights are scales
		self.assertTrue(gemlite_linear.channel_scale_mode == 3)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			shape = W.shape
			x = (torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
			y_ref = torch.matmul(x.half(), W.T) * (_scales * scales_x)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_fp8xWn_scaled_activations(self):
		#FP8 x Wn - asymmetric, with activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP8, 
						output_dtype=DType.FP16,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales, zeros, None);

		if(group_size == in_features):
			#weight unpacking and shift if group_size == in_features else (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1) and (gemlite_linear.channel_scale_mode == 3) or
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 2))
		else:
			#activations and weights are scaled psot accumulation if group_size==in_features else (2)
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		#Scaling activations
		def scaled_activations(x):
			return scale_fct(x, max_val=448, w_dtype=torch.float8_e4m3fn)
		gemlite_linear.scale_activations = scaled_activations

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn).half()
			_x, _scaled_x = scaled_activations(x)
			y_ref = torch.matmul(_x.half(), W.T) * _scaled_x
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_fp8xWn_no_activation_scaling(self):
		#FP8 x Wn - asymmetric, no activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP8, 
						output_dtype=DType.FP16,
						scaled_activations=False)

		gemlite_linear.pack(W_q, scales, zeros, None)

		if(group_size == in_features):
			#Weight shift only if group_size==in_features else (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1 and gemlite_linear.channel_scale_mode == 1) or
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 0))
		else:
			#weight scaling only - post accumulator if group_size==in_features else (0) 
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
			y_ref = torch.matmul(x.half(), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))
