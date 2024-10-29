#python -m unittest test_gemlitelineartriton.py

import unittest
import torch
from gemlite.core import GemLiteLinearTriton, DType, set_autotune

set_autotune({'GEMV_REVSPLITK':False, 'GEMV':False, 'GEMM_SPLITK':False, 'GEMM':False}, exhaustive=False, use_cuda_graph=False)

device = 'cuda:0'
matmul_types = ['GEMV_REVSPLITK', 'GEMV', 'GEMM_SPLITK', 'GEMM']

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


in_features, out_features = 4096, 4096*2
W_nbits, group_size       = 4, in_features #128
W, W_q, scales, zeros     = gen_data(in_features, out_features, W_nbits=W_nbits, group_size=group_size)

class TestGemLiteLinearTriton(unittest.TestCase):

	def test_fp16xfp16(self):
		gemlite_linear = GemLiteLinearTriton(W_nbits=16, 
						group_size=None, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.FP16, 
						output_dtype=DType.FP16,
						scaled_activations=False)

		gemlite_linear.pack(W, None, None, None, fma_mode=False);

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#No channel-wise scaling
		self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 1e-3

		x = (torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.)
		y_ref = torch.matmul(x.half(), W.T)
		for matmul_type in matmul_types:
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


		gemlite_linear.pack(W_q, scales, zeros, None, fma_mode=False);

		if(group_size == in_features):
			#Weights are unpacked() then shift only if group_size == in_features (1) otherwise (3)
			self.assertTrue(gemlite_linear.W_group_mode == 1) 
			#Since the scales are channel-wise, we perform scaling post K-sum
			self.assertTrue(gemlite_linear.channel_scale_mode == 1)
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 0) 

		tol = 1e-3

		x = torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.
		y_ref = torch.matmul(x.half(), W.T)
		for matmul_type in matmul_types:
			y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
			err   = (y_ref - y_gem).abs().mean().item()
			self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_symmetric_no_activation_scaling(self):
		#INT8 x Wn - symmetric / no scaling activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=False) 


		gemlite_linear.pack(W_q, scales=scales, zeros=7, bias=None);

		if(group_size == in_features):
			#Weights are unpacked() then shifted by 7
			self.assertTrue(gemlite_linear.W_group_mode == 1) 
			#Since the scales are channel-wise, we perform scaling post K-sum
			self.assertTrue(gemlite_linear.channel_scale_mode == 1)
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		x = (torch.randint(-10, 10, (1, in_features), device=device)).to(torch.int8)

		tol = 1e-3

		shape = W_q.shape
		y_ref = torch.matmul(x.half(), ((W_q.half().reshape([-1, group_size]) - 7) * scales).reshape(shape).T) 
		for matmul_type in matmul_types:
			y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
			err   = (y_ref - y_gem).abs().mean().item()
			self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales=None, zeros=7, bias=None);

		#Scaling activations
		scales_x = torch.ones((1, 1), dtype=torch.float16, device='cuda:0') * 0.001
		def scaled_activations(x):
			return x, scales_x
		gemlite_linear.scale_activations = scaled_activations

		#Weights are unpacked() then shifted by 7
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Activations only are scaled
		self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		tol = 1e-3

		x = (torch.randint(-10, 10, (1, in_features), device=device)).to(torch.int8)
		y_ref = torch.matmul(x.half(), (W_q.half() - 7).T) * scales_x
		for matmul_type in matmul_types:
			y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
			err   = (y_ref - y_gem).abs().mean().item()
			self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8Wn_scaled_weights_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales=scales, zeros=7, bias=None);

		#Scaling activations
		scales_x = torch.ones((1, 1), dtype=torch.float16, device='cuda:0') * 0.001
		def scaled_activations(x):
			return x, scales_x
		gemlite_linear.scale_activations = scaled_activations

		if(group_size == in_features):
			#Weights are unpacked() then shifted by 7 if group_size == in_features (1), otherwise (3)
			self.assertTrue(gemlite_linear.W_group_mode == 1) 
			#Activations only are scaled if group_size != in_features (2) otherwise bot are scales merged (3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 3)
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3) 
			self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		tol = 1e-3

		shape = W_q.shape
		x = (torch.randint(-10, 10, (1, in_features), device=device)).to(torch.int8)
		y_ref = torch.matmul(x.half(), ((W_q.half().reshape([-1, group_size]) - 7) * scales).reshape(shape).T) * scales_x
		for matmul_type in matmul_types:
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


		gemlite_linear.pack(W.to(torch.float8_e4m3fn), None, None, None, fma_mode=False);

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#No channel-wise scaling
		self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		x = (torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
		y_ref = torch.matmul(x.half(), W.T)
		for matmul_type in matmul_types:
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

		gemlite_linear.pack(W.to(torch.float8_e4m3fn), scales=scales, zeros=None, bias=None);

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

		shape = W.shape
		x = (torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
		y_ref = torch.matmul(x.half(), W.T) * (scales * scales_x)
		for matmul_type in matmul_types:
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


		gemlite_linear.pack(W_q, scales, zeros, None, fma_mode=False);

		if(group_size == in_features):
			#weight unpacking and shift if group_size == in_features else (3)
			self.assertTrue(gemlite_linear.W_group_mode == 1)
			#activations and weights are scaled psot accumulation if group_size==in_features else (2)
			self.assertTrue(gemlite_linear.channel_scale_mode == 3)
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		#Scaling activations
		scales_x = torch.ones((1, 1), dtype=torch.float16, device='cuda:0') * 0.01
		def scaled_activations(x):
			return x, scales_x
		gemlite_linear.scale_activations = scaled_activations

		tol = 5e-3 #needs higher tolerance with fp8

		x = (torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
		y_ref = torch.matmul(x.half(), W.T) * scales_x
		for matmul_type in matmul_types:
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

		gemlite_linear.pack(W_q, scales, zeros, None, fma_mode=False);

		if(group_size == in_features):
			#Weight shift only if group_size==in_features else (3)
			self.assertTrue(gemlite_linear.W_group_mode == 1)
			#weight scaling only - post accumulator if group_size==in_features else (0) 
			self.assertTrue(gemlite_linear.channel_scale_mode == 1)
		else:
			self.assertTrue(gemlite_linear.W_group_mode == 3)
			self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		x = (torch.randn((1, in_features), dtype=torch.float16, device=device) / 10.).to(torch.float8_e4m3fn)
		y_ref = torch.matmul(x.half(), W.T)
		for matmul_type in matmul_types:
			y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
			err   = (y_ref - y_gem).abs().mean().item()
			self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))
