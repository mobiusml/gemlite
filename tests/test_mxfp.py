#python -m unittest test_mxfp.py

import unittest
import torch
from gemlite import reset_config, set_autotune
from gemlite.triton_kernels.config import KERNEL
from gemlite.helper import *

def is_fp8_supported():
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(0) 
    return capability >= (8, 9)  

device        = 'cuda:0'
compute_dtype = torch.bfloat16 #float16, bfloat16
matmul_types  = ['GEMM_SPLITK', 'GEMM'] #TODO: add GEMV use-cases
reset_config()
set_autotune(False)
KERNEL.ENABLE_CACHING = False

torch.random.manual_seed(0)
in_features, out_features = 4096, 2048
batch_sizes               = [1, 4, 16]
linear_layer              = torch.nn.Linear(in_features=in_features, out_features=out_features, device=device, dtype=compute_dtype, bias=False)
linear_layer.weight.data /= 10.
linear_layer.weight.requires_grad = False

#Pre-cache data for faster processing
input_data = {}
for batch_size in batch_sizes:
	torch.random.manual_seed(0)
	input_data[batch_size] = torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.

class TestGemliteMXFP(unittest.TestCase):
	def eval(self, gemlite_linear, tol: float = 1e-3):
		for batch_size in batch_sizes:
			x = input_data[batch_size]
			y_ref = linear_layer(x)
			for matmul_type in matmul_types:
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_A16W8_MXFP(self):
		gemlite_linear = A16W8_MXFP(device=device, dtype=compute_dtype).from_linear(linear_layer, del_orig=False)
		self.assertTrue(gemlite_linear.W_q.numel() * gemlite_linear.W_q.itemsize == (in_features * out_features))
		self.assertTrue(not gemlite_linear.scaled_activations)
		self.eval(gemlite_linear, tol = 2e-4)

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_A8W8_MXFP_dynamic(self):
		gemlite_linear = A8W8_MXFP_dynamic(device=device, dtype=compute_dtype, post_scale=False).from_linear(linear_layer, del_orig=False)
		self.assertTrue(gemlite_linear.W_q.numel() * gemlite_linear.W_q.itemsize == (in_features * out_features))
		self.assertTrue(gemlite_linear.scaled_activations)
		self.eval(gemlite_linear, tol = 2e-4)

	def test_A16W4_MXFP(self):
		gemlite_linear = A16W4_MXFP(device=device, dtype=compute_dtype).from_linear(linear_layer, del_orig=False)
		self.assertTrue(gemlite_linear.W_q.numel() * gemlite_linear.W_q.itemsize == (in_features * out_features // 2))
		self.assertTrue(not gemlite_linear.scaled_activations)
		self.eval(gemlite_linear, tol = 7e-4)

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_A8W4_MXFP_dynamic(self):
		gemlite_linear = A8W4_MXFP_dynamic(device=device, dtype=compute_dtype).from_linear(linear_layer, del_orig=False)
		self.assertTrue(gemlite_linear.W_q.numel() * gemlite_linear.W_q.itemsize == (in_features * out_features // 2))
		self.assertTrue(gemlite_linear.scaled_activations)
		self.eval(gemlite_linear, tol = 7e-4)

	def test_A4W4_MXFP_dynamic(self):
		gemlite_linear = A4W4_MXFP_dynamic(device=device, dtype=compute_dtype).from_linear(linear_layer, del_orig=False)
		self.assertTrue(gemlite_linear.W_q.numel() * gemlite_linear.W_q.itemsize == (in_features * out_features // 2))
		self.assertTrue(gemlite_linear.scaled_activations)
		self.eval(gemlite_linear, tol = 1e-3)

