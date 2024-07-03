# pip install ninja;
# pip uninstall torch -y; pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121;
# pip install hqq;
# apt-get install libncurses5 -y; pip install bitblas;

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 ipython3 

######################################################################################################################
import bitblas
import torch
from gemlite import GemLiteMatmul, DType
from hqq.core.quantize import *
from hqq.backends.torchao import patch_hqq_to_aoint4

device = 'cuda:0'
dtype  = torch.float16

torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.coordinate_descent_tuning = True
######################################################################################################################
#Bitblas with dummy weights
from bitblas import Matmul
class BitBlassLinear(torch.nn.Module):
	def __init__(self, weight, w_shift, w_scale, nbits=4, group_size=-1, batch_size=1, bias=None, device=device, compute_dtype=torch.float16):
		super().__init__()

		#In/Out tensors params
		self.compute_dtype = compute_dtype
		self.device = device
		self.dtype_str = str(self.compute_dtype ).split('.')[-1]

		#Shapes
		self.batch_size = batch_size
		self.shape = weight.shape
		self.in_features, self.out_features = self.shape[::-1]

		#Bias
		self.bias = bias 
		if(self.bias is not None):
			if(type(self.bias) is torch.Tensor):
				self.bias = self.bias.to(dtype=self.compute_dtype, device=self.device)
			if(type(self.bias) is torch.nn.Parameter):
				self.bias.data = self.bias.data.to(dtype=self.compute_dtype, device=self.device)

		#Quant params
		self.group_size = self.in_features if(group_size==-1) else group_size
		self.nbits = nbits

		storage_nbit = 8  # assume int8 storage
		n_float_per_elem = storage_nbit // self.nbits

		matmul_config = bitblas.MatmulConfig(
			M=self.batch_size,
			N=self.out_features,
			K=self.in_features,
			fast_decoding=True,
			A_dtype=self.dtype_str,
			W_dtype="int8" if (self.nbits==8) else f"uint{self.nbits}",
			accum_dtype=self.dtype_str,
			out_dtype=self.dtype_str,
			layout="nt",
			with_bias=self.bias is not None,
			group_size=self.group_size,
			with_scaling=False if (self.nbits==8) else True, #True
			with_zeros=False if (self.nbits==8) else True, #True
			zeros_mode="original",
		)

		self.matmul = Matmul(matmul_config)
		self.w_scale = w_scale
		self.w_shift = w_shift

		#Fake data: todo use asym_quant of weight
		if(self.nbits==8):
			self.qweight = (W * w_scale_f).to(torch.int8)
		else:
			self.qweight = torch.randint(0, 2**self.nbits - 1, size=(self.out_features, self.in_features // n_float_per_elem), dtype=torch.uint8, device=self.device)
		self.scales  = torch.ones((self.out_features, self.in_features // self.group_size), dtype=self.compute_dtype, device=self.device)*w_scale
		self.zeros   = torch.ones((self.out_features, self.in_features // self.group_size), dtype=self.compute_dtype, device=self.device)*w_shift

	def forward(self, x):
		out = torch.empty([x.shape[0], self.out_features], dtype=x.dtype, device=x.device)
		if(self.nbits==8):
			self.matmul.forward(A=x, W=self.qweight, bias=self.bias, output=out)
		else:
			self.matmul.forward(A=x, W=self.qweight, scale=self.scales, zeros=self.zeros, bias=self.bias, output=out)
		return out

######################################################################################################################
import torch, time, gc
import numpy as np 

#Shapes
batch_size = 1
shapes = [
(batch_size, 4096*6, 4096*6),
(batch_size, 4096*5, 4096*5),
(batch_size, 4096*4, 4096*4), 
(batch_size, 14336, 14336), 
(batch_size, 4096*2, 4096*2), 
(batch_size, 4096*2, 4096),  
(batch_size, 4096, 4096*2), 
(batch_size, 4096, 4096), 
]

from triton.testing import do_bench
def eval_time_triton(fct, params): 
	return do_bench(lambda: fct(**params)) 

def eval_time_cuda_event(fct, params, iters=500):
	#https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L113-L117
	cache_size = 256 * 1024 * 1024
	cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=device)

	t = []
	for i in range(iters):
		cache.zero_()
		torch.cuda.synchronize()
		start = torch.cuda.Event(enable_timing=True) 
		end   = torch.cuda.Event(enable_timing=True)
		start.record()
		_ = fct(**params)
		end.record()
		torch.cuda.synchronize()
		t.append(start.elapsed_time(end))

	return np.mean(t[-iters//2:])

eval_time = eval_time_triton 


########################################################################################################################################
nbits = 4 #8, 4, 2

for b, K, N  in shapes:
	x = torch.randn((b, K), device=device, dtype=dtype).contiguous()/10.
	W = torch.randn((N, K), device=device, dtype=dtype).contiguous()/10.

	x_maxabs_val = 1. 
	w_maxabs_val = 1. 

	x		   = x_maxabs_val*(x / x.abs().max())
	x_scale_f   = 127. / x_maxabs_val;
	x_scale_fp8 = torch.tensor(448. / x_maxabs_val, device=W.device);
	
	W		  = w_maxabs_val*(W / W.abs().max())
	w_scale_f  = 127. / w_maxabs_val;
	w_scale_f8 = torch.tensor(448. / w_maxabs_val, device=W.device);
	W_int	  = (W * w_scale_f).to(torch.int8)
	W_uint	 = (W * w_scale_f + 127).to(torch.uint8)
	W_fp8	  = (W * w_scale_f8).to(torch.float8_e4m3fn)

	################################################################################################
	#re-quantize to compare quantized error
	x_int  = torch.round(x * x_scale_f).to(torch.int8).contiguous()
	x	  = (x_int.float() / x_scale_f).to(dtype).contiguous()
	x_bf16 = x.to(torch.bfloat16)
	assert (x.float() - (x_int.float() / x_scale_f)).abs().mean().item() <= 1e-4

	w_shift   = 2**(nbits - 1) - 1
	w_scale_f = 2**(nbits - 1) - 1

	W_uint = torch.randint(0, 2**nbits + (-1 if nbits==8 else 0), (N, K), device=device, dtype=torch.uint8).contiguous() 
	W	  = ((W_uint.float() - w_shift) / w_scale_f).to(dtype).contiguous()
	assert (W.float() - ( (W_uint.float() - w_shift)/w_scale_f)).abs().mean().item() <= 1e-3

	W_int  = (W_uint.int() - w_shift).to(torch.int8).contiguous()
	assert (W.float() - (W_int.float()/w_scale_f)).abs().mean().item() <= 1e-3

	#W_uint = (W * w_scale_f + 127).to(torch.uint8)
	W_int32_packed = GemLiteMatmul.pack_warped_int32(W_uint, nbits=nbits) 

	################################################################################################
	#Fucntions
	#Default torch matmul 
	def f_ref(x, W):
		return torch.matmul(x, W.T)

	#BitBlas
	f_bitblas = None
	if(nbits in [8, 4, 2, 1]):
		bitblas_linear = BitBlassLinear(weight=W, w_shift=w_shift, w_scale=w_scale_f, nbits=nbits, group_size=-1, batch_size=b)
		def f_bitblas(x, W):
			return bitblas_linear.forward(x)

	#TorchAO tinyGemm
	f_aoint4 = None
	if(nbits==4):
		dummy_linear = torch.nn.Linear(1, 1, bias=False)
		dummy_linear.weight.data = W;
		quant_config = BaseQuantizeConfig(nbits=nbits, group_size=None, quant_zero=False, quant_scale=False, axis=1)
		hqq_layer   = HQQLinear(dummy_linear, quant_config=quant_config, compute_dtype=torch.bfloat16, device='cuda:0')
		hqq_layer.name = 'dummy'
		aoint4_linear = patch_hqq_to_aoint4(hqq_layer, None)
		def f_aoint4(x, W):
			return aoint4_linear.forward(x)

	#GemLite
	gemlite_fp16_fp16  = GemLiteMatmul(W_nbits=nbits, input_dtype=DType.FP16, output_dtype=DType.FP16).forward
	def f_cuda(x, W, w_shift, w_scale):
		return gemlite_fp16_fp16(x, W, w_shift, w_scale)

	gemlite_int8_int32 = GemLiteMatmul(W_nbits=nbits, input_dtype=DType.INT8, output_dtype=DType.INT32).forward
	def f_cuda_int(x, W, w_shift):
		return gemlite_int8_int32(x, W, w_shift)

	#torch fp8
	def f_fp8(x, W_fp8):
		return torch._scaled_mm((x * x_scale_fp8).to(torch.float8_e4m3fn), W_fp8.T, out_dtype=dtype, scale_a=1/x_scale_fp8 , scale_b=1/w_scale_f8)[0]

	#torch compile for fp16 x int8
	@torch.compile()
	def f_torch_compile(x, W_int):
		return (x @ W_int.T.to(x.dtype)) / (w_scale_f)

	################################################################################################
	#Eval time
	cuda_time	  = eval_time(f_cuda,     {'x':x,     'W':W_int32_packed, 'w_shift':w_shift, 'w_scale':w_scale_f}) 
	cuda_int_time = eval_time(f_cuda_int, {'x':x_int, 'W':W_int32_packed, 'w_shift':w_shift}) 
	
	torch_time	  = eval_time(f_ref, {'x':x, 'W':W})

	bitblas_time  = eval_time(f_bitblas, {'x':x, 'W':W}) if(nbits in [8, 4, 2, 1]) else None
	
	#fp8_time     = eval_time(f_fp8, {'x':x, 'W_fp8':W_fp8}) if (nbits==8) else None
	aoint4_time   = eval_time(f_aoint4, {'x':x_bf16, 'W':W}) if(nbits==4) else None
	if(nbits==8): 
		torch_compile_time = eval_time(f_torch_compile, {'x':x, 'W_int':W_int}) 
		mixed_compile	   = f_torch_compile(x, W_int).flatten()
	else:
		torch_compile_time, mixed_compile = None, None
	
	fp16 = f_ref(x, W).flatten()
	#fp8  = f_fp8(x, W_fp8).flatten()
	cuda	 = f_cuda(**{'x':x, 'W':W_int32_packed, 'w_shift':w_shift, 'w_scale':w_scale_f}).to(dtype).flatten() 
	cuda_int = f_cuda_int(**{'x':x_int, 'W':W_int32_packed, 'w_shift':w_shift}).flatten().float()/(x_scale_f * w_scale_f) 

	print('----------------------------------------------------------------------')
	print("Shape:", str(b) + 'x' + str(K) + ' , ' + str(K) + 'x' + str(N))

	if(aoint4_time): 
		print('aoint4 vs fp16 |', 'speed-up', str(np.round(torch_time/aoint4_time, 4)) + 'x')
	if(bitblas_time):
		print('bitblas vs fp16 |', 'speed-up', str(np.round(torch_time/bitblas_time, 4)) + 'x')
	if(torch_compile_time):
		print('mixed_torch compile vs fp16 |', 'speed-up', str(np.round(torch_time/torch_compile_time, 4)) + 'x' , '| mean error', torch.abs(mixed_compile - fp16).mean().item(), '| max error', torch.abs(mixed_compile - fp16).max().item())
	#print('mixed_fp8 vs fp16 |', 'speed-up', str(np.round(torch_time/fp8_time, 4)) + 'x', '| mean error', torch.abs(fp8 - fp16).mean().item(), '| max error', torch.abs(fp8 - fp16).max().item())
	print('gemlite_mixed_A16Wn vs fp16 |', 'speed-up', str(np.round(torch_time/cuda_time, 4)) + 'x' , '| mean error', torch.abs(cuda - fp16).mean().item(), '| max error', torch.abs(cuda - fp16).max().item())
	print('gemlite_mixed_A8Wn vs fp16 |', 'speed-up', str(np.round(torch_time/cuda_int_time, 4)) + 'x' , '| mean error', torch.abs(cuda_int - fp16).mean().item(), '| max error', torch.abs(cuda_int - fp16).max().item())
	print()

	torch.cuda.empty_cache()
	gc.collect()


