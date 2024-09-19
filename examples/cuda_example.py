import torch 
import numpy as np 

from gemlite.core import GemLiteLinearCUDA, DType

device = 'cuda:0'
in_features, out_features = 4096*1, 4096*2
W_nbits = 4

############################################################################################
#Fp16 x Wn -> Fp16
gemlite_linear = GemLiteLinearCUDA(W_nbits, group_size=in_features, in_features=in_features, out_features=out_features, input_dtype=DType.FP16, output_dtype=DType.FP16)

# Pack
W_q    = torch.randint(0, 2**W_nbits, (in_features, out_features), dtype=torch.uint8, device=device)
scales = torch.randn((1, in_features), dtype=gemlite_linear.compute_dtype, device=device) / 100.
zeros  = 7.
gemlite_linear.pack(W_q.T, scales, zeros, None);

# Equivalent dequantized matrix
_scales = scales
if(isinstance(scales, torch.Tensor)):
    if(scales.numel()==W_q.shape[0]):
        _scales = scales.view(-1, 1)
    if(scales.numel()==W_q.shape[1]):
        _scales = scales.view(1, -1)
W = ((W_q.float() - zeros)*_scales).to(torch.float16)

# Matmul
x = torch.randn((1, in_features), dtype=gemlite_linear.compute_dtype, device=device)/10.
y_q = gemlite_linear.forward(x)
y_ref = torch.matmul(x, W)
print("FP16 x Wn -> FP16 | ", "Mean Absolute Error:", (y_q - y_ref).abs().mean().item())


############################################################################################
#Int8 x Wn -> Int32
gemlite_linear = GemLiteLinearCUDA(W_nbits, group_size=1, in_features=in_features, out_features=out_features, input_dtype=DType.INT8, output_dtype=DType.INT32)

#Pack
W_q    = torch.randint(0, 2**W_nbits, (in_features, out_features), dtype=torch.uint8, device=device)
shift  = 2**W_nbits // 2 - 1 #symmetric Int4
gemlite_linear.pack(W_q.T, None, shift, None);

# Equivalent dequantized matrix
W = ((W_q.float() - shift)).to(torch.float16)

# Matmul
x = torch.randint(-7, 7, (1, in_features), dtype=torch.int8, device=device)
y_q = gemlite_linear.forward(x)
y_ref = torch.matmul(x.to(torch.float16), W.to(torch.float16))
print("Int8 x Wn -> Int32 | ", "Mean Absolute Error:", (y_q - y_ref).abs().mean().item())
