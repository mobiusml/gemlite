import torch 

def check_valid(x, W, quant_linear, tol=1e-3):
    y_ref = torch.matmul(x, W.T)
    y_q   = quant_linear(x)
    assert (y_ref - y_q).abs().mean() < tol, 'Outputs do not match'

############################################################################################
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

in_features, out_features = 4096*4, 4096*4
#W_nbits, group_size = 8, in_features 
W_nbits, group_size = 4, 64 
#W_nbits, group_size = 2, 64
compute_dtype = torch.float16 #float16 / bfloat16

linear       = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False, device='cpu')
quant_config = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size, quant_zero=False, quant_scale=False, axis=1)
hqq_layer    = HQQLinear(linear, quant_config=quant_config, compute_dtype=compute_dtype, device='cuda:0', del_orig=False) 

orig_shape   = (out_features, in_features)
W            = hqq_layer.dequantize().reshape(orig_shape)
############################################################################################

from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]
gemlite_linear = GemLiteLinearTriton(W_nbits, 
                                    group_size=group_size, 
                                    in_features=in_features, 
                                    out_features=out_features, 
                                    input_dtype=gemlite_dtype, 
                                    output_dtype=gemlite_dtype)

W_q           = hqq_layer.unpack(dtype=torch.uint8).view(orig_shape)
scales        = hqq_layer.meta['scale']
zeros         = hqq_layer.meta['zero']
gemlite_linear.pack(W_q, scales, zeros, None)

x = torch.randn((8, in_features), dtype=gemlite_linear.compute_dtype, device='cuda:0')/10.
check_valid(x, W, gemlite_linear)

