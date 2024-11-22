# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
import torch
from torch import Tensor
import numpy as np
from enum import Enum
import math, json
import warnings
from typing import Union, Tuple

#Dtypes
from .dtypes import *

# Triton
import triton.language as tl
from triton.testing import do_bench, do_bench_cudagraph
from .triton_kernels import *

###################################################################################################################################
# CUDA backend
###################################################################################################################################
# CUDA extension
try:
    import gemlite_lib
except:
    gemlite_lib = None
    print('Failed to import gemlite_lib (CUDA backend)')

GEMLITE_GEMV_FP16_INPUT_FP16_OUTPUT = {
    8: gemlite_lib.gemv_A16fW8iO16f,  # (x, W, w_shift, w_scale)
    4: gemlite_lib.gemv_A16fW4iO16f,
    2: gemlite_lib.gemv_A16fW2iO16f,
}

GEMLITE_GEMV_INT8_INPUT_INT32_OUTPUT = {
    8: gemlite_lib.gemv_A8iW8iO32i,  # (x, W, w_shift)
    4: gemlite_lib.gemv_A8iW4iO32i,
    2: gemlite_lib.gemv_A8iW2iO32i,
}

class GemLiteLinearCUDA(torch.nn.Module):
    warp_size = 32
    warps_per_block = 32
    cols_per_warp = 1
    threads_per_group = warp_size // cols_per_warp

    # Input weights W_uint should be uint8 [0, ...]
    def __init__(
        self,
        W_nbits,
        group_size,
        in_features,
        out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
        acc_dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.orig_shape = (out_features, in_features)
        self.W_nbits = W_nbits
        self.group_size = group_size if group_size != -1 else in_features
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.signature = (in_features, out_features, W_nbits, group_size)
        self.compute_dtype = torch.float16
        self.forward_raw = None

        if input_dtype == DType.FP16 and output_dtype == DType.FP16:
            self.kernel_fct = GEMLITE_GEMV_FP16_INPUT_FP16_OUTPUT[self.W_nbits]
            if group_size == 1:
                self.forward_raw = lambda x, W_q, zeros, scales: self.kernel_fct(x, W_q, zeros, scales)
            if group_size == in_features:
                self.forward_raw = lambda x, W_q, zeros, scales: self.kernel_fct(x * scales, W_q, zeros, 1)
            if group_size == out_features:
                self.forward_raw = lambda x, W_q, zeros, scales: self.kernel_fct(x, W_q, zeros, 1) * scales

        if input_dtype == DType.INT8 and output_dtype == DType.INT32:
            self.kernel_fct = GEMLITE_GEMV_INT8_INPUT_INT32_OUTPUT[self.W_nbits]
            if group_size in [1]:
                self.forward_raw = lambda x, W_q, zeros, scales: self.kernel_fct(x, W_q, zeros)

        if self.forward_raw is None:
            raise NotImplementedError(
                "Unsupport configuration: ",
                (
                    ("input_dtype", self.input_dtype),
                    ("output_dtype", self.output_dtype),
                    ("W_nbits", self.W_nbits),
                    ("group_size", self.group_size),
                ),
            )

        self.acc_dtype = None

    # Universal bitpacking with int32
    def pack(self, W_q, scales: torch.Tensor, zeros=0, bias=None):
        tile_size = self.threads_per_group

        step = 32 // self.W_nbits
        pad = int(step * np.ceil(W_q.shape[1] / step) - W_q.shape[1])
        # pad  += int(tile_size*np.ceil(W_q.shape[1]/tile_size) - W_q.shape[1])
        if pad > 0:
            W_q = torch.nn.functional.pad(W_q, pad=(0, pad), value=0)

        W_shape = W_q.shape
        W_q     = W_q.to(torch.int32).reshape(-1, tile_size)

        i, shift = 0, 32
        shift -= self.W_nbits
        W_q_packed = W_q[i::step, :] << shift
        for i in range(1, step):
            shift -= self.W_nbits
            W_q_packed |= W_q[i::step, :] << shift

        self.W_q = W_q_packed.reshape(W_shape[0], W_shape[1] // step)
        if scales is not None:
            self.scales = scales if isinstance(scales, torch.Tensor) else 1 / scales
        else:
            self.scales = None
        self.zeros = zeros
        self.bias = None if (bias is None) else torch.nn.Parameter(bias.to(device=self.W_q.device, dtype=self.compute_dtype))
        self.device = self.W_q.device

        return self

    def unpack(self, W_q_packed, dtype=torch.uint8):
        tile_size  = self.threads_per_group
        step       = 32 // self.W_nbits
        W_shape    = [self.W_q_packed.shape[0], self.W_q_packed.shape[1] * step]
        W_q_packed = self.W_q_packed.view((-1, tile_size))
        W_r        = torch.empty([step * self.W_q_packed.numel() // tile_size, tile_size], dtype=dtype, device=self.W_q_packed.device)
        mask       = 2**self.W_nbits - 1

        shift = 32
        for i in range(0, step):
            shift -= self.W_nbits
            W_r[i::step, :] = (W_q_packed >> shift) & mask

        return W_r.view(W_shape)

    # Main forward pass
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        out = self.forward_raw(x.view(-1, x.shape[-1]), self.W_q, self.zeros, self.scales).view(out_shape)

        if self.bias is not None:
            out += self.bias
        return out


###################################################################################################################################
# Triton backend
###################################################################################################################################
def eval_time_for_auto_mode(fct, params):
    for _ in range(5): fct(*params) #Run first to kick-off Triton autotune
    if(AUTOTUNE_ENABLE.USE_CUDA_GRAPH):
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        out = do_bench_cudagraph(lambda: fct(*params), rep=50, return_mode='mean')
    else:
        out = do_bench(lambda: fct(*params), warmup=200, rep=50, return_mode='mean')
    return out

def get_closest_m(M):
    return 2 ** int(math.ceil(math.log2(M)))

def cache_kernel_config(kernel, prune_keys, config={}):
    kernel_cache = getattr(kernel, 'cache', None)
    if(kernel_cache is not None):
        for k in kernel_cache:
            _k = k[:len(prune_keys)]
            config[str(_k)] = kernel_cache[k].all_kwargs()
    return config

GEMLITE_ACC_DTYPE           = {DType.FP16: DType.FP16, DType.FP8: DType.FP32, DType.FP8e5: DType.FP32, DType.INT8: DType.INT32}
GEMLITE_TRITON_KERNELS      = [gemv_A16fWnO16f, gemv_revsplitK_A16fWnO16f, gemv_splitK_A16fWnO16f, gemm_splitK_A16fWnO16f, gemm_A16fWnO16f] 
GEMLITE_TRITON_MAPPING      = {kernel.matmul_type : kernel for kernel in GEMLITE_TRITON_KERNELS}
GEMLITE_TRITON_CONFIG_CACHE = {}
GEMLITE_TRITON_CACHE        = {}

# Triton
_GROUP_SIZE_WARNED = False;
class GemLiteLinearTriton(torch.nn.Module):
    SUPPORTED_BITS_TRITON = [1, 2, 4, 8, 16]
    SUPPORTED_DTYPES      = [DType.FP16, DType.FP8, DType.FP8e5, DType.INT8]

    def __init__(
        self,
        W_nbits,
        group_size,
        in_features,
        out_features,
        input_dtype = DType.FP16,
        output_dtype = DType.FP16,
        acc_dtype = None,
        scaled_activations = False,
    ):
        global _GROUP_SIZE_WARNED

        super().__init__()
        if W_nbits not in GemLiteLinearTriton.SUPPORTED_BITS_TRITON:
            raise NotImplementedError("Only " + str(GemLiteLinearTriton.SUPPORTED_BITS_TRITON) + " W_nbits are supported.")
        if in_features % 128 != 0 or out_features % 128 != 0:
            raise NotImplementedError("Invalid input shapes")

        group_size = 1 if (group_size is None) else group_size

        assert group_size >= 32, "Only group_size >= 32 is supported."

        # if(group_size < 128 and (_GROUP_SIZE_WARNED is False)):
        #     warnings.warn("Make sure to enable autotuning for group_size lower than 128: `set_autotune({'GEMV_REVSPLITK':True, 'GEMV':True, 'GEMM_SPLITK':True, 'GEMM':True})`")
        #     _GROUP_SIZE_WARNED = True

        self.in_features  = in_features
        self.out_features = out_features
        self.orig_shape   = (out_features, in_features)
        self.W_nbits      = W_nbits
        self.group_size   = group_size
        self.unpack_mask  = 2**self.W_nbits - 1
        self.elements_per_sample = None
        self.signature = (in_features, out_features, W_nbits, group_size)

        self.input_dtype   = input_dtype
        self.output_dtype  = output_dtype
        self.compute_dtype = torch.float16
        self.meta_dtype    = DType.FP16
        self.kernels       = GEMLITE_TRITON_KERNELS

        #Warning: Input dtype should be the same as dequantize() weights dtype.
        if input_dtype not in GemLiteLinearTriton.SUPPORTED_DTYPES:
            raise NotImplementedError("Unsupport input dtype: " + str(self.input_dtype))

        #Accumulation
        self.acc_dtype = GEMLITE_ACC_DTYPE[self.input_dtype] if(acc_dtype is None) else acc_dtype

        #Scales activations
        self.scaled_activations = scaled_activations
        self.scales_x = None

        if(AUTOTUNE_ENABLE.EXHAUSTIVE):
            self.forward = self.forward_auto_with_warmup
        else:
            self.forward = self.forward_auto_no_warmup

        #Default GEMV for packed vs. non-packed data
        self.default_gemv = 'GEMV_REVSPLITK' if (self.W_nbits < 8) else 'GEMV_SPLITK'
            
        #Set torch flags
        torch._dynamo.config.inline_inbuilt_nn_modules = False #2.5.0 fix

    #Override this function to perform dynamic activation quantization
    def scale_activations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x, self.scales_x

    # Pack data, adapted from: following the same logic as: https://github.com/LeiWang1999/AutoGPTQ.tvm/blob/dcd135b9784b9f98235fc91467fe3c3c8afa34fc/auto_gptq/nn_modules/qlinear_triton.py#L413-L419
    def pack_weights_over_rows(self, W_q, W_nbits, packing_bitwidth=32, transpose=True):
        elements_per_sample = packing_bitwidth // W_nbits

        W_q     = W_q.to(torch.int32)
        W_q_out = torch.zeros((W_q.shape[0] // elements_per_sample, W_q.shape[1]), dtype=torch.int32, device=W_q.device) 

        i, row = 0, 0
        while row < W_q_out.shape[0]:
            for j in range(i, i + (packing_bitwidth // W_nbits)):
                W_q_out[row] |= W_q[j] << (W_nbits * (j - i))
            i += elements_per_sample
            row += 1

        if(packing_bitwidth == 8) : W_q_out = W_q_out.to(torch.uint8)
        if(packing_bitwidth == 16): W_q_out = W_q_out.to(torch.int16)
        if(packing_bitwidth == 32): W_q_out = W_q_out.to(torch.int32)

        if(transpose): W_q_out = W_q_out.t()

        return W_q_out, elements_per_sample

    def pack_weights_over_cols(self, W_q, W_nbits, packing_bitwidth=32, transpose=True):
        elements_per_sample = packing_bitwidth // W_nbits

        W_q     = W_q.to(torch.int32)
        W_q_out = torch.zeros((W_q.shape[0], W_q.shape[1] // elements_per_sample), dtype=torch.int32, device=W_q.device) 

        i, col = 0, 0
        while col <  W_q_out.shape[1]: 
            shift = 0
            for j in range(i, i + elements_per_sample):
                W_q_out[:, col] |= (W_q[:, j] << shift)
                shift += W_nbits
            i += elements_per_sample
            col += 1

        if(packing_bitwidth == 8) : W_q_out = W_q_out.to(torch.uint8)
        if(packing_bitwidth == 16): W_q_out = W_q_out.to(torch.int16)
        if(packing_bitwidth == 32): W_q_out = W_q_out.to(torch.int32)

        if(transpose): W_q_out = W_q_out.t()

        return W_q_out, elements_per_sample

    #Make sure to feed UINT8 W_q for packing
    def pack(self, W_q: Tensor, scales: Tensor, zeros: Union[Tensor, int], bias: Union[Tensor, None]=None, fma_mode: bool=False, contiguous: bool=True, packing_bitwidth: int=32):

        #Unpacked weights
        self.W_q = None
        if(W_q.dtype in [torch.float16, torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]):
            if(W_q.dtype == torch.float16): 
                assert self.W_nbits == 16, "Invalid fp16 weights."
            else: 
                assert self.W_nbits == 8, "Invalid 8-bit weights."

            self.W_q = W_q.t() #row-major
            self.elements_per_sample = 1

        if(W_q.dtype == torch.uint8): #Packed weigths
            self.W_q, self.elements_per_sample = self.pack_weights_over_cols(W_q.view(self.orig_shape), W_nbits=self.W_nbits, packing_bitwidth=packing_bitwidth, transpose=True) #Over-K
            #self.W_q, self.elements_per_sample = self.pack_weights_over_rows(W_q.view(self.orig_shape), W_nbits=self.W_nbits, packing_bitwidth=packing_bitwidth, transpose=True) #Over-N

        if(self.W_q is None):
            raise Exception('Weights were not packed, please check your W_q.dtype')

        #Bias / device
        self.bias   = None if (bias is None) else torch.nn.Parameter(bias.to(device=self.W_q.device, dtype=self.compute_dtype))
        self.device = self.W_q.device

        #initial values
        self.W_group_mode       = -1
        self.channel_scale_mode = 0

        #FP16 x FP16 / FP8 x FP8 / INT8 x INT8 - no meta-data case 
        if((scales is None) and (zeros is None)):
            self.zeros  = None
            self.scales = None
            self.W_group_mode = 0
            self.channel_scale_mode = 2 if self.scaled_activations else 0 
            return 

        #The rest of the use-cases require some kind of meta-data
        if(scales is not None):
            self.scales = scales.view((self.out_features, -1)).t()
        else:
            self.scales = None

        #Symmetric no shift
        if(zeros is None):  
            self.zeros = None
            self.W_group_mode = 2
        else:
            #Asymmetric or Symmetric with shift
            if(isinstance(zeros, torch.Tensor)):
                if(fma_mode): #W ~ Wq * scales + zeros
                    self.zeros = (-zeros.float()*scales.float()).to(zeros.dtype).view((self.out_features, -1)).t()
                    self.W_group_mode = 4
                else: #W ~ (Wq - zeros) * scales
                    self.zeros = zeros.view((self.out_features, -1)).t()
                    self.W_group_mode = 3 
            else: #Integer
                self.zeros = int(zeros) 
                if(self.scales is not None):
                    self.W_group_mode = 3 #Symmetric with shift
                else:
                    self.W_group_mode = 1 #Shift only with integer

        assert self.W_group_mode > -1, "Invalid scales/zeros settings."

        #channel-wise scaling 
        self.meta_is_chanenlwise = False if(self.scales is None) else self.scales.numel() == self.out_features 

        ###########################################
        #weight-only
        if((self.scaled_activations == False) and (self.meta_is_chanenlwise == True)):
            self.channel_scale_mode = 1
            self.W_group_mode       = 1 if(self.zeros is not None) else 0 #only with fma_mode=False

        #activation-only
        if((self.scaled_activations == True) and (self.meta_is_chanenlwise == False)):
            self.channel_scale_mode = 2

        #weight + activation mode
        if((self.scaled_activations == True) and (self.meta_is_chanenlwise == True)):
             self.channel_scale_mode = 3
             self.W_group_mode       = 1 if(self.zeros is not None) else 0 #only with fma_mode=False
        ##########################################
            
        #Keep meta pre-processing even with channel-wise scales/zeros

        # #weight-only
        # if((self.scaled_activations == False) and (self.meta_is_chanenlwise == True)):
        #     pass

        # #activation-only
        # if((self.scaled_activations == True) and (self.meta_is_chanenlwise == False)):
        #     self.channel_scale_mode = 2

        # #weight + activation mode
        # if((self.scaled_activations == True) and (self.meta_is_chanenlwise == True)):
        #     self.channel_scale_mode = 2
        ###########################################

        if(self.channel_scale_mode in [1, 3]):
            assert self.W_group_mode not in [3, 4], "Can't use channel_scale_mode with W_group_mode == 3 or 4."

        # if(self.input_dtype == DType.INT8):
        #     assert self.W_group_mode in [1], "Only channel-wise symmetric quantization is supported for INT8 inputs."

        #Dummy values 
        if(isinstance(self.zeros, int)): #Union[Tensor, int] not supported by custom op
            self.zeros = torch.tensor(self.zeros, dtype=torch.int32, device=self.device)
        if(self.zeros is None):
            self.zeros = torch.tensor([], dtype=torch.int32, device=self.device)
        if(self.scales is None):
            self.scales = torch.tensor([], dtype=torch.int32, device=self.device)

        if(self.scales is not None):
            self.meta_dtype = DType.FP32 if self.scales.dtype == torch.float32 else DType.FP16

        #Force contiguous
        if(contiguous):
            self.data_contiguous = True
            self.W_q = self.W_q.contiguous()
            if(isinstance(self.scales, torch.Tensor)):
                self.scales = self.scales.contiguous()
            if(isinstance(self.zeros, torch.Tensor)):
                self.zeros = self.zeros.contiguous()
        else:
            self.data_contiguous = False

        #TODO: Register buffers

        return self

    # Warm up all the selected kernels
    def warmup(self, signature, args):
        global GEMLITE_TRITON_CACHE
        t = [np.inf] * len(self.kernels)
        for i, _kernel in enumerate(self.kernels):
            if signature[0] > 1 and _kernel.matmul_type == "GEMV": #skip gemvs for larger batch-sizes: GEMV 
                continue 
            if signature[0] > 1 and _kernel.matmul_type == "GEMV_SPLITK": #skip gemvs for larger batch-sizes: GEMV_SPLTIK
                continue 
            if signature[0] > 2 and _kernel.matmul_type == "GEMV_REVSPLITK": #skip gemvs for larger batch-sizes: GEMV_REVSPLITK
                continue 
            if signature[0] > 32 and _kernel.matmul_type == "GEMM_SPLITK": #skip SPLIT_K for larger batch-
                continue
            if signature[0] < 16 and _kernel.matmul_type == "GEMM": #skip GEMM for smaller matrices
                continue  
            t[i] = eval_time_for_auto_mode(_kernel.forward, args)

        indx = np.argmin(t)
        GEMLITE_TRITON_CACHE[signature] = {
            "matmul_type": self.kernels[indx].matmul_type,
            "time": t[indx],
            "time_all": list(zip([k.matmul_type for k in self.kernels] , t))
        }

    #Exhaustive search 
    def forward_auto_with_warmup(self, x: Tensor) -> Tensor:
        global GEMLITE_TRITON_CACHE
        x, scaled_x = self.scale_activations(x)
        out_shape = x.shape[:-1] + (self.out_features,)

        x_input = x.view(-1, x.shape[-1])
        args = [
            x_input,
            self.W_q,
            self.scales,
            self.zeros,
            scaled_x,
            self.W_nbits,
            self.group_size,
            self.unpack_mask,
            self.elements_per_sample,
            self.input_dtype.value,
            self.output_dtype.value,
            self.acc_dtype.value,
            self.meta_dtype.value,
            self.channel_scale_mode,
            self.W_group_mode,
            self.data_contiguous,
        ]

        _signature = (get_closest_m(x_input.shape[0]),) + self.signature
        if _signature not in GEMLITE_TRITON_CACHE:
            self.warmup(_signature, args)

        matmul_type = GEMLITE_TRITON_CACHE[_signature]["matmul_type"]
        out         = GEMLITE_TRITON_MAPPING[matmul_type].forward(*args).view(out_shape)

        if self.bias is not None:
            out += self.bias
        return out

    def forward_auto_no_warmup(self, x: Tensor) -> Tensor:
        _batch_size = x.view(-1, x.shape[-1]).shape[0]
        if(_batch_size > 64):
            return self.forward_manual(x, matmul_type='GEMM') #GEMM
        if(_batch_size > 1):
            return self.forward_manual(x, matmul_type='GEMM_SPLITK') #GEMM_SPLITK
        else:
            return self.forward_manual(x, matmul_type=self.default_gemv) #GEMV / GEMV_REVSPLITK / GEMV_SPLITK

    
    def forward_manual(self, x: Tensor, matmul_type: str="GEMM") -> Tensor:
        x, scaled_x = self.scale_activations(x)
        out_shape = x.shape[:-1] + (self.out_features,)

        out = (
            GEMLITE_TRITON_MAPPING[matmul_type]
            .forward(
                x.view(-1, x.shape[-1]),
                self.W_q,
                self.scales,
                self.zeros,
                scaled_x,
                self.W_nbits,
                self.group_size,
                self.unpack_mask,
                self.elements_per_sample,
                self.input_dtype.value,
                self.output_dtype.value,
                self.acc_dtype.value,
                self.meta_dtype.value,
                self.channel_scale_mode,
                self.W_group_mode,
                self.data_contiguous,
            )
            .view(out_shape)
        )

        if self.bias is not None:
            out += self.bias
        return out

    @staticmethod
    def cache_config(filename, prune_keys = ['M', 'N', 'K', 'group_size', 'elements_per_sample']):
        #global GEMLITE_TRITON_MAPPING
        #_GEMLITE_TRITON_MAPPING = GEMLITE_TRITON_CACHE

        config = {}

        try: 
            with open(filename) as json_file:
                config = json.load(json_file)
        except:
            pass

        #Can't use GEMLITE_TRITON_MAPPING for some reason kernel.cache is empty
        _GEMLITE_TRITON_MAPPING = {}
        from .triton_kernels.gemv_A16fWnO16f_int32packing import gemv_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV'] = gemv_A16fWnO16f

        from .triton_kernels.gemv_revsplitK_A16fWnO16f_int32packing import gemv_revsplitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV_REVSPLITK'] = gemv_revsplitK_A16fWnO16f

        from .triton_kernels.gemv_splitK_A16fWnO16f_int32packing import gemv_splitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMV_SPLITK'] = gemv_splitK_A16fWnO16f

        from .triton_kernels.gemm_splitK_A16fWnO16f_int32packing import gemm_splitK_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMM_SPLITK'] = gemm_splitK_A16fWnO16f

        from .triton_kernels.gemm_A16fWnO16f_int32packing import gemm_A16fWnO16f
        _GEMLITE_TRITON_MAPPING['GEMM'] = gemm_A16fWnO16f

        for name in _GEMLITE_TRITON_MAPPING:
            if(name not in config): config[name] = {}
            config[name].update(cache_kernel_config(_GEMLITE_TRITON_MAPPING[name].kernel, prune_keys))

        with open(filename, "w") as json_file: 
            json.dump(config, json_file)

    @staticmethod
    def load_config(filename):
        global GEMLITE_TRITON_CONFIG_CACHE
        try:
            with open(filename) as json_file:
                GEMLITE_TRITON_CONFIG_CACHE = json.load(json_file)
        except:
            pass

###################################################################################################################################
###################################################################################################################################
GemLiteLinear = GemLiteLinearTriton  # Triton by default
