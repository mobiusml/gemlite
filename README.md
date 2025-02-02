# GemLite

<div align="center" style="margin-bottom: 1em;">
<h2>Triton Kernels for Efficient Low-Bit Matrix Multiplication</h2>

  <img src="images/gemlite%20banner.png" alt="GemLite Logo" height="150">
  
  [![Twitter][mobius-twitter-badge]][mobius-twitter]

  Made with ❤ by the team at [Mobius Labs](https://www.mobiuslabs.com/) for  'Aana' (ആന : Elephant) suite of multimodal product.  
  
</div>

**GemLite** is a collection of Triton kernels designed for efficient low-bit matrix multiplication, emphasizing simplicity and reusability. It provides a practical solution for achieving significant performance gains, delivering up to **7-8x faster prefill** and **3-6x faster decoding** compared to default Torch AO kernels. For more detailed benchmarks, check the [Performance](#performance) section.

GemLite strikes the perfect balance between **flexibility** and **performance**, allowing users to easily use and modify the codebase to develop high-performance kernels optimized for their specific hardware. We have included multiple versions of the kernels to maximize performance across different matrix shapes.

The project started with CUDA kernels, but we have switched to <a href="https://github.com/triton-lang/triton/">Triton</a> for enhanced flexibility. For the old CUDA version, please refer to <a href="https://github.com/mobiusml/gemlite/tree/stable_cuda_only">this branch.</a>

### Result Teaser 
| End-to-end Performance (Llama3 8-bit)              | Matmul Performance (A16W8)               |
| --------------------------------------------------- | ---------------------------------------- |
| ![End to End Performance](https://github.com/mobiusml/gemlite/blob/master/images/llama3_8bit.svg) | ![Matmul Performance](https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_32768x32768_4090RTX.svg) |

Extensive performance results across different bitwidths, batch sizes, and devices are available in the [Performance](#performance) section below.

# Table of Contents
- [Recent Highlights](#recent-highlights)
- [Getting Started](#getting-started)
- [Deep Dive](#deep-dive)
- [Performance](#performance)
- [Talks and Resources](#talks-and-resources)
- [Contributing](#contributing)

# Recent Highlights
- GemLite is now integrated with <a href="https://github.com/pytorch/ao">TorchAO</a>/<a href="https://github.com/sgl-project/sglang">SGLang</a> for 4-bit quantization. Check-out the <a href="https://pytorch.org/blog/accelerating-llm-inference/">blogpost</a>!
- **Major performance improvement**: especially on the A100 and H100.
- **Flexible bitpacking**: use 8-bit packing for improved batched performance on the A100 and H100 with packed data.
- **Autotune caching**: save/load the best autotune configs across all the kernels with a single line of code.
- **Helper functions**: helper functions make it easier to get started, especially useful for dynamic quantization.  
- **New GEMV RevSplitK algorithm**: outperforms GEMM Split-K and GEMV for batch-size=1 with packed data.
- **Channel-wise scaling**: Added support for channel-wise scaling for weights, activations, and both.
- **Precision support**: Includes FP16 x Wn, FP8 x FP8, FP8 x Wn, INT8 x INT8 and INT8 x Wn.
- **torch.compile() support**.


# Getting Started
## Installation
### Latest Stable Version
```
pip install gemlite
```
### Latest
```
pip install git+https://github.com/mobiusml/gemlite/
```

## Usage
```Python
from gemlite.core import DType, GemLiteLinear, GEMLITE_ACC_DTYPE

#Set accumulation dtype (only do this once)
GEMLITE_ACC_DTYPE[DType.FP16] = DType.FP32 #For A100/H100 (default)
#GEMLITE_ACC_DTYPE[DType.FP16] = DType.FP16 #For 3090/4090

#Main constructor
gemlite_linear = GemLiteLinear(
    W_nbits, #weight quantization bitwidth. supported: [8, 4, 2, 1]
    group_size=group_size, # any group_size divisible by 32 - enable autotune for group_size < 128 (!)
    in_features=in_features, # input size
    out_features=out_features, #ouput size
    input_dtype=DType.FP16, #FP16, FP8, INT8
    output_dtype=DType.FP16, #FP16, FP32, FP8, INT32
    scaled_activations=False, #If the activations are scaled or not
)

#Packing: we follow the same format as hqq (https://github.com/mobiusml/hqq/)
gemlite_linear.pack(W_q, scales, zeros, bias, packing_bitwidth=32) #32-bit packing by default

#For activation quantization you need to override this function which should return the activation scales:
#gemlite_linear.scale_activations = f(x: torch.Tensor) -> x_scaled: torch.Tensor, scales: torch.Tensor # x ~ x_scaled * scaled

#Forward
out = gemlite_linear(x)
```
### Helper Functions
Additionally, we offer helper functions that operate as follows:

```Python
from gemlite.helper import *

#Non-packed 8-bit weights (INT8 or FP8)
gemlite_linear = A16W8(device='cuda:0').from_linear(linear_layer) #FP16 activations
gemlite_linear = A8W8_int8_dynamic(device='cuda:0').from_linear(linear_layer) #INT8 activations
gemlite_linear = A8W8_fp8_dynamic(device='cuda:0').from_linear(linear_layer) #FP8 activations

#Packed weights for 4-bit/2-bit/1-bit (HQQ format)
gemlite_linear = A16Wn(device='cuda:0').from_hqqlinear(hqqlinear_layer) #FP16 activations
gemlite_linear = A8Wn_dynamic(device='cuda:0').from_hqqlinear(hqqlinear_layer) #FP8 activations

```
### Config Caching
Triton autotuning can be time-consuming. To accelerate this process, we provide tools to automatically cache and load the optimal autotuning configurations for all kernels:
```Python
import gemlite
gemlite.core.GEMLITE_TRITON_RESTRICT_M = True #Restrict the batch-size to powers of 2 if True
gemlite.core.GemLiteLinear.cache_config('gemlite_config.json') #Cache- run this over multiple batch-sizes
gemlite.core.GemLiteLinear.load_config('gemlite_config.json') #Load
``` 
Ensure that you have one JSON cache file per GPU model. When the cache is loaded, the kernels will skip autotuning, leading to a faster startup time.

## Deep Dive
We implement various versions of the Triton kernels: 
* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_A16fWnO16f_int32packing.py">GEMV</a></b>: This GEMV kernel splits the activations into 1D chunks, performs the dot product using `tl.sum`, and accumulates via atomic addition. It is primarily intended for use with small batch sizes (M < 16). As `tl.atomic_add` does not support bfloat16, this kernel is limited to float16.

* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_A16fWnO16f_int32packing.py">GEMM</a></b>: This GEMM kernel is implemented similarly to <a href="https://github.com/fpgaminer/GPTQ-triton">GPTQ-triton</a>. Since it uses tensor cores, activations must be padded with zeros along the batch dimension to fit at least 16 rows. It supports both float32 and float16 accumulation for fp16 inputs, but only float32 accumulation for bfloat16.

* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_splitK_A16fWnO16f_int32packing.py">GEMM Split-K</a></b>: This Split-K GEMM kernel is implemented similarly to <a href="https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py">the gptq Split-K version</a>. We build on the gemm version above and add another dimension in the grid which splits the K dimension into multiple jobs that calculate partial sums, which are atomically added and finally stored. Split-K performs particularly well for batched LLM decoding (batch-size between 1 and 32). 

* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py">Gemv RevSplit-K</a></b>: 
This newly proposed algorithm in GemLite operates in contrast to the GEMM Split-K approach, but within a GEMV context. By doubling the workload per Triton program launched in the GEMV kernel, it reduces the frequency of loading scales/zeros and lowers the number of threads needed. As a result, this method delivers the best performance for batch-size=1 decoding. 

All kernels are flexible, supporting 8, 4, 2, and 1-bit weight precisions as well as both fp16 and int8/fp8 activations.

## Limitations
* All kernels require a minimum group-size of 32.
* The default accumulation DType for FP16 inputs is FP16. If you encounter precision issues, you can try <a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/core.py#L28">reverting to FP32</a>.
* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py">Gemv RevSplit-K</a></b>, which is the default kernel for batch-size=1, does not work with 1-bit weights packed as 32-bit with a group-size of 32. In this case, you should use 8-bit bitpacking via `.pack(...,packing_bitwidth=8)`, or revert to using the `GEMV` kernel instead.

## Performance
### End-2-End Performance
We present various end-2-end Llama results generated with <a href="https://github.com/pytorch/ao/tree/main/torchao/_models/llama">gptfast</a>. GemLite leads to up to 7-8x faster prefill and 3-6x faster decoding compared to the default torchao kernels:

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/llama3_8bit.svg" alt="llama3_8bit.svg" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/llama3_8bit_dynamic.svg" alt="llama3_8bit_dynamic.svg" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/llama3_4bit.svg" alt="llama3_4bit.svg" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/llama2_prefill.svg" alt="llama2_prefill.svg" style="width:98%">
  </div>
 </center>
</div> 


We also run comparison with VLLM's MarlinHQQ kernel which supports asymmetric quantization with a group size of 64. GemLite matches or even outperforms the highly optimized CUDA kernel end-2-end.
<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/llama3_vllm.svg" alt="llama3_vllm.svg" style="width:98%">
  </div>
 </center>
</div> 

### Matmul Performance
We present performance results across various batch sizes on the RTX 4090. Performance is measured as the speed-up relative to A16W16 (fp16 `torch.matmul`). You can reproduce these results by running `examples/benchmark_triton.py` after installing the necessary dependencies via `install_dependencies.sh`.

<details>
<summary>8-bit Weights</summary>


<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_4096x4096_4090RTX.svg" alt="8bit_gs=infeatures_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_8192x8192_4090RTX.svg" alt="8bit_gs=infeatures_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_16384x16384_4090RTX.svg" alt="8bit_gs=infeatures_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_32768x32768_4090RTX.svg" alt="8bit_gs=infeatures_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 


</details>


<details>
<summary>4-bit Weights</summary>
<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_4096x4096_4090RTX.svg" alt="4bit_gs=128_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_8192x8192_4090RTX.svg" alt="4bit_gs=128_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_16384x16384_4090RTX.svg" alt="4bit_gs=128_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_32768x32768_4090RTX.svg" alt="4bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 
</details>


<details>
<summary>2-bit Weights</summary>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_4096x4096_4090RTX.svg" alt="2bit_gs=128_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_8192x8192_4090RTX.svg" alt="2bit_gs=128_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_16384x16384_4090RTX.svg" alt="2bit_gs=128_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_32768x32768_4090RTX.svg" alt="2bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 
</details>

## Talks and Resources
Check out the talk lead author <a href="https://github.com/mobicham/">Dr. Hicham Badri</a> gave about GemLite at [GPU MODE](https://www.youtube.com/watch?v=7c3c3bCGzKU&t=4838s&ab_channel=GPUMODE). You can also find the slides [here](https://docs.google.com/presentation/d/1R9B6RLOlAblyVVFPk9FtAq6MXR1ufj1NaT0bjjib7Vc/edit#slide=id.g310b85e2148_0_135).

Please note that GemLite is under active development, and the content discussed in the talk may evolve as the library continues to improve.

## Contributing
Contributions are always welcome! Please feel free to raise issues, submit pull requests, or start a discussion.

If you're looking to integrate GemLite with major inference and AI libraries, we'd love to hear about it!

[mobius-twitter-badge]: https://img.shields.io/twitter/follow/Mobius_Labs?style=social
[mobius-twitter]: https://twitter.com/Mobius_Labs
