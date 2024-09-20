# GemLite
<a href="https://github.com/mobiusml/gemlite/">GemLite</a>  is a collection of straightforward CUDA and Triton kernels for efficient, fused low-bit matrix multiplication. It is specifically designed for <b>simplicity</b> and <b>reasubility</b>.

This project was initiated because we found it challenging to customize the low-bit kernels that are currently available.
<a href="https://github.com/mobiusml/gemlite/">GemLite</a> provides both flexibility and performance, enabling users to easily modify the codebase to develop high-performance kernels tailored to their specific needs.

While <a href="https://github.com/mobiusml/gemlite/">GemLite</a> can outperform the best existing implementations on large matrices, there's still potential for further optimization!

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_ 32768x32768_4090RTX.svg" alt="8bit_gs=infeatures_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 


<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_ 32768x32768_4090RTX.svg" alt="4bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_ 32768x32768_4090RTX.svg" alt="2bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

# Getting Started
## Installation
```
#Install the nightly or your own version
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121;
pip install git+https://github.com/mobiusml/gemlite/
```

## Usage
```Python
from gemlite.core import DType, GemLiteLinear 

#Currently using the Triton backend as the default
gemlite_linear = GemLiteLinear(
    W_nbits, #supported: [8, 4, 2, 1]
    group_size=group_size, # any group_size divisible by 32
    in_features=in_features, # input size
    out_features=out_features, #ouput size
    input_dtype=DType.FP16, #FP16 or BF16
    output_dtype=DType.FP16, #FP16 or BF16
    acc_dtype=DType.FP16, #FP16 or FP32 
)

#Packing: we follow the same format as hqq (https://github.com/mobiusml/hqq/)
gemlite_linear.pack(W_q, scales, zeros, bias)

#Forward
out = gemlite_linear(x)
```
You can explore various examples in the <a href="https://github.com/mobiusml/gemlite/tree/master/examples">examples folder</a>. Before running them, ensure you have installed the necessary dependencies by executing `./install_dependencies.sh`.

# Deep Dive
## Triton
We implement two versions of the Triton kernels: 
* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_A16fWnO16f_int32packing.py">GEMV</a></b>: This GEMV kernel splits both activations and weights into 1D chunks, performs the dot product using `tl.sum`, and accumulates via atomic addition. It is primarily intended for use with small batch sizes (M < 16). As `tl.atomic_add` does not support bfloat16, this kernel is limited to float16.

* <b><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_A16fWnO16f_int32packing.py">GEMM</a></b>: This GEMM kernel is implemented similarly to <a href="https://github.com/fpgaminer/GPTQ-triton">GPTQ-triton</a>. Since it uses tensor cores, activations must be padded with zeros along the batch dimension to fit at least 16 rows. It supports both float32 and float16 accumulation for fp16 inputs, but only float32 accumulation for bfloat16.

Both kernels are flexible, supporting 8, 4, 2, and 1-bit weight precisions.

To achieve optimal performance, it’s crucial to configure the eviction policy correctly. This is especially important in memory-bound scenarios, where we aim to cache activations by setting `eviction_policy="evict_last"`. Float16 accumulation further improves performance in compute-bound scenarios. 

For bitpacking, we adapt the method from the GPTQ Triton V2 implementation, which can be found <a href="https://github.com/LeiWang1999/GPTQModel/blob/main/gptqmodel/nn_modules/qlinear/qlinear_tritonv2.py#L97-L105">here</a>.

### Limitations
* Performance needs improvement for smaller matrices or lower batch sizes, particularly with the GEMV kernel.
* There is a <a href="https://github.com/triton-lang/triton/issues/2637">high overhead</a> when launching Triton kernels, which becomes more noticeable with lighter workloads. Unfortunately, Cudagraphs does not seem to resolve this issue.
* Autotuning is time-consuming, so the current kernels were optimized with a limited set of configurations. More exhaustive autotuning would likely yield better results. If you plan to run these kernels with different settings or devices, consider adding more configurations for better performance.
* Performance has been mainly optimized for the 4090 RTX (see the autotune configs in the kernel files).

### Performance
We present performance results across various batch sizes on the RTX 4096. Performance is measured as the speed-up relative to A16W16 (fp16 `torch.matmul`). You can reproduce these results by running `examples/benchmark_triton.py` after installing the necessary dependencies via `install_dependencies.sh`.

<details>
<summary>8-bit Weights</summary>
<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_ 4096x4096_4090RTX.svg" alt="8bit_gs=infeatures_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_ 8192x8192_4090RTX.svg" alt="8bit_gs=infeatures_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_ 16384x16384_4090RTX.svg" alt="8bit_gs=infeatures_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/8bit_gs=infeatures_ 32768x32768_4090RTX.svg" alt="8bit_gs=infeatures_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 
</details>


<details>
<summary>4-bit Weights</summary>
<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_ 4096x4096_4090RTX.svg" alt="4bit_gs=128_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_ 8192x8192_4090RTX.svg" alt="4bit_gs=128_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_ 16384x16384_4090RTX.svg" alt="4bit_gs=128_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/4bit_gs=128_ 32768x32768_4090RTX.svg" alt="4bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 
</details>

<details>
<summary>2-bit Weights</summary>
<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_ 4096x4096_4090RTX.svg" alt="2bit_gs=128_4096x4096_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_ 8192x8192_4090RTX.svg" alt="2bit_gs=128_8192x8192_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_ 16384x16384_4090RTX.svg" alt="2bit_gs=128_16384x16384_4090RTX" style="width:98%">
  </div>
 </center>
</div> 

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/2bit_gs=128_ 32768x32768_4090RTX.svg" alt="2bit_gs=128_32768x32768_4090RTX" style="width:98%">
  </div>
 </center>
</div> 
</details>

## CUDA
We explain in detail how the implementation works in our <a href="https://mobiusml.github.io/gemlite_blogpost/">blogpost</a>. 
The main idea is similar to some fast GEMV implementations available like <a href="https://github.com/Bruce-Lee-LY/cuda_hgemv">Bruce-Lee-LY's implementation</a> and <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>: process chunks of the input vector within a group of threads (warp) to calculate partial dot products and warp-reduce the final result. 

In our case, each warp processes `cols_per_warp = 1` columns across 32 threads, and each block processes `cols_per_block = 32` columns. There are 3 main steps:
* Step 1: we cache the input vector in the shared memory first, since more than 1 column is processed per block.
* Step 2: each thread within a warp calculates a partial dot product.
* Step 3: warp-reduce to sum the results from the warp threads.

Steps 1 and 3 are standard procedures, whether the weights are quantized or not. The key innovation occurs in step 2, where the weights are dequantized on-the-fly to compute the partial dot product. The performance boost comes from reducing memory reads, not the computation itself, as quantized weights allow us to read less data from global memory.

To ensure the flexibility of the kernels, we use two arrays:
* `loc_shifts`: these are pre-cached thread-level indices, depending on the number of packed elements per quantized weight point.
* `q_shifts` : an array of shifts used for dequantizing the weights.
Since the bitwidth is fixed, only a single unpacking mask is required `W_nbits **2 - 1`.

Here's a small snippet example of step 2:
```C++
  //Main loop: float acc                                      
  float sum = 0.f;
  float _x, _w;
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx   = i * threads_per_group + group_lane_id; //vector index
    const size_t W_idx   = x_idx + group_col * W_rows; //matrix index (un-packed)

    //Map the index from the un-packed matrix into the packed matrix 
    const size_t q_index = (W_idx / W_idx_div) * threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]); //Load 1 quantized weight value

    //Each thread calculates a partial dot product by dequantizing on-the-fly
    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j++){
      _x = static_cast<float>(x_shared[x_idx + loc_shifts[j]]);
      _w = static_cast<float>((W_q >> q_shifts[j]) & unpack_mask) - w_zero; //dequantize
      sum += _x * _w;
    } 
  }
```
For the case of 4-bit and int32 bitpacking, we need the following:
```C++
  //Extra params
  const size_t W_nbits = 4; //4 x 4-bit elements packed in 1 int32
  const unsigned int unpack_mask = 0xf; // W_nbits **2 - 1
  const size_t elements_per_sample = 8; //packing_nbits / W_nbits
 
  //Cache
  const size_t  loc_shifts[elements_per_sample] = {0                  , threads_per_group  , threads_per_group*2, threads_per_group*3,
                                                  threads_per_group*4,  threads_per_group*5, threads_per_group*6, threads_per_group*7};
  const uint8_t q_shifts[elements_per_sample]   = {28, 24, 20, 16, 12, 8, 4, 0}; //32 - W_nbits*i
```
As mentioned in the code above, we first read one quantized element `W[q_index])`, and then dequantize it on-the-fly via a loop using `loc_shifts` and `q_shifts`. To make the kernel compatible with different bitwidths, we simply need to adjust the params above: `unpack_mask`, `elements_per_sample`, `loc_shifts`, `q_shifts`. For odd bitwidths, some zero-padding may be required to ensure the weight shapes are a multiple of `32 / nbits`.

For bitpacking, we use a universal int32 bitpacking approach to maintain flexibility in the code. The key detail is that we pack the elements with a stride matching the number of threads per warp (32). This ensures coalesced memory access, allowing successive threads to read from the cached quantized element  `W[q_index]`.

We provide various implementations of step 2:
* Half-precision input kernels can be found in `cuda/gemv_A16fWnO16f_int32packing.cu`:
  - FP32 accumulation with `float4`: `gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel()`
  - FP32 accumulation with `float`: `gemv_A16fWniO16f_fp32accfloat_int32pack_core_kernel()`
  - FP16 accumulatiom with `half2`: `cuda/gemv_A16fWniO16f_fp16acchalf2_int32pack_core_kernel()`
* Integer (8-bit) input kernels can be found in kernels are in `gemv_A8iWnO32 ickd_int32packing.cu`:
  - INT32 accumulation with `char4` and `dp4a`: `gemv_A8iWniO32i_int32accchar4_int32pack_core_kernel()`
  - INT32 accumulation with `int32`: `gemv_A8iWniO32i_int32accint_int32pack_core_kernel()`

### Performance
Although the kernels are designed for general purposes, they perform well in practice. Below are benchmark numbers for both the RTX 3090 and the RTX 4090. You can reproduce these results using the code `examples/benchmark_cuda.py`.
 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_3090_fp16.png" alt="3090" style="width:49%">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_4090_fp16.png" alt="4090" style="width:48%">
  </div>
 </center>
</div> 

### Limitations
* Only a GEMV (batch size of 1) is supported at the moment.
* Grouping not supported at the moment.
* Support for odd bitwidths, such as 3-bit, is broken due to padding issues that result in the number of rows not being divisible by the number of columns per warp (32). Although it’s possible to pad shared memory with zeros to match the required padding and add an if statement to prevent accessing out-of-range indices, this approach does not yield correct results currently.
* It might be beneficial to adopt the approach used in <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>, which processes a predefined chunk size with threads. This would allow for more flexible use of shared memory.


# Citation 📜
```
@misc{badri2024gemlite,
title  = {Gemlite: Towards Building Custom Low-Bit Fused CUDA Kernels.},
url    = {https://github.com/mobiusml/gemlite},
author = {Hicham Badri, Appu Shaji},
month  = {August},
year   = {2024}
```
 

