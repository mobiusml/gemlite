# Gemlite
Gemlite is a collection of simple CUDA kernels for fused low-bit GEMV:
* It is easy to read and customize.
* It's flexible: one core kernel can be used for any n-bit weight quantization.
* Multiple implementations: fp32 accumuation, fp16 accumulation, etc.
* Available with both fp16 and int8 activations.
* No specifc GPU instructions, making it even compatible with older gpus.

The main reason we started this project is because we found that it is very difficult to customize the low-bit kernels currently available. They are either hard-coded for a specific use-case (4-bit with a fixed group-size), only perform well on specific GPUs, or the code is so large and complex that it's difficult to understand.

Instead of focusing on very specific and performant use-cases, we provide a set of simple and re-usable kernels that can be used as a good starting point.

## How it Works
We explain in detail how the implementation works in our <a href="https://mobiusml.github.io/gemlite_blogpost/">blogpost</a>. 
The main idea is similar to some fast GEMV implementations available like <a href="https://github.com/Bruce-Lee-LY/cuda_hgemv">Bruce-Lee-LY's implementation</a> and <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>: process chunks of the input vector within a group of threads (warp) to calculate partial dot products and warp-reduce the final result. 

In our case, each warp processes `cols_per_warp = 1` columns across 32 threads, and each block processes `cols_per_block = 32` columns. There are 3 main steps:
* Step 1: we cache the input vector in the shared memory first, since more than 1 column is processed per block.
* Step 2: each thread within a warp calculates a partial dot product.
* Step 3: warp-reduce to sum the results from the warp threads.

Step 1 and 3 are standard practice, regardless if the weights are quantized or not. The magic happens in step 2, where the weights are dequantized on-the-fly to calculate the partial dot product. The speed is actually coming from the reduced memory reads, not from the computation, since we read less data from the global memory with quantized weights.

In order to make the kernels flexible, we use two arrays:
* `loc_shifts`: these are pre-cached thread-level indices, they depend on the number of packed elements per quantized weight point.
* `q_shifts` : array of shifts to use to dequantize the weights.
Since the bitwidth is fixed, we only need a single unpacking mask `W_nbits **2 - 1`.

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
As mentioned in the code above, we first read one quantized element `W[q_index])`, and then dequantize it on-the-fly via a loop using `loc_shifts` and `q_shifts`. To make the kernel work for other bitwidths, we simply need to change the params above: `unpack_mask`, `elements_per_sample`, `loc_shifts`, `q_shifts`. Odd bitwidths would require some zero-padding to make the weight shapes a multiple of `32 / nbits`.

For bitpacking, we use a universal int32 bitpacking approach to make the code flexible. The main "gotcha" is that, we pack the elements with a stride that corresponds to the number of threads per warp (32). This ensures that memory access is coalesced, so that successive threads read from the cached quantized element 
`W[q_index])`.

We provide various implementations of step 2:
* Half-precision input kernels can be found in `cuda/gemv_A16fWnO16f_int32packing.cu`:
  - FP32 accumulation with `float4`: `gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel()`
  - FP32 accumulation with `float`: `gemv_A16fWniO16f_fp32accfloat_int32pack_core_kernel()`
  - FP16 accumulatiom with `half2`: `cuda/gemv_A16fWniO16f_fp16acchalf2_int32pack_core_kernel()`
* Integer (8-bit) input kernels can be found in kernels are in `gemv_A8iWnO32i_int32packing.cu`:
  - INT32 accumulation with `char4` and `dp4a`: `gemv_A8iWniO32i_int32accchar4_int32pack_core_kernel()`
  - INT32 accumulation with `int32`: `gemv_A8iWniO32i_int32accint_int32pack_core_kernel()`

## Using the Implemented Kernels
If you want to test or use the kernels directly, you can follow the example below:
```Python
from gemlite import GemLiteMatmul, DType

#Bitpack
W_int32_packed = GemLiteMatmul.pack_warped_int32(W_uint, nbits=nbits)
#W_uint is the uint8 quantized weight with values ranging from 0 to 2**nbits - 1

#Fp16 input -> Fp16 output
gemlite_fp16_fp16  = GemLiteMatmul(W_nbits=nbits, input_dtype=DType.FP16, output_dtype=DType.FP16).forward
out = gemlite_fp16_fp16(x_fp16, W_int32_packed, w_zero, w_scale)
#equivalent to torch.matmul(x_fp16, (W_uint.float() - w_zero) / w_scale_f)

#Int8 input -> Int32 output
gemlite_int8_int32 = GemLiteMatmul(W_nbits=nbits, input_dtype=DType.INT8, output_dtype=DType.INT32).forward
out = gemlite_int8_int32(x_int8, W_int32_packed, w_zero)
#equivalent to torch.matmul(x_int8, (W_uint.int() - w_zero))
```
The code above should work with `nbits=8, 4, 2` and a batch-size of 1 for the input `x`.

## Performance
Even-though the kernels are general purpose, they tend to perform well. Below bechmark numbers on both the 3090 and 4090 (you can reproduce these numbers with the code snippet `examples/benchmark.py`).
 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_3090_fp16.png" alt="3090" style="width:49%">
	<img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_4090_fp16.png" alt="4090" style="width:48%">
  </div>
 </center>
</div> 

## Limitations
* Only a batch-size of 1 is supported. 
* Only scalar zero-point/scaling for the moment. Channel-wise normalization can be done outside the matmul call, but grouping support is missing.
* Odd bitwidths support like 3-bit is broken because they require padding that makes the number of rows not divisible by the number of columns per warp (32). There's a way to pad the shared memory with zeros to match the padding and add an `if` statement to avoid accessing indices outside the range, but it doesn't give the correct results for the moment.
* Maybe it's a better to follow <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>'s idea that uses a predefined chunk-size to be processed by the threads. This would allow a more flexible usage of the shared memory.


### Citation 📜
```
@misc{badri2024gemlite,
title  = {Gemlite: Towards Building Custom Low-Bit Fused CUDA Kernels.},
url    = {https://github.com/mobiusml/gemlite},
author = {Hicham Badri, Appu Shaji},
month  = {August},
year   = {2024}
```
 

