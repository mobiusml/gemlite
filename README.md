# GemLite
GemLite is a set of simple CUDA kernels for fused low-bit GEMV:
* It is easy to read and customize.
* It's flexible: one core kernel can be used for any n-bit weight quantization.
* Multiple implementations: fp32 accumuation, fp16 accumulation, etc.
* Available with both fp16 and int8 activations.
* No specifc GPU instructions, making it even compatible with older gpus.

The main reason we started this project is because we found that it's very difficult to customize the low-bit kernels currently available. They are either hard-coded for a specific use-case (4-bit with a fixed group-size), only perform well on specific GPUs, or the code is so large, complex and confusion that it's a nightmare to even understand what's going on.

Instead of focusing on very specific and performant use-cases, we provide a set of simple and re-usable kernels that can be used as a good starting point.

## Installation
`pip install git+https://github.com/mobiusml/gemlite.git`

## How it Works
The main idea is similar to some fast GEMV implementations available like <a href="https://github.com/Bruce-Lee-LY/cuda_hgemv">Bruce-Lee-LY/'s hgmev</a> and <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>. They rely on processing chunks of the input vector within a warp and warp-reduce the final result. 

In our case, each warp processes `cols_per_warp = 1` columns across 32 threads, and each block processes `cols_per_block = 32` columns. There are 3 main steps:
* Step 1: we cache the input vector in the shared memory first, since more than 1 column is processed per block.
* Step 2: each thread within a warp calculates a partial dot product.
* Step 3: warp-reduce to sum the results from the threads.

Step 1 and 3 are standard practice, regardless if the weights are quantized or not. The magic happens in step 2, where the weights are dequantized to calculate the partial dot product. The speed is actually coming from the reduced memory reads, not from computation, since we read less data with quantized weights.

In order to make the kernels flexible, we use two arrays:
* `loc_shifts`: these are per-cached thread-level indices, they depend on the number of packed elements per quantized weight point.
* `q_shifts` : array of shifts to use to dequantize the weights. Since the bitwidth is fixed, we only need a single unpacking mask `W_nbits **2 - 1`.

Here's a small snippet of step 2:
```C++
  //Main loop: float acc                                      
  float sum = 0.f;
  float _x, _w;
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx   = i * threads_per_group + group_lane_id;
    const size_t W_idx   = x_idx + group_col * W_rows;

    const size_t q_index = (W_idx / W_idx_div) * threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]);

    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j++){
      _x = static_cast<float>(x_shared[x_idx + loc_shifts[j]]);
      _w = static_cast<float>((W_q >> q_shifts[j]) & unpack_mask) - w_zero;
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
So the idea is, we first read one quantized point `W[q_index])`, and then dequantize it via a loop going through `loc_shifts` and `q_shifts`. To make the kernel work for other bitwidths, we simply need to change the elements above.

The last thing to mention is the bitpacking. We use a universal int32 bitpacking logic to make the code flexible. The main "gotcha" is that, we pack elements with a stride that corresponds to the number of threads per warp (32). This ensures that memory access is coalesced, so that successive threads read from the cached quantized point 
`W[q_index])`.

We provide various implementations of step 2, but we only use one for fp16 activations and one for int8 activations. The variants include:
* FP32 accumulation with `float4`
* FP32 accumulation with `float`
* FP16 accumulatiom with `half2`
* INT32 accumulation with `char4` and `dp4a`
* INT32 accumulation with `int32`

## Performance
Even-though the kernels are general purpose, they tend to perform pretty well. Below bechmark numbers on both the 3090 and 4090 (you can reproduce these numbers with the code snippet `examples/benchmark.py`)
 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_3090_fp16.png" alt="3090" style="width:49%">
	<img src="https://github.com/mobiusml/gemlite/blob/master/images/gemlite_4090_fp16.png" alt="4090" style="width:48%">
  </div>
 </center>
</div> 

## Limitations
* Only a batch-size of 1 is supported for the moment. 
* Only scalar zero-point/scaling for the moment. Channel-wise normalization can be done outside the matmul call, but grouping needs to be added.
* Odd bitwidths like 3-bit is broken because they require padding that makes the number of rows not divisible by the number of columns per warp (32). There's a way to pad the shared memory with zeros to match the padding and add an `if` statement to avoid accessing indices outside the range, but it doesn't give the correct results for the moment.
* Maybe it's a better idea to follow  <a href="https://github.com/wangsiping97/FastGEMV">FastGEMV</a>'s idea that uses a predefined chunk-size to be processed by the threads. This makes using shared memory easier which would allow us to store more things. Currently the size of the shared memory depends on the number of rows, so we can only defined its size outside the kernel.

