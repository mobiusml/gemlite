// Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
//********************************************************

#include "helper.cu"

//**************************************************************
//fp16 x n-bit as 32-bit packed, mixed fp16 accumulation with half2
__device__ __forceinline__ void gemv_A16fWniO16f_fp16acchalf2_int32pack_core_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                  const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                  const float w_zero, const float w_scale, 
                                  const unsigned int unpack_mask, const size_t elements_per_sample, 
                                  const size_t* __restrict__ loc_shifts, const uint8_t* __restrict__ q_shifts)  
{

  //Set warp params
  const size_t group_id   = threadIdx.x / threads_per_group;
  const size_t group_col  = blockIdx.x * cols_per_block + group_id;
  const size_t warp_iters = div_ceil(W_rows, threads_per_group);
  const size_t group_lane_id = threadIdx.x % threads_per_group;
  if (group_col >= W_cols) {return;}

  //Shared memory
  extern __shared__ half x_shared[];
  size_t x_smem_size = div_ceil(x_cols, threads_per_block); //x_cols / W_rows

  #pragma unroll
  for (size_t i = 0; i < x_smem_size; ++i) {
    size_t x_idx = i * threads_per_block + threadIdx.x;
    x_shared[x_idx] = x[x_idx];   
  }
  __syncthreads();

  //Main loop: fp16 acc                                          
  half2 x_2, w_2;
  const half2 w_shift_half2 = __half2half2((half) (-w_zero));
  half2 sum_half2 = __make_half2(0.f);
  size_t k;
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx   = i * threads_per_group + group_lane_id;
    const size_t W_idx   = x_idx + group_col * W_rows;

    const size_t q_index = (W_idx / W_idx_div)*threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]);

    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j+=2){
      k=j;
      x_2.x = x_shared[x_idx + loc_shifts[k]];
      w_2.x = static_cast<half>((W_q >> q_shifts[k]) & unpack_mask);

      k++;
      x_2.y = x_shared[x_idx + loc_shifts[k]];
      w_2.y = static_cast<half>((W_q >> q_shifts[k]) & unpack_mask);
      sum_half2 = __hfma2(x_2, __hadd2(w_2, w_shift_half2), sum_half2);
    }
  }

  //fp32 warp reduction
  half sum = warpReduceSum<half>(__reduce_sum(sum_half2) / static_cast<half>(w_scale), threads_per_group);
  if (group_lane_id == 0) {y[group_col] = sum;}
}


//**************************************************************
//fp16 x n-bit as 32-bit packed, fp32 accumulation with float4
__device__ __forceinline__ void gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                  const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                  const float w_zero, const float w_scale, 
                                  const unsigned int unpack_mask, const size_t elements_per_sample, 
                                  const size_t* __restrict__ loc_shifts, const uint8_t* __restrict__ q_shifts)  
{

  //Set warp params
  const size_t group_id  = threadIdx.x / threads_per_group;
  const size_t group_col = blockIdx.x * cols_per_block + group_id;
  size_t warp_iters      = div_ceil(x_cols, threads_per_group); //x_cols / W_rows
  const size_t group_lane_id = threadIdx.x % threads_per_group;
  if (group_col >= W_cols) {return;}

  //Shared memory
  extern __shared__ half x_shared[];
  size_t x_chunk_size = div_ceil(x_cols, threads_per_block); //x_cols / W_rows

  #pragma unroll
  for (size_t i = 0; i < x_chunk_size; ++i) {
    size_t x_idx = i * threads_per_block + threadIdx.x;
    x_shared[x_idx] = x[x_idx];   
  }
  __syncthreads();

  //Main loop: float4 acc                                         
  float4 x_4, w_4;
  size_t k;
  float4 sum_float4 = __make_float4(0.f);
  const float4 w_shift_float4 = __make_float4(-w_zero);
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx = i * threads_per_group + group_lane_id;
    const size_t W_idx = x_idx + group_col * W_rows;

    const size_t q_index = (W_idx / W_idx_div) * threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]);

    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j+=4){
      k=j;
      x_4.x = static_cast<float>(x_shared[x_idx + loc_shifts[k]]);
      w_4.x = static_cast<float>((W_q >> q_shifts[k]) & unpack_mask);

      k++;
      x_4.y = static_cast<float>(x_shared[x_idx + loc_shifts[k]]);
      w_4.y = static_cast<float>((W_q >> q_shifts[k]) & unpack_mask);

      k++;
      x_4.z = static_cast<float>(x_shared[x_idx + loc_shifts[k]]);
      w_4.z = static_cast<float>((W_q >> q_shifts[k]) & unpack_mask);

      k++;
      x_4.w = static_cast<float>(x_shared[x_idx + loc_shifts[k]]);
      w_4.w = static_cast<float>((W_q >> q_shifts[k]) & unpack_mask);

      sum_float4 = __fmaf4(x_4, __fadd4(w_4, w_shift_float4), sum_float4); 
    }
  }

  //fp32 warp reduction
  float sum = warpReduceSum<float>(__reduce_sum(sum_float4) / (w_scale), threads_per_group);
  if (group_lane_id == 0) {y[group_col] = __float2half(sum);}
}


//**************************************************************
//fp16 x n-bit as 32-bit packed, fp32 accumulation with float
__device__ __forceinline__ void gemv_A16fWniO16f_fp32accfloat_int32pack_core_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                  const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                  const float w_zero, const float w_scale, 
                                  const unsigned int unpack_mask, const size_t elements_per_sample, 
                                  const size_t* __restrict__ loc_shifts, const uint8_t* __restrict__ q_shifts)  
{

  //Set warp params
  const size_t group_id  = threadIdx.x / threads_per_group;
  const size_t group_col = blockIdx.x * cols_per_block + group_id;
  size_t warp_iters      = div_ceil(x_cols, threads_per_group); //x_cols / W_rows
  const size_t group_lane_id = threadIdx.x % threads_per_group;
  if (group_col >= W_cols) {return;}

  //Shared memory
  extern __shared__ half x_shared[];
  size_t x_chunk_size = div_ceil(x_cols, threads_per_block); //x_cols / W_rows

  #pragma unroll
  for (size_t i = 0; i < x_chunk_size; ++i) {
    size_t x_idx = i * threads_per_block + threadIdx.x;
    x_shared[x_idx] = x[x_idx];   
  }
  __syncthreads();

  //Main loop: float acc                                      
  float sum = 0.f;
  float _x, _w;
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  warp_iters = div_ceil(warp_iters, elements_per_sample) * elements_per_sample;

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

  //fp32 warp reduction
  sum = warpReduceSum<float>(sum / (w_scale), threads_per_group);
  if (group_lane_id == 0) {y[group_col] = __float2half(sum);}
}


//******************************************************************************************************************************************************/

//fp16 x 8-bit as 32-bit packed, fp32 accumulation
__global__ void gemv_A16fW8iO16f_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                        const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                        const float w_zero, const float w_scale)  
{
  //Extra params
  const size_t W_nbits = 8;
  const unsigned int unpack_mask = 0xFF; // W_nbits **2 - 1
  const size_t elements_per_sample = 4; //packing_nbits / W_nbits
 
  //Cache
  const size_t loc_shifts[elements_per_sample] = {0, threads_per_group , threads_per_group*2, threads_per_group*3};
  const uint8_t q_shifts[elements_per_sample]  = {24, 16, 8, 0}; //32 - W_nbits*i

  gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel(x, W, y, x_rows, W_rows, x_cols, W_cols, w_zero, w_scale, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}


//fp16 x 4-bit as 32-bit packed, fp32 accumulation
__global__ void gemv_A16fW4iO16f_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                        const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                        const float w_zero, const float w_scale)  
{
  //Extra params
  const size_t W_nbits = 4;
  const unsigned int unpack_mask = 0xf; // W_nbits **2 - 1
  const size_t elements_per_sample = 8; //packing_nbits / W_nbits
 
  //Cache
  const size_t  loc_shifts[elements_per_sample] = {0                  , threads_per_group  , threads_per_group*2, threads_per_group*3,
                                                  threads_per_group*4,  threads_per_group*5, threads_per_group*6, threads_per_group*7};
  const uint8_t q_shifts[elements_per_sample]   = {28, 24, 20, 16, 12, 8, 4, 0}; //32 - W_nbits*i

  gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}

// //fp16 x 3-bit as 32-bit packed, fp32 accumulation
// __global__ void gemv_A16fW3iO16f_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
//                                         const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
//                                         const float w_zero, const float w_scale) 
// {
//   //Extra params
//   const size_t W_nbits = 3;
//   const unsigned int unpack_mask = 0x7; // W_nbits **2 - 1
//   const size_t elements_per_sample = 10; //packing_nbits / W_nbits

//   //Cache
//   const size_t loc_shifts[elements_per_sample] = {0                  ,  threads_per_group ,   threads_per_group*2,  threads_per_group*3,
//                                                   threads_per_group*4,  threads_per_group*5,  threads_per_group*6,  threads_per_group*7,
//                                                   threads_per_group*8,  threads_per_group*9};

//   const uint8_t q_shifts[elements_per_sample] = {29, 26, 23, 20, 17, 14, 11, 8, 5, 2}; 

//   gemv_A16fWniO16f_fp32accfloat_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
// }


//fp16 x 2-bit as 32-bit packed, fp32 accumulation
__global__ void gemv_A16fW2iO16f_kernel(const half* __restrict__ x, const int32_t* __restrict__ W, half* y, 
                                        const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                        const float w_zero, const float w_scale) 
{
  //Extra params
  const size_t W_nbits = 2;
  const unsigned int unpack_mask = 0x3; // W_nbits **2 - 1
  const size_t elements_per_sample = 16; //packing_nbits / W_nbits

  //Cache
  const size_t loc_shifts[elements_per_sample] = {0                  ,  threads_per_group ,   threads_per_group*2,  threads_per_group*3,
                                                  threads_per_group*4,  threads_per_group*5,  threads_per_group*6,  threads_per_group*7,
                                                  threads_per_group*8,  threads_per_group*9,  threads_per_group*10, threads_per_group*11,
                                                  threads_per_group*12, threads_per_group*13, threads_per_group*14, threads_per_group*15};

  const uint8_t q_shifts[elements_per_sample] = {30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0}; 

  gemv_A16fWniO16f_fp32accfloat4_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}


torch::Tensor gemv_A16fWniO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale, size_t W_nbits)
{
    //TORCH_CHECK(W_nbits == 8 | W_nbits == 4 | W_nbits == 2, "Unsupported W_nbits.");
    TORCH_CHECK((warp_size*warps_per_block) <= 1024, "Invalid warp_size / warps_per_block.");

    CHECK_INPUT(x);
    CHECK_INPUT(W);

    size_t x_rows = x.size(0);
    size_t x_cols = x.size(1);
    size_t W_rows = W.size(1);
    size_t W_cols = W.size(0); //matmul(x, W.T) 
    
    const size_t packing_bits = 32; 
    W_rows *= (packing_bits / W_nbits);
    TORCH_CHECK(x_rows == 1, "Only batch-size=1 is supported.");
    TORCH_CHECK(x_cols == W_rows, "Vector cols / Matrix rows mismatch.");
    TORCH_CHECK(W_rows >= threads_per_block, "Invalid W_rows >= threads_per_block.")
    
    auto dev   = x.device();
    auto dtype = c10::ScalarType::Half;
    auto y     = torch::empty({(int) x_rows, (int) W_cols}, torch::TensorOptions().dtype(dtype).device(dev)); 

    //Grid settings with warping
    dim3 block_size(threads_per_block, 1); 
    dim3 grid_size(cdiv(W_cols, cols_per_block), block_size.y);
  
    //Inputs / outputs ptr  
    const half* x_ptr    = reinterpret_cast<const half*>(x.const_data_ptr<at::Half>());
    const int32_t* W_ptr = reinterpret_cast<const int32_t*>(W.const_data_ptr<int32_t>());
    half* y_ptr          = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    //Shared memory size
    size_t shared_mem_size = x_cols * sizeof(half); //W_rows , x_cols

    switch (W_nbits){
      case 8: gemv_A16fW8iO16f_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale); break; 
      case 4: gemv_A16fW4iO16f_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale); break; 
      //case 3: gemv_A16fW3iO16f_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale); break; 
      case 2: gemv_A16fW2iO16f_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero, w_scale); break; 
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}


torch::Tensor gemv_A16fW8iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale){
  return gemv_A16fWniO16f(x, W, w_zero, w_scale, 8);
}

torch::Tensor gemv_A16fW4iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale){
  return gemv_A16fWniO16f(x, W, w_zero, w_scale, 4);
}

// torch::Tensor gemv_A16fW3iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale){
//   return gemv_A16fWniO16f(x, W, w_zero, w_scale, 3);
// }

torch::Tensor gemv_A16fW2iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale){
  return gemv_A16fWniO16f(x, W, w_zero, w_scale, 2);
}
