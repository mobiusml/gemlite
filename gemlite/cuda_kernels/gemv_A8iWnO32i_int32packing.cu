// Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
//********************************************************

#include "helper.cu"

//*********************************************************************************
//int8 x n-bit as 32-bit packed, int32 accumulation with dp4a
__device__ __forceinline__ void gemv_A8iWniO32i_int32accchar4_int32pack_core_kernel(const int8_t* __restrict__ x, const int32_t* __restrict__ W, int32_t* y, 
                                  const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                  const int w_zero,  
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
  extern __shared__ int8_t x_shared[];
  size_t x_smem_size = div_ceil(x_cols, threads_per_block); //x_cols / W_rows

  #pragma unroll
  for (size_t i = 0; i < x_smem_size; i++) {
    size_t idx = i * threads_per_block + threadIdx.x;
    x_shared[idx] = x[idx]; 
  }
  __syncthreads();

  //Main loop: char4 acc
  int sum_int = 0.f;                                           
  char4 x_4, w_4;
  size_t k;
  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx   = i * threads_per_group + group_lane_id;
    const size_t W_idx   = x_idx + group_col * W_rows;

    const size_t q_index = (W_idx / W_idx_div)*threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]);

    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j+=4){
      k = j;
      x_4.x = (x_shared[x_idx + loc_shifts[k]]);
      w_4.x = ((W_q >> q_shifts[k]) & unpack_mask) - w_zero;

      k++;
      x_4.y = (x_shared[x_idx + loc_shifts[k]]);
      w_4.y = ((W_q >> q_shifts[k]) & unpack_mask) - w_zero;

      k++;
      x_4.z = (x_shared[x_idx + loc_shifts[k]]);
      w_4.z = ((W_q >> q_shifts[k]) & unpack_mask) - w_zero;

      k++;
      x_4.w = (x_shared[x_idx + loc_shifts[k]]);
      w_4.w = ((W_q >> q_shifts[k]) & unpack_mask) - w_zero;

      sum_int = __dp4a(x_4, w_4, sum_int);
    }
  }

  //int32 warp reduction
  sum_int = warpReduceSum<int>(sum_int, threads_per_group);
  if (group_lane_id == 0) {y[group_col] = sum_int;}
}


//*********************************************************************************
//int8 x n-bit as 32-bit packed, int32 accumulation without dp4a
__device__ __forceinline__ void gemv_A8iWniO32i_int32accint_int32pack_core_kernel(const int8_t* __restrict__ x, const int32_t* __restrict__ W, int32_t* y, 
                                  const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                  const int w_zero,  
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
  extern __shared__ int8_t x_shared[];
  size_t x_smem_size = div_ceil(x_cols, threads_per_block); //x_cols / W_rows

  #pragma unroll
  for (size_t i = 0; i < x_smem_size; i++) {
    size_t idx = i * threads_per_block + threadIdx.x;
    x_shared[idx] = x[idx]; 
  }
  __syncthreads();

  //Main loop: int32 acc
  int sum_int = 0;                                           
  int32_t _w; 
  int8_t _x;

  const uint16_t W_idx_div = elements_per_sample * threads_per_group; //max val 1024

  #pragma unroll
  for (size_t i = 0; i < warp_iters; i += elements_per_sample) {
    const size_t x_idx   = i * threads_per_group + group_lane_id;
    const size_t W_idx   = x_idx + group_col * W_rows;

    const size_t q_index = (W_idx / W_idx_div)*threads_per_group + W_idx % threads_per_group;
    const int32_t W_q    = __ldg(&W[q_index]);

    #pragma unroll
    for (size_t j=0; j < elements_per_sample; j++){
      _x = x_shared[x_idx + loc_shifts[j]];
      _w = ((W_q >> q_shifts[j]) & unpack_mask) - w_zero;
      sum_int += _x * _w;
    }
  }

  //int32 warp reduction
  sum_int = warpReduceSum<int>(sum_int, threads_per_group);
  if (group_lane_id == 0) {y[group_col] = sum_int;}
}


//******************************************************************************************************************************************************/

//int8 x 8-bit as 32-bit packed, int32 accumulation
__global__ void gemv_A8iW8iO32i_kernel(const int8_t* __restrict__ x, const int32_t* __restrict__ W, int32_t* y, 
                                       const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                       const int w_zero) 
{
  //Extra params
  const size_t W_nbits = 8;
  const unsigned int unpack_mask = 0xFF; // W_nbits **2 - 1
  const size_t elements_per_sample = 4; //packing_nbits / W_nbits
 
  //Cache
  const size_t  loc_shifts[elements_per_sample] = {0, threads_per_group , threads_per_group*2, threads_per_group*3};
  const uint8_t q_shifts[elements_per_sample]   = {24, 16, 8, 0}; //32 - W_nbits*i

  gemv_A8iWniO32i_int32accint_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}


//int8 x 4-bit as 32-bit packed, int32 accumulation
__global__ void gemv_A8iW4iO32i_kernel(const int8_t* __restrict__ x, const int32_t* __restrict__ W, int32_t* y, 
                                       const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                       const int w_zero) 
{
  //Extra params
  const size_t W_nbits = 4;
  const unsigned int unpack_mask = 0xf; // W_nbits **2 - 1
  const size_t elements_per_sample = 8; //packing_nbits / W_nbits
 
  //Cache
  const size_t  loc_shifts[elements_per_sample] = {0                 , threads_per_group  , threads_per_group*2, threads_per_group*3,
                                                  threads_per_group*4, threads_per_group*5, threads_per_group*6, threads_per_group*7};
  const uint8_t q_shifts[elements_per_sample]   = {28, 24, 20, 16, 12, 8, 4, 0}; //32 - W_nbits*i

  gemv_A8iWniO32i_int32accint_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}


//int8 x 2-bit as 32-bit packed, int32 accumulation
__global__ void gemv_A8iW2iO32i_kernel(const int8_t* __restrict__ x, const int32_t* __restrict__ W, int32_t* y, 
                                       const size_t x_rows, const size_t x_cols, const size_t W_rows, const size_t W_cols, 
                                       const int w_zero)  
{
  //Extra params
  const size_t W_nbits = 2;
  const unsigned int unpack_mask = 0x3; // W_nbits **2 - 1
  const size_t elements_per_sample = 16; //packing_nbits / W_nbits

  //Cache
  const size_t loc_shifts[elements_per_sample] = {0                  ,  threads_per_group,    threads_per_group*2,  threads_per_group*3,
                                                  threads_per_group*4,  threads_per_group*5,  threads_per_group*6,  threads_per_group*7,
                                                  threads_per_group*8,  threads_per_group*9,  threads_per_group*10, threads_per_group*11,
                                                  threads_per_group*12, threads_per_group*13, threads_per_group*14, threads_per_group*15};

  const uint8_t q_shifts[elements_per_sample] = {30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0}; 
  
  gemv_A8iWniO32i_int32accint_int32pack_core_kernel(x, W, y, x_rows, x_cols, W_rows, W_cols, w_zero, unpack_mask, elements_per_sample, loc_shifts, q_shifts);
}


torch::Tensor gemv_A8iWniO32i(torch::Tensor x, torch::Tensor W, const int w_zero, size_t W_nbits)
{
    TORCH_CHECK(W_nbits == 8 | W_nbits == 4 | W_nbits == 2, "Unsupported W_nbits.");
    TORCH_CHECK((warp_size*warps_per_block) <= 1024, "Invalid warp_sze / warps_per_block.");

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
    auto dtype = c10::ScalarType::Int;
    auto y     = torch::empty({(int) x_rows, (int) W_cols}, torch::TensorOptions().dtype(dtype).device(dev)); 

    //Grid settings with warping
    dim3 block_size(threads_per_block, 1); 
    dim3 grid_size(cdiv(W_cols, cols_per_block), block_size.y);
  
    //Inputs / outputs ptr  
    const int8_t* x_ptr  = reinterpret_cast<const int8_t*>(x.const_data_ptr<int8_t>());
    const int32_t* W_ptr = reinterpret_cast<const int32_t*>(W.const_data_ptr<int32_t>());
    int32_t* y_ptr       = reinterpret_cast<int32_t*>(y.data_ptr<int32_t>());

    //Shared memory size
    size_t shared_mem_size = x_cols * sizeof(int8_t); //W_rows , x_cols

    switch (W_nbits){
      case 8: gemv_A8iW8iO32i_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero); break; 
      case 4: gemv_A8iW4iO32i_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero); break; 
      case 2: gemv_A8iW2iO32i_kernel<<<grid_size, block_size, shared_mem_size>>>(x_ptr, W_ptr, y_ptr, x_rows, x_cols, W_rows, W_cols, w_zero); break; 
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}


torch::Tensor gemv_A8iW8iO32i(torch::Tensor x, torch::Tensor W, const int w_zero){
  return gemv_A8iWniO32i(x, W, w_zero, 8);
}

torch::Tensor gemv_A8iW4iO32i(torch::Tensor x, torch::Tensor W, const int w_zero){
  return gemv_A8iWniO32i(x, W, w_zero, 4);
}

torch::Tensor gemv_A8iW2iO32i(torch::Tensor x, torch::Tensor W, const int w_zero){
  return gemv_A8iWniO32i(x, W, w_zero, 2);
}
