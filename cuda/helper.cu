// Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
//********************************************************

#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline  unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

//Custom Dispatcher to support Float, Half, Bfloat16 
#define AT_DISPATCHER_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)  \

#define AT_DISPATCHER(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCHER_CASE(__VA_ARGS__))

//*********************************************************************************************************/
// Warp-level parameters
#define warp_size 32 //32
#define warps_per_block 32 //32
#define cols_per_warp 1 //1

#define threads_per_block (warp_size * warps_per_block)  // <1024
#define cols_per_block (cols_per_warp * warps_per_block) // default 32 cols per block
#define threads_per_group (warp_size / cols_per_warp)  // default 32 threads per group

//*********************************************************************************************************/
// Arithmetics
__device__ __forceinline__ size_t div_ceil(size_t a, size_t b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ uint32_t div_ceil(uint32_t a, uint32_t b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ half2 __make_half2(const half a) {
  half2 c;
  c.x = a; 
  c.y = a; 
  return c;
}

__device__ __forceinline__ char4 __make_char4(const char a) {
  char4 c;
  c.x = a; 
  c.y = a; 
  c.z = a;
  c.w = a;
  return c;
}

__device__ __forceinline__ uchar4 __make_uchar4(const unsigned char a) {
  uchar4 c;
  c.x = a; 
  c.y = a; 
  c.z = a;
  c.w = a;
  return c;
}

__device__ __forceinline__ float2 __make_float2(const float a) {
  float2 c;
  c.x = a; 
  c.y = a; 
  return c;
}

__device__ __forceinline__ float4 __make_float4(const float a) {
  float4 c;
  c.x = a; 
  c.y = a; 
  c.z = a;
  c.w = a;
  return c;
}

__device__ __forceinline__ int4 __make_int4(const int a) {
  int4 c;
  c.x = a; 
  c.y = a; 
  c.z = a;
  c.w = a;
  return c;
}

__device__ __forceinline__ half __reduce_sum(const half2 a) {
  return a.x + a.y;
}

__device__ __forceinline__ float __reduce_sum(const float2 a) {
  return a.x + a.y;
}

__device__ __forceinline__ float __reduce_sum(const float4 a) {
  return a.x + a.y + a.z + a.w;
}

__device__ __forceinline__ int __reduce_sum(const int4 a) {
  return a.x + a.y + a.z + a.w;
}

__device__ __forceinline__ half __reduce_sum(const half2 a, const half2 b) {
  half2 c = __hadd2(a, b);
  return c.x + c.y;
}

__device__ __forceinline__ float4 __fadd4(const float4 a, const float4 b) {
  float4 c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  c.z = a.z + b.z;
  c.w = a.w + b.w;
  return c;
}

__device__ __forceinline__ float2 __fadd2(const float2 a, const float2 b) {
  float2 c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__device__ __forceinline__ float2 __fmul2(const float2 a, const float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

__device__ __forceinline__ float4 __fmul4(const float4 a, const float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

__device__ __forceinline__ float4 __fmaf4(const float4 a, const float4 b, const float4 c) {
  float4 d;
  d.x = __fmaf_rn(a.x, b.x, c.x);
  d.y = __fmaf_rn(a.y, b.y, c.y);
  d.z = __fmaf_rn(a.z, b.z, c.z);
  d.w = __fmaf_rn(a.w, b.w, c.w);
  return d;
}

__device__ __forceinline__ half __mul_reduce_half2x4(const half2 a1, const half2 b1, const half2 a2, const half2 b2)  {
  half2 tmp = __hadd2(__hmul2(a1, b1), __hmul2(a2, b2));
  return tmp.x + tmp.y;
}


__device__ __forceinline__ float __dot(float4 a, float4 b){
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device__ __forceinline__ float __dot(float2 a, float2 b){
  return a.x*b.x + a.y*b.y;
}

__device__ __forceinline__ half __dot(half2 a, half2 b){
  return a.x*b.x + a.y*b.y;
}


//***************************************************************************************************************/
//Warp-reduce

template<typename dtype>
__device__ __forceinline__ dtype warpReduceSum(dtype sum, size_t num_threads) {
  constexpr unsigned int mask = 0xffffffff;
  if (num_threads >= 32) sum += __shfl_xor_sync(mask, sum, 16); 
  if (num_threads >= 16) sum += __shfl_xor_sync(mask, sum, 8); 
  if (num_threads >= 8)  sum += __shfl_xor_sync(mask, sum, 4); 
  if (num_threads >= 4)  sum += __shfl_xor_sync(mask, sum, 2); 
  if (num_threads >= 2)  sum += __shfl_xor_sync(mask, sum, 1); 
  return sum;
}

//*********************************************************************************************************/