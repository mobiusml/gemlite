// Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
//********************************************************

#include <torch/extension.h>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/script.h>

//Mixed half-precision x low-bit weights: matmul(x_fp16, W_n) / (w_scale) -> fp16 output
torch::Tensor gemv_A16fW8iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale);
torch::Tensor gemv_A16fW4iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale);
//torch::Tensor gemv_A16fW3iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale);
torch::Tensor gemv_A16fW2iO16f(torch::Tensor x, torch::Tensor W, const float w_zero, const float w_scale);

//matmul(x_int8, W_n) -> int32 output / NO normalization
torch::Tensor gemv_A8iW8iO32i(torch::Tensor x, torch::Tensor W, const int w_zero);
torch::Tensor gemv_A8iW4iO32i(torch::Tensor x, torch::Tensor W, const int w_zero);
torch::Tensor gemv_A8iW2iO32i(torch::Tensor x, torch::Tensor W, const int w_zero);

//PY link
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("gemv_A16fW8iO16f", &gemv_A16fW8iO16f, "gemv_A16fW8iO16f");
	m.def("gemv_A16fW4iO16f", &gemv_A16fW4iO16f, "gemv_A16fW4iO16f");
	//m.def("gemv_A16fW3iO16f", &gemv_A16fW3iO16f, "gemv_A16fW3iO16f");
	m.def("gemv_A16fW2iO16f", &gemv_A16fW2iO16f, "gemv_A16fW2iO16f");

	m.def("gemv_A8iW8iO32i", &gemv_A8iW8iO32i, "gemv_A8iW8iO32i");
	m.def("gemv_A8iW4iO32i", &gemv_A8iW4iO32i, "gemv_A8iW4iO32i");
	m.def("gemv_A8iW2iO32i", &gemv_A8iW2iO32i, "gemv_A8iW2iO32i");
}
