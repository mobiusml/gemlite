# SPDX-License-Identifier: Apache-2.0
# Unit and adversarial tests for GEMV_DIRECT (M=1 A16W2 direct-accumulation GEMV)

import unittest
import torch
import triton
import math

# Backward compatibility shims for test environments running PyTorch < 2.7 / Triton < 3.3
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8
if not hasattr(triton, "set_allocator"):
    triton.set_allocator = lambda fn: None

from gemlite import reset_config, set_autotune
from gemlite.core import (
    GemLiteLinearTriton,
    DType,
    TORCH_TO_DTYPE,
    forward_functional,
    get_matmul_type,
    GEMLITE_MATMUL_TYPES,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TestGemvDirect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for Triton kernel tests")
        reset_config()
        set_autotune(False)

    def _test_shape_correctness(self, in_features, out_features, group_size=64, dtype=torch.bfloat16, fma_mode=True):
        torch.manual_seed(42)
        W_nbits = 2
        gemlite_dtype = TORCH_TO_DTYPE[dtype]

        linear = GemLiteLinearTriton(
            W_nbits=W_nbits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=gemlite_dtype,
            output_dtype=gemlite_dtype,
            scaled_activations=False,
        )

        W_q = torch.randint(0, 4, (out_features, in_features), device=device, dtype=torch.uint8)
        num_groups = (out_features * in_features) // group_size
        scales = (torch.rand((num_groups, 1), device=device, dtype=dtype) * 0.1 + 0.01)
        zeros = torch.randint(0, 4, (num_groups, 1), device=device, dtype=dtype)

        linear.pack(W_q, scales=scales, zeros=zeros, fma_mode=fma_mode)

        # Compute explicit dequantized reference matrix
        W_flat = W_q.reshape(-1, group_size).to(dtype)
        if fma_mode:
            # GemLite fma_mode: stores zeros_stored = -zeros * scales, evaluates Wq * scales + zeros_stored
            W_dequant = (W_flat * scales - (zeros * scales)).reshape(out_features, in_features)
        else:
            W_dequant = ((W_flat - zeros) * scales).reshape(out_features, in_features)

        x = torch.randn((1, in_features), device=device, dtype=dtype)
        y_ref = torch.matmul(x, W_dequant.T)

        # Run direct kernel manual forward
        y_direct = linear.forward_manual(x, matmul_type="GEMV_DIRECT")
        # Run revsplitK baseline
        y_revsplit = linear.forward_manual(x, matmul_type="GEMV_REVSPLITK")

        # Numerical checks
        diff_ref = (y_direct - y_ref).abs()
        max_err = diff_ref.max().item()
        mean_err = diff_ref.mean().item()

        # Cosine similarity in float32 to avoid bf16 mantissa truncation (1/256 step)
        cos_sim = torch.nn.functional.cosine_similarity(y_direct.float().flatten(), y_ref.float().flatten(), dim=0).item()
        cos_sim_rev = torch.nn.functional.cosine_similarity(y_direct.float().flatten(), y_revsplit.float().flatten(), dim=0).item()

        self.assertGreater(cos_sim, 0.9999, f"Low cosine similarity to reference: {cos_sim:.6f} on shape {in_features}x{out_features}")
        self.assertGreater(cos_sim_rev, 0.9999, f"Low cosine similarity to revsplitK: {cos_sim_rev:.6f} on shape {in_features}x{out_features}")
        self.assertLess(mean_err, 5e-2, f"High mean error {mean_err:.5f} on shape {in_features}x{out_features}")

    def test_shapes_bf16(self):
        test_shapes = [
            (1536, 1536),
            (1536, 8960),
            (4096, 1024),
            (4096, 4096),
            (4096, 11008),
            (4096, 14336),
            (5120, 13824),
            (8192, 8192),
        ]
        for in_f, out_f in test_shapes:
            with self.subTest(shape=f"{in_f}x{out_f}", dtype="bf16"):
                self._test_shape_correctness(in_f, out_f, group_size=64, dtype=torch.bfloat16, fma_mode=True)

    def test_shapes_fp16(self):
        test_shapes = [
            (1536, 1536),
            (1536, 8960),
            (4096, 4096),
            (4096, 11008),
        ]
        for in_f, out_f in test_shapes:
            with self.subTest(shape=f"{in_f}x{out_f}", dtype="fp16"):
                self._test_shape_correctness(in_f, out_f, group_size=64, dtype=torch.float16, fma_mode=True)

    def test_group_size_128(self):
        for in_f, out_f in [(2048, 4096), (4096, 11008)]:
            with self.subTest(shape=f"{in_f}x{out_f}", group_size=128):
                self._test_shape_correctness(in_f, out_f, group_size=128, dtype=torch.bfloat16, fma_mode=True)

    def test_symmetric_mode(self):
        in_features, out_features = 4096, 4096
        dtype = torch.bfloat16
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        linear = GemLiteLinearTriton(
            W_nbits=2,
            group_size=64,
            in_features=in_features,
            out_features=out_features,
            input_dtype=gemlite_dtype,
            output_dtype=gemlite_dtype,
            scaled_activations=False,
        )
        W_q = torch.randint(0, 4, (out_features, in_features), device=device, dtype=torch.uint8)
        num_groups = (out_features * in_features) // 64
        scales = (torch.rand((num_groups, 1), device=device, dtype=dtype) * 0.1 + 0.01)
        linear.pack(W_q, scales=scales, zeros=None)

        x = torch.randn((1, in_features), device=device, dtype=dtype)
        y_direct = linear.forward_manual(x, matmul_type="GEMV_DIRECT")
        y_revsplit = linear.forward_manual(x, matmul_type="GEMV_REVSPLITK")

        cos_sim = torch.nn.functional.cosine_similarity(y_direct.float().flatten(), y_revsplit.float().flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.9999, f"Symmetric mode mismatch: {cos_sim}")

    def test_unaligned_out_features(self):
        """Test non-multiple of BLOCK_N (e.g. 2032 or 4032) to verify boundary masking."""
        for in_f, out_f in [(2048, 2032), (4096, 4032)]:
            with self.subTest(shape=f"{in_f}x{out_f}"):
                self._test_shape_correctness(in_f, out_f, group_size=64, dtype=torch.bfloat16, fma_mode=True)

    def test_auto_dispatch_policy(self):
        """Verify conditional auto-dispatch policy:
        - M=1, W_nbits=2, out_features >= 4096 -> GEMV_DIRECT
        - M=1, W_nbits=2, out_features < 4096 -> GEMV_REVSPLITK
        - M > 1 -> GEMM_SPLITK or GEMM
        """
        # M=1, large shapes -> GEMV_DIRECT
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=4096), "GEMV_DIRECT")
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=8960), "GEMV_DIRECT")
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=14336), "GEMV_DIRECT")
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=28672), "GEMV_DIRECT")

        # M=1, small shapes -> GEMV_REVSPLITK (protect losing shapes)
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=1024), "GEMV_REVSPLITK")
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=1536), "GEMV_REVSPLITK")
        self.assertEqual(get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=2048), "GEMV_REVSPLITK")

        # Non 2-bit -> GEMV_REVSPLITK
        self.assertEqual(get_matmul_type(1, W_nbits=4, mx_dtype=False, out_features=8192), "GEMV_REVSPLITK")
        self.assertEqual(get_matmul_type(1, W_nbits=8, mx_dtype=False, out_features=8192), "GEMV_SPLITK")

        # Batch size > 1 -> GEMM_SPLITK or GEMM
        self.assertEqual(get_matmul_type(2, W_nbits=2, mx_dtype=False, out_features=8192), "GEMV_REVSPLITK")
        self.assertEqual(get_matmul_type(8, W_nbits=2, mx_dtype=False, out_features=8192), "GEMM_SPLITK")
        self.assertEqual(get_matmul_type(64, W_nbits=2, mx_dtype=False, out_features=8192), "GEMM")

if __name__ == "__main__":
    unittest.main()
