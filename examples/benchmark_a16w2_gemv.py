# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script for GEMV_DIRECT vs GEMV_REVSPLITK vs GEMV in upstream GemLite.
Evaluates M=1 A16W2 decoding across realistic LLM transformer shapes under:
1. Warm L2 cache condition (repeated GEMV on single layer)
2. Cold/Evicted L2 cache condition (multi-layer rotation busting L2 cache)
Measures median, p10, p90, effective payload bandwidth, and evaluates cross-over dispatch boundary.
"""

import os
import sys
import json
import time
import torch
import triton
import numpy as np

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
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "CUDA device required for benchmark"

SHAPES = [
    (1536, 256),     # Qwen small projection
    (1536, 1536),    # Qwen hidden square
    (1536, 8960),    # Qwen MLP up
    (4096, 1024),    # LLaMA 8B Q-reduction
    (4096, 4096),    # LLaMA 8B hidden square
    (4096, 11008),   # LLaMA 8B gate/up
    (4096, 14336),   # Mistral / Qwen-7B MLP
    (5120, 13824),   # LLaMA-13B / Qwen-14B MLP
    (8192, 8192),    # LLaMA-70B hidden square
    (8192, 28672),   # LLaMA-70B MLP up
]

def benchmark_kernel(layers, x, matmul_type, num_warmup=10, num_iters=50):
    num_layers = len(layers)
    stream = torch.cuda.current_stream()
    
    # Warmup
    for _ in range(num_warmup):
        for l in layers:
            l.forward_manual(x, matmul_type=matmul_type)
    torch.cuda.synchronize()

    # Timing with CUDA events per invocation
    times_us = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(num_iters):
        layer = layers[i % num_layers]
        start_event.record(stream)
        layer.forward_manual(x, matmul_type=matmul_type)
        end_event.record(stream)
        end_event.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1000.0) # ms to us

    times_us = np.array(times_us)
    return {
        "median_us": float(np.median(times_us)),
        "p10_us": float(np.percentile(times_us, 10)),
        "p90_us": float(np.percentile(times_us, 90)),
        "min_us": float(np.min(times_us)),
        "std_us": float(np.std(times_us)),
    }

def main():
    print(f"=== GemLite Upstream GEMV Benchmark on {torch.cuda.get_device_name(0)} ===")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__}, Triton: {torch.version.cuda}")

    reset_config()
    set_autotune(False) # Benchmark clean un-cached default configurations

    results = []
    group_size = 64
    dtype = torch.bfloat16
    gemlite_dtype = TORCH_TO_DTYPE[dtype]

    for K, N in SHAPES:
        print(f"\nEvaluating shape K={K}, N={N} (A16W2, G={group_size})...")
        # Compute exact physical buffer size
        # Packed weights: (K // 16) * N * 4 bytes (int32) = K * N * 0.25 bytes
        # Scales: (K // G) * N * 2 bytes (bf16)
        # Zeros: (K // G) * N * 2 bytes (bf16)
        # Activation: K * 2 bytes (bf16)
        # Output: N * 2 bytes (bf16)
        weights_bytes = (K * N * 2) // 8
        metadata_bytes = 2 * (K // group_size) * N * 2
        total_param_bytes = weights_bytes + metadata_bytes
        total_payload_bytes = total_param_bytes + K * 2

        # 1. Warm cache condition (single layer)
        torch.manual_seed(0)
        single_layer = GemLiteLinearTriton(2, group_size, K, N, gemlite_dtype, gemlite_dtype, scaled_activations=False)
        W_q = torch.randint(0, 4, (N, K), device=device, dtype=torch.uint8)
        num_g = (N * K) // group_size
        s = torch.rand((num_g, 1), device=device, dtype=dtype) * 0.1 + 0.01
        z = torch.randint(0, 4, (num_g, 1), device=device, dtype=dtype)
        single_layer.pack(W_q, scales=s, zeros=z, fma_mode=True)
        x = torch.randn((1, K), device=device, dtype=dtype)

        res_warm_direct = benchmark_kernel([single_layer], x, "GEMV_DIRECT")
        res_warm_revsplit = benchmark_kernel([single_layer], x, "GEMV_REVSPLITK")

        # 2. Cold/Evicted L2 cache condition (multi-layer rotation)
        num_layers_rotation = 8 if (total_param_bytes * 8 > 12 * 1024 * 1024) else 16
        layers_rotation = []
        for l_idx in range(num_layers_rotation):
            torch.manual_seed(l_idx + 1)
            layer = GemLiteLinearTriton(2, group_size, K, N, gemlite_dtype, gemlite_dtype, scaled_activations=False)
            W_qi = torch.randint(0, 4, (N, K), device=device, dtype=torch.uint8)
            si = torch.rand((num_g, 1), device=device, dtype=dtype) * 0.1 + 0.01
            zi = torch.randint(0, 4, (num_g, 1), device=device, dtype=dtype)
            layer.pack(W_qi, scales=si, zeros=zi, fma_mode=True)
            layers_rotation.append(layer)

        res_cold_direct = benchmark_kernel(layers_rotation, x, "GEMV_DIRECT")
        res_cold_revsplit = benchmark_kernel(layers_rotation, x, "GEMV_REVSPLITK")

        # Effective Payload Bandwidth (physical buffer bytes / time)
        # bandwidth (GB/s) = bytes / (us * 1e-6) / 1e9 = bytes / us / 1000
        bw_direct_cold = (total_payload_bytes / res_cold_direct["median_us"]) / 1000.0
        bw_revsplit_cold = (total_payload_bytes / res_cold_revsplit["median_us"]) / 1000.0

        speedup_warm = res_warm_revsplit["median_us"] / res_warm_direct["median_us"]
        speedup_cold = res_cold_revsplit["median_us"] / res_cold_direct["median_us"]

        auto_dispatched = get_matmul_type(1, W_nbits=2, mx_dtype=False, out_features=N)

        shape_result = {
            "K": K,
            "N": N,
            "total_param_mb": total_param_bytes / (1024**2),
            "warm_direct_us": res_warm_direct["median_us"],
            "warm_revsplit_us": res_warm_revsplit["median_us"],
            "warm_speedup": speedup_warm,
            "cold_direct_us": res_cold_direct["median_us"],
            "cold_direct_p10": res_cold_direct["p10_us"],
            "cold_direct_p90": res_cold_direct["p90_us"],
            "cold_revsplit_us": res_cold_revsplit["median_us"],
            "cold_revsplit_p10": res_cold_revsplit["p10_us"],
            "cold_revsplit_p90": res_cold_revsplit["p90_us"],
            "cold_speedup": speedup_cold,
            "payload_bw_direct_gbs": bw_direct_cold,
            "payload_bw_revsplit_gbs": bw_revsplit_cold,
            "auto_dispatch": auto_dispatched,
            "winner": "GEMV_DIRECT" if speedup_cold > 1.05 else ("GEMV_REVSPLITK" if speedup_cold < 0.95 else "TIE"),
        }
        results.append(shape_result)

        print(f"  [Cold/Evicted] DIRECT: {res_cold_direct['median_us']:.1f} us ({bw_direct_cold:.1f} GB/s payload) | REVSPLITK: {res_cold_revsplit['median_us']:.1f} us ({bw_revsplit_cold:.1f} GB/s payload)")
        print(f"  [Speedup] Cold: {speedup_cold:.2f}x | Auto-Dispatch: {auto_dispatched} | Winner: {shape_result['winner']}")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Shape (KxN)':18} | {'Param MB':10} | {'DIRECT (us)':12} | {'REVSPLITK (us)':14} | {'Speedup':9} | {'Payload GB/s':14} | {'Auto-Dispatch':15} | {'Winner':12}")
    print("=" * 120)
    for r in results:
        shape_str = f"{r['K']}x{r['N']}"
        print(f"{shape_str:18} | {r['total_param_mb']:10.2f} | {r['cold_direct_us']:12.1f} | {r['cold_revsplit_us']:14.1f} | {r['cold_speedup']:9.2f}x | {r['payload_bw_direct_gbs']:14.1f} | {r['auto_dispatch']:15} | {r['winner']:12}")
    print("=" * 120)

if __name__ == "__main__":
    main()
