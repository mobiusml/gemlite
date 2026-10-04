# SPDX-License-Identifier: Apache-2.0
"""
End-to-End LLM Generation Benchmark: GEMV_DIRECT vs GEMV_REVSPLITK in GemLite.
Quantizes Qwen2.5-1.5B to A16W2 (2-bit weights, group_size=64) using GemLiteLinearTriton.
Measures generation throughput (tokens/sec), latency (ms/token), and peak VRAM.
"""

import os
import sys
sys.modules['torchao'] = None

import time
import json
import torch
import triton
import torch.nn as nn

# Backward compatibility shims for test environments running PyTorch < 2.7 / Triton < 3.3
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8
if not hasattr(triton, "set_allocator"):
    triton.set_allocator = lambda fn: None

from transformers import AutoModelForCausalLM, AutoTokenizer
from gemlite import reset_config, set_autotune
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE

device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "CUDA device required for benchmark"

def quantize_module_to_gemlite_w2(orig_linear, group_size=64):
    in_features = orig_linear.in_features
    out_features = orig_linear.out_features
    dtype = torch.bfloat16
    gemlite_dtype = TORCH_TO_DTYPE[dtype]

    gemlite_linear = GemLiteLinearTriton(
        W_nbits=2,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=gemlite_dtype,
        output_dtype=gemlite_dtype,
        scaled_activations=False,
    )

    with torch.no_grad():
        W = orig_linear.weight.data.float()
        flat = W.reshape(-1, group_size)
        mn = flat.min(dim=-1, keepdim=True)[0]
        mx = flat.max(dim=-1, keepdim=True)[0]
        scale = torch.clamp((mx - mn) / 3.0, min=1e-8)
        q = torch.round((flat - mn) / scale).clamp(0, 3).to(torch.uint8).reshape(out_features, in_features)

        # In GemLite fma_mode: W ~ Wq * scale + zeros_stored where zeros_stored = -zeros_in * scale
        zeros_in = (-mn / scale).reshape(-1, 1).to(dtype)
        scale_bf16 = scale.reshape(-1, 1).to(dtype)

        bias = orig_linear.bias.data if orig_linear.bias is not None else None
        gemlite_linear.pack(q, scales=scale_bf16, zeros=zeros_in, bias=bias, fma_mode=True)

    return gemlite_linear

def replace_linear_layers(model, group_size=64):
    """Recursively replaces model's nn.Linear projection layers with GemLiteLinearTriton."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name != "lm_head":
            # Only quantize projection layers (in_features divisible by group_size)
            if child.in_features % group_size == 0 and child.out_features % 32 == 0:
                setattr(model, name, quantize_module_to_gemlite_w2(child, group_size=group_size))
        else:
            replace_linear_layers(child, group_size=group_size)

def set_model_matmul_type(model, matmul_type):
    """Sets dispatch mode on all GemLiteLinearTriton layers."""
    for module in model.modules():
        if isinstance(module, GemLiteLinearTriton):
            if matmul_type == "AUTO":
                module.forward = module.forward_auto_no_warmup
            else:
                module.forward = lambda x, mt=matmul_type, m=module: m.forward_manual(x, matmul_type=mt)

def run_decode_benchmark(model, input_ids, gen_tokens=64, num_warmup=8):
    prompt_len = input_ids.shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=num_warmup, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark pure generation
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)
    torch.cuda.synchronize()
    t_elapsed = time.perf_counter() - t0

    tokens_generated = out.shape[1] - prompt_len
    tok_sec = tokens_generated / t_elapsed
    ms_per_tok = (t_elapsed / tokens_generated) * 1000.0

    return {
        "tokens_generated": tokens_generated,
        "elapsed_sec": t_elapsed,
        "tokens_per_sec": tok_sec,
        "ms_per_token": ms_per_tok,
    }

def main():
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"=== GemLite End-to-End LLM Benchmark ({model_id}) on {torch.cuda.get_device_name(0)} ===")

    reset_config()
    set_autotune(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    prompt = "Artificial Intelligence systems and GPU architecture optimization require understanding of memory bandwidth,"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

    gen_tokens = 64
    e2e_results = {}

    # 1. Baseline BF16 Model
    print("\n1. Benchmarking Baseline PyTorch BF16 Model...")
    torch.cuda.reset_peak_memory_stats()
    model_bf16 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    vram_bf16 = torch.cuda.max_memory_allocated() / (1024**3)

    res_bf16 = run_decode_benchmark(model_bf16, input_ids, gen_tokens=gen_tokens)
    res_bf16["peak_vram_gb"] = vram_bf16
    print(f"   BF16 Throughput: {res_bf16['tokens_per_sec']:.2f} tok/s ({res_bf16['ms_per_token']:.2f} ms/tok, VRAM: {vram_bf16:.2f} GB)")
    e2e_results["PyTorch BF16"] = res_bf16

    # Convert model to GemLite A16W2
    print("\n2. Quantizing Model to GemLite A16W2 (2-bit, G=64)...")
    replace_linear_layers(model_bf16, group_size=64)
    model_gemlite = model_bf16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_w2 = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"   A16W2 Model Peak VRAM: {vram_w2:.2f} GB (VRAM saving: {(1.0 - vram_w2/vram_bf16)*100:.1f}%)")

    # 3. GemLite GEMV_REVSPLITK
    print("\n3. Benchmarking GemLite GEMV_REVSPLITK (Baseline low-bit path)...")
    set_model_matmul_type(model_gemlite, "GEMV_REVSPLITK")
    res_revsplit = run_decode_benchmark(model_gemlite, input_ids, gen_tokens=gen_tokens)
    res_revsplit["peak_vram_gb"] = vram_w2
    print(f"   GEMV_REVSPLITK: {res_revsplit['tokens_per_sec']:.2f} tok/s ({res_revsplit['ms_per_token']:.2f} ms/tok)")
    e2e_results["GemLite A16W2 (REVSPLITK)"] = res_revsplit

    # 4. GemLite GEMV_DIRECT
    print("\n4. Benchmarking GemLite GEMV_DIRECT (New direct-accumulation specialization)...")
    set_model_matmul_type(model_gemlite, "GEMV_DIRECT")
    res_direct = run_decode_benchmark(model_gemlite, input_ids, gen_tokens=gen_tokens)
    res_direct["peak_vram_gb"] = vram_w2
    speedup_vs_rev = res_direct["tokens_per_sec"] / res_revsplit["tokens_per_sec"]
    print(f"   GEMV_DIRECT: {res_direct['tokens_per_sec']:.2f} tok/s ({res_direct['ms_per_token']:.2f} ms/tok, Speedup vs REVSPLITK: {speedup_vs_rev:.2f}x)")
    e2e_results["GemLite A16W2 (GEMV_DIRECT)"] = res_direct

    # 5. GemLite AUTO Dispatch
    print("\n5. Benchmarking GemLite AUTO Dispatch (Adaptive routing policy)...")
    set_model_matmul_type(model_gemlite, "AUTO")
    res_auto = run_decode_benchmark(model_gemlite, input_ids, gen_tokens=gen_tokens)
    res_auto["peak_vram_gb"] = vram_w2
    speedup_auto_vs_rev = res_auto["tokens_per_sec"] / res_revsplit["tokens_per_sec"]
    print(f"   AUTO Dispatch: {res_auto['tokens_per_sec']:.2f} tok/s ({res_auto['ms_per_token']:.2f} ms/tok, Speedup vs REVSPLITK: {speedup_auto_vs_rev:.2f}x)")
    e2e_results["GemLite A16W2 (AUTO)"] = res_auto

    # Optional: save results if output path provided in env
    output_path = os.environ.get("BENCHMARK_OUTPUT_JSON")
    if output_path:
        with open(output_path, "w") as f:
            json.dump(e2e_results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"{'Configuration':30} | {'Throughput (tok/s)':20} | {'Latency (ms/tok)':18} | {'VRAM (GB)':10}")
    print("=" * 90)
    for name, r in e2e_results.items():
        print(f"{name:30} | {r['tokens_per_sec']:20.2f} | {r['ms_per_token']:18.2f} | {r['peak_vram_gb']:10.2f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
