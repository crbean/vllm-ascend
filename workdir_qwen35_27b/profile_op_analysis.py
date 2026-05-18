#!/usr/bin/env python3
"""
Analyze profiling data from NPU inference to identify performance bottlenecks.
Works with torch_npu.profiler output.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback


def run_profiling(model_path, tp_size=4, max_model_len=4096, output_dir="./profile_data"):
    """Run inference with torch_npu.profiler and collect trace data."""
    import torch
    import torch_npu
    from torch_npu import profiler as npu_profiler
    from vllm import LLM, SamplingParams

    os.makedirs(output_dir, exist_ok=True)

    # Create LLM
    print("Initializing LLM for profiling...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enforce_eager=True,
    )

    # Warmup
    print("Warmup inference...")
    prompts = ["What is the capital of France?"]
    sampling_params = SamplingParams(max_tokens=16, temperature=0)
    llm.generate(prompts, sampling_params)

    # Profiled inference
    print("Starting profiling...")
    with npu_profiler.profile(
        activities=[npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU],
        schedule=npu_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=npu_profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        # Decode profiling (single prompt, 64 tokens)
        prompts = ["Write a short poem about the sea."]
        sampling_params = SamplingParams(max_tokens=64, temperature=0)
        llm.generate(prompts, sampling_params)
        prof.step()

    print(f"Profiling data saved to {output_dir}")

    # Also collect summary statistics
    print("\nTop 20 operators by NPU total time:")
    print(prof.key_averages().table(sort_by="self_npu_time_total", row_limit=20))

    # Save operator stats
    stats = []
    for evt in prof.key_averages():
        stats.append({
            "name": evt.key,
            "cpu_time_total_us": evt.cpu_time_total,
            "npu_time_total_us": evt.npu_time_total,
            "self_cpu_time_total_us": evt.self_cpu_time_total,
            "self_npu_time_total_us": evt.self_npu_time_total,
            "count": evt.count,
            "cpu_memory_usage": evt.cpu_memory_usage,
            "npu_memory_usage": evt.npu_memory_usage,
        })

    stats_file = os.path.join(output_dir, "op_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Operator stats saved to {stats_file}")

    # Cleanup
    del llm
    gc.collect()
    torch.npu.empty_cache()

    return stats_file


def analyze_hotspots(stats_file):
    """Analyze operator stats to identify hotspots."""
    with open(stats_file) as f:
        stats = json.load(f)

    # Sort by self_npu_time_total descending
    npu_stats = [s for s in stats if s["self_npu_time_total_us"] > 0]
    npu_stats.sort(key=lambda x: x["self_npu_time_total_us"], reverse=True)

    total_npu_time = sum(s["self_npu_time_total_us"] for s in npu_stats)

    print(f"\n{'='*80}")
    print(f"PERFORMANCE HOTSPOT ANALYSIS")
    print(f"{'='*80}")
    print(f"Total NPU time: {total_npu_time/1e6:.3f}s")
    print(f"\nTop 20 NPU time consumers:")
    print(f"{'Operator':<50} {'Self NPU (ms)':>14} {'Count':>8} {'% Total':>8} {'Avg (us)':>10}")
    print("-" * 95)

    cumulative = 0
    for s in npu_stats[:20]:
        pct = s["self_npu_time_total_us"] / total_npu_time * 100 if total_npu_time > 0 else 0
        cumulative += pct
        avg_us = s["self_npu_time_total_us"] / s["count"] if s["count"] > 0 else 0
        name = s["name"][:48]
        print(f"{name:<50} {s['self_npu_time_total_us']/1e3:>14.3f} {s['count']:>8} "
              f"{pct:>7.2f}% {avg_us:>10.1f}")

    print(f"\nCumulative top 20: {cumulative:.1f}%")

    # Categorize operators
    categories = {
        "MatMul/GEMM": ["matmul", "mm", "bmm", "linear", "addmm", "baddbmm"],
        "Attention": ["attention", "sdpa", "flash", "fused_infer_attention",
                      "scaled_dot"],
        "Normalization": ["norm", "rms_norm", "layer_norm", "gemma_rms"],
        "Activation": ["silu", "gelu", "relu", "sigmoid", "and_mul"],
        "Conv": ["conv", "conv1d", "conv3d"],
        "RoPE": ["rotary", "rope", "mrope"],
        "GDN": ["gated_delta", "recurrent", "causal_conv"],
        "Communication": ["allreduce", "all_gather", "reduce_scatter", "hccl"],
        "DataMovement": ["copy", "transpose", "reshape", "view", "permute",
                         "contiguous", "expand", "scatter", "gather"],
        "Other": [],
    }

    cat_times = {k: 0 for k in categories}
    for s in npu_stats:
        name_lower = s["name"].lower()
        categorized = False
        for cat, keywords in categories.items():
            if cat == "Other":
                continue
            if any(kw in name_lower for kw in keywords):
                cat_times[cat] += s["self_npu_time_total_us"]
                categorized = True
                break
        if not categorized:
            cat_times["Other"] += s["self_npu_time_total_us"]

    print(f"\nOperator categories (by NPU time):")
    print(f"{'Category':<25} {'Time (ms)':>12} {'% Total':>8}")
    print("-" * 48)
    for cat in sorted(cat_times, key=cat_times.get, reverse=True):
        pct = cat_times[cat] / total_npu_time * 100 if total_npu_time > 0 else 0
        print(f"{cat:<25} {cat_times[cat]/1e3:>12.3f} {pct:>7.2f}%")

    return npu_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/cb/glm5.1/vllm-custom/Qwen3.5-27B")
    parser.add_argument("--output-dir", default="./profile_data")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing profiling data")
    args = parser.parse_args()

    if args.analyze_only:
        stats_file = os.path.join(args.output_dir, "op_stats.json")
        if os.path.exists(stats_file):
            analyze_hotspots(stats_file)
        else:
            print(f"No stats file found at {stats_file}")
    else:
        stats_file = run_profiling(args.model_path, output_dir=args.output_dir)
        analyze_hotspots(stats_file)
