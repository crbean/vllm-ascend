#!/usr/bin/env python3
"""
Performance benchmark script for Qwen3.5-27B on Ascend NPU.
Collects throughput, latency, and memory metrics.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

import subprocess

import torch
import torch_npu


def get_npu_hbm_from_smi():
    """Get per-NPU HBM usage via npu-smi (works across processes)."""
    import re
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True,
                                text=True, timeout=10)
        infos = []
        lines = result.stdout.split('\n')
        for line in lines:
            if '0000:' not in line:
                continue
            pairs = re.findall(r'(\d+)\s*/\s*(\d+)', line)
            if len(pairs) >= 2:
                hbm_used, hbm_total = pairs[-1]
                infos.append({
                    "device": len(infos),
                    "hbm_used_MB": int(hbm_used),
                    "hbm_total_MB": int(hbm_total),
                })
        return infos
    except Exception as e:
        print(f"  Warning: npu-smi query failed: {e}")
        return []


def get_npu_memory_info():
    """Get per-NPU memory info: total, allocated, reserved."""
    infos = []
    for i in range(torch.npu.device_count()):
        total = torch.npu.get_device_properties(i).total_memory / 1024 / 1024
        allocated = torch.npu.memory_allocated(i) / 1024 / 1024
        reserved = torch.npu.memory_reserved(i) / 1024 / 1024
        infos.append({
            "device": i,
            "total_MB": round(total, 1),
            "allocated_MB": round(allocated, 1),
            "reserved_MB": round(reserved, 1),
            "free_MB": round(total - reserved, 1),
        })
    return infos


def run_benchmark(tp_size, input_len, output_len, num_prompts, model_path,
                  max_model_len=4096):
    """Run a single benchmark scenario and return metrics."""
    from vllm import LLM, SamplingParams

    # Warm up with a single prompt first
    print(f"\n{'='*60}")
    print(f"TP={tp_size}, input_len={input_len}, output_len={output_len}, "
          f"num_prompts={num_prompts}")
    print(f"{'='*60}")

    # Create deterministic prompts of exact input_len
    # Use a base text and pad/truncate to exact length
    base_text = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. The attendees of the workshop became the leaders of AI research for decades. Many of them predicted that machines as intelligent as humans would exist within a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1974, in response to criticism from James Lighthill and ongoing pressure from Congress, the US and British governments cut off exploratory research in AI. The next few years would later be called an AI winter. In the 1980s, a form of AI called expert systems was adopted by corporations around the world, and knowledge became the focus of AI research. "

    prompts = []
    for i in range(num_prompts):
        # Repeat base text to reach target length (approximately)
        tokenizer = None
        # We'll use approximate char-to-token ratio
        # For this model, roughly 4 chars per token
        target_chars = input_len * 4
        text = base_text * (target_chars // len(base_text) + 1)
        prompts.append(text)

    print(f"  Initializing LLM (TP={tp_size}, max_model_len={max_model_len})...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    init_time = time.time() - t0
    print(f"  LLM initialized in {init_time:.1f}s")

    # Get memory info after model load
    mem_after_load = get_npu_memory_info()
    for m in mem_after_load:
        print(f"  NPU {m['device']}: allocated={m['allocated_MB']:.0f}MB, "
              f"reserved={m['reserved_MB']:.0f}MB, free={m['free_MB']:.0f}MB")

    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0,
    )

    # Run benchmark
    print(f"  Running benchmark ({num_prompts} prompts)...")
    torch.npu.synchronize()
    t_start = time.time()

    outputs = llm.generate(prompts, sampling_params)

    torch.npu.synchronize()
    t_end = time.time()
    total_time = t_end - t_start

    # Collect metrics
    total_input_tokens = 0
    total_output_tokens = 0
    ttft_list = []

    for output in outputs:
        # vLLM output has metrics
        total_input_tokens += len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_output_tokens += completion_tokens

    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / total_time
    output_throughput = total_output_tokens / total_time
    avg_latency_per_token = (total_time / total_output_tokens * 1000
                             if total_output_tokens > 0 else 0)

    # Get peak memory after inference via npu-smi (works across processes)
    smi_mem = get_npu_hbm_from_smi()
    peak_hbm = max(m['hbm_used_MB'] for m in smi_mem) if smi_mem else 0
    mem_after_infer = get_npu_memory_info()
    for m in smi_mem:
        print(f"  NPU {m['device']} (npu-smi): HBM {m['hbm_used_MB']} / {m['hbm_total_MB']} MB")

    result = {
        "tp_size": tp_size,
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": num_prompts,
        "total_time_s": round(total_time, 3),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": round(throughput, 2),
        "output_throughput_tokens_per_s": round(output_throughput, 2),
        "avg_latency_ms_per_token": round(avg_latency_per_token, 3),
        "peak_hbm_mb": round(peak_hbm, 1),
        "init_time_s": round(init_time, 1),
    }

    print(f"\n  Results:")
    print(f"    Total time: {total_time:.3f}s")
    print(f"    Total input tokens: {total_input_tokens}")
    print(f"    Total output tokens: {total_output_tokens}")
    print(f"    Throughput: {throughput:.2f} tokens/s")
    print(f"    Output throughput: {output_throughput:.2f} tokens/s")
    print(f"    Avg decode latency: {avg_latency_per_token:.3f} ms/token")
    print(f"    Peak HBM: {peak_hbm:.1f} MB")

    # Clean up
    del llm
    del outputs
    gc.collect()
    torch.npu.empty_cache()

    return result


def run_memory_analysis(model_path, tp_size=4, max_model_len=4096):
    """Analyze memory breakdown: weights, KV cache, activations."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Memory Analysis (TP={tp_size}, max_model_len={max_model_len})")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enforce_eager=True,
    )

    mem_info = get_npu_memory_info()
    smi_mem = get_npu_hbm_from_smi()

    # Get model parameter count and size
    total_params = 0
    total_param_bytes = 0
    for name, param in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters():
        total_params += param.numel()
        total_param_bytes += param.numel() * param.element_size()

    # Convert to MB
    param_mb = total_param_bytes / 1024 / 1024
    # Per NPU (TP=4, each holds 1/4 of params roughly)
    per_npu_param_mb = param_mb / tp_size

    print(f"\n  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Total param memory: {param_mb:.1f} MB ({param_mb/1024:.2f} GB)")
    print(f"  Per-NPU param memory: {per_npu_param_mb:.1f} MB ({per_npu_param_mb/1024:.2f} GB)")

    for m in mem_info:
        print(f"  NPU {m['device']}: total={m['total_MB']:.0f}MB, "
              f"allocated={m['allocated_MB']:.0f}MB, "
              f"reserved={m['reserved_MB']:.0f}MB, "
              f"free={m['free_MB']:.0f}MB")
    for m in smi_mem:
        print(f"  NPU {m['device']} (npu-smi): HBM {m['hbm_used_MB']} / {m['hbm_total_MB']} MB")

    result = {
        "total_params_B": round(total_params / 1e9, 2),
        "total_param_MB": round(param_mb, 1),
        "per_npu_param_MB": round(per_npu_param_mb, 1),
        "npu_memory": mem_info,
        "npu_smi_memory": smi_mem,
        "tp_size": tp_size,
        "max_model_len": max_model_len,
    }

    del llm
    gc.collect()
    torch.npu.empty_cache()

    return result


def wait_for_npu_idle(timeout=120):
    """Wait for NPU devices to become idle."""
    print("Waiting for NPU devices to become idle...")
    start = time.time()
    while time.time() - start < timeout:
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
        if 'No running processes' in result.stdout:
            print("All NPU devices idle.")
            return True
        time.sleep(5)
    print("WARNING: Timeout waiting for NPU idle, proceeding anyway.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B NPU Performance Benchmark")
    parser.add_argument("--model-path", type=str,
                        default="/home/cb/glm5.1/vllm-custom/Qwen3.5-27B",
                        help="Path to model weights")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model length for vLLM")
    parser.add_argument("--output-file", type=str,
                        default="/home/cb/glm5.1/GTSclaw_skills/adapt-vllm-ascend/teams/adapt-vllm-ascend-team/workdir_qwen35_27b/bench_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--skip-memory", action="store_true",
                        help="Skip memory analysis")
    parser.add_argument("--skip-tp8", action="store_true",
                        help="Skip TP=8 test")
    parser.add_argument("--only-scenario", type=str, default=None,
                        help="Run only specific scenario, e.g. 'tp4_short_1'")
    args = parser.parse_args()

    all_results = []

    # Define test scenarios
    scenarios = {
        # TP=4 scenarios
        "tp4_short_1": {"tp_size": 4, "input_len": 128, "output_len": 128, "num_prompts": 1},
        "tp4_short_4": {"tp_size": 4, "input_len": 128, "output_len": 128, "num_prompts": 4},
        "tp4_short_8": {"tp_size": 4, "input_len": 128, "output_len": 128, "num_prompts": 8},
        "tp4_medium_1": {"tp_size": 4, "input_len": 512, "output_len": 256, "num_prompts": 1},
        "tp4_medium_4": {"tp_size": 4, "input_len": 512, "output_len": 256, "num_prompts": 4},
        "tp4_long_1": {"tp_size": 4, "input_len": 2048, "output_len": 512, "num_prompts": 1},
        # TP=8 scenarios
        "tp8_short_1": {"tp_size": 8, "input_len": 128, "output_len": 128, "num_prompts": 1},
        "tp8_short_4": {"tp_size": 8, "input_len": 128, "output_len": 128, "num_prompts": 4},
        "tp8_short_8": {"tp_size": 8, "input_len": 128, "output_len": 128, "num_prompts": 8},
        "tp8_medium_1": {"tp_size": 8, "input_len": 512, "output_len": 256, "num_prompts": 1},
        "tp8_medium_4": {"tp_size": 8, "input_len": 512, "output_len": 256, "num_prompts": 4},
        "tp8_long_1": {"tp_size": 8, "input_len": 2048, "output_len": 512, "num_prompts": 1},
    }

    # Memory analysis
    if not args.skip_memory:
        try:
            wait_for_npu_idle()
            mem_result = run_memory_analysis(
                args.model_path, tp_size=4, max_model_len=args.max_model_len)
            all_results.append({"type": "memory_analysis", "data": mem_result})
        except Exception as e:
            print(f"Memory analysis failed: {e}")
            traceback.print_exc()

    # Run throughput scenarios
    for name, params in scenarios.items():
        if args.only_scenario and args.only_scenario != name:
            continue
        if name.startswith("tp8") and args.skip_tp8:
            continue

        try:
            wait_for_npu_idle()
            result = run_benchmark(
                model_path=args.model_path,
                max_model_len=args.max_model_len,
                **params,
            )
            all_results.append({"type": "throughput", "scenario": name, "data": result})
        except Exception as e:
            print(f"Scenario {name} failed: {e}")
            traceback.print_exc()
            all_results.append({
                "type": "error",
                "scenario": name,
                "error": str(e),
            })
            # Try to clean up
            gc.collect()
            torch.npu.empty_cache()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<20} {'tokens/s':>12} {'out_tok/s':>12} "
          f"{'latency_ms':>12} {'peak_HBM_MB':>12} {'time_s':>8}")
    print("-" * 80)
    for r in all_results:
        if r["type"] == "throughput":
            d = r["data"]
            print(f"{r['scenario']:<20} {d['throughput_tokens_per_s']:>12.2f} "
                  f"{d['output_throughput_tokens_per_s']:>12.2f} "
                  f"{d['avg_latency_ms_per_token']:>12.3f} "
                  f"{d['peak_hbm_mb']:>12.1f} "
                  f"{d['total_time_s']:>8.3f}")
        elif r["type"] == "error":
            print(f"{r['scenario']:<20} ERROR: {r['error'][:50]}")


if __name__ == "__main__":
    main()
