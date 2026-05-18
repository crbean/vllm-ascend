#!/usr/bin/env python3
"""Benchmark with ACL Graph mode (no enforce_eager)."""
import gc, json, os, re, subprocess, sys, time, traceback
import torch, torch_npu

MODEL = '/home/cb/glm5.1/vllm-custom/Qwen3.5-27B'
RESULTS_FILE = '/home/cb/glm5.1/GTSclaw_skills/adapt-vllm-ascend/teams/adapt-vllm-ascend-team/workdir_qwen35_27b/bench_results_aclgraph.json'

def get_hbm_from_smi():
    try:
        r = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=10)
        infos = []
        for line in r.stdout.split('\n'):
            if '0000:' not in line: continue
            pairs = re.findall(r'(\d+)\s*/\s*(\d+)', line)
            if len(pairs) >= 2:
                hbm_used, hbm_total = pairs[-1]
                infos.append({"device": len(infos), "hbm_used_MB": int(hbm_used)})
        return infos
    except: return []

def run_scenario(tp_size, input_len, output_len, num_prompts):
    from vllm import LLM, SamplingParams
    print(f"\n{'='*60}")
    print(f"ACL Graph: TP={tp_size}, in={input_len}, out={output_len}, batch={num_prompts}")
    print(f"{'='*60}")

    base = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. The attendees of the workshop became the leaders of AI research for decades. Many of them predicted that machines as intelligent as humans would exist within a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1974, in response to criticism from James Lighthill and ongoing pressure from Congress, the US and British governments cut off exploratory research in AI. The next few years would later be called an AI winter. "
    prompts = [base * (input_len * 4 // len(base) + 1) for _ in range(num_prompts)]

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=tp_size, trust_remote_code=True,
              max_model_len=4096, gpu_memory_utilization=0.92)
    init_time = time.time() - t0
    print(f"  LLM init: {init_time:.1f}s")

    # Warmup
    llm.generate(['Hi'], SamplingParams(max_tokens=4, temperature=0))

    sp = SamplingParams(max_tokens=output_len, temperature=0)
    torch.npu.synchronize()
    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    torch.npu.synchronize()
    t2 = time.time()
    elapsed = t2 - t1

    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    total = total_in + total_out
    hbm = max(m['hbm_used_MB'] for m in get_hbm_from_smi()) if get_hbm_from_smi() else 0

    result = {
        "mode": "aclgraph", "tp_size": tp_size,
        "input_len": input_len, "output_len": output_len,
        "num_prompts": num_prompts,
        "total_time_s": round(elapsed, 3),
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "total_tokens": total,
        "throughput_tokens_per_s": round(total / elapsed, 2),
        "output_throughput_tokens_per_s": round(total_out / elapsed, 2),
        "avg_latency_ms_per_token": round(elapsed / total_out * 1000, 3) if total_out > 0 else 0,
        "peak_hbm_mb": hbm, "init_time_s": round(init_time, 1),
    }

    print(f"  Total: {elapsed:.3f}s, {total_out} out tokens")
    print(f"  Throughput: {result['throughput_tokens_per_s']:.2f} tok/s")
    print(f"  Output: {result['output_throughput_tokens_per_s']:.2f} tok/s")
    print(f"  Decode latency: {result['avg_latency_ms_per_token']:.3f} ms/token")
    print(f"  Peak HBM: {hbm} MB")

    del llm, outputs; gc.collect(); torch.npu.empty_cache()
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    scenarios = {
        "short_1": (4, 128, 128, 1),
        "short_4": (4, 128, 128, 4),
        "short_8": (4, 128, 128, 8),
        "medium_1": (4, 512, 256, 1),
        "medium_4": (4, 512, 256, 4),
        "long_1": (4, 2048, 512, 1),
    }

    results = []
    for name, (tp, ilen, olen, bs) in scenarios.items():
        if args.only and args.only != name: continue
        try:
            r = run_scenario(tp, ilen, olen, bs)
            results.append({"type": "throughput", "scenario": name, "data": r})
        except Exception as e:
            print(f"FAILED {name}: {e}")
            traceback.print_exc()
            results.append({"type": "error", "scenario": name, "error": str(e)})
            gc.collect(); torch.npu.empty_cache()

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("ACL GRAPH BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<15} {'tok/s':>10} {'out/s':>10} {'lat_ms':>10} {'HBM_MB':>10}")
    print("-" * 60)
    for r in results:
        if r["type"] == "throughput":
            d = r["data"]
            print(f"{r['scenario']:<15} {d['throughput_tokens_per_s']:>10.2f} "
                  f"{d['output_throughput_tokens_per_s']:>10.2f} "
                  f"{d['avg_latency_ms_per_token']:>10.3f} {d['peak_hbm_mb']:>10}")
