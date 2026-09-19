"""
Benchmark TP inference with standard AllReduce — the vLLM dispatch path.

Unlike bench_inference_tp_layer.py (which uses fused symmetric memory ops),
this benchmark uses dist.all_reduce on dist.group.WORLD — the same collective
path that vLLM and most production TP inference systems use.

Two sections:

  1. TP AllReduce by inference phase
     AllReduce at the exact tensor sizes produced by TP transformer layers
     during decode (batch × hidden, latency-dominated) and prefill
     (seq_len × hidden, bandwidth-dominated).

  2. TP layer (vLLM-style)
     A simplified transformer layer: 4 GEMMs + 2 AllReduces per layer.
     QKV (colwise) → O proj + AllReduce → Gate+Up (colwise) → SiLU →
     Down + AllReduce. Benchmarked at decode and prefill tensor sizes.

Usage:
  torchrun --nproc_per_node=8 bench_inference_tp_vllm.py
  torchrun --nproc_per_node=8 bench_inference_tp_vllm.py --section allreduce
  torchrun --nproc_per_node=8 bench_inference_tp_vllm.py --models Llama-70B --json results/tp_vllm.json
"""

import argparse
import math

import torch
import torch.distributed as dist

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, get_gpu_peak_bandwidth,
    reset_nccl_tuning, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# AllReduce and GEMM cost per dtype at vLLM tensor shapes.
DTYPES = ("bf16", "fp16", "fp32")


MODELS = {
    "Llama-8B":   {"hidden": 4096,  "intermediate": 14336},
    "Llama-70B":  {"hidden": 8192,  "intermediate": 28672},
    "Llama-405B": {"hidden": 16384, "intermediate": 53248},
}

DECODE_BATCHES = [1, 8, 32, 128]
PREFILL_LENGTHS = [128, 512, 2048, 8192]


def algo_bw(nbytes, p50_us):
    if p50_us <= 0:
        return 0.0
    return nbytes / (p50_us * 1e-6) / 1e9


def bus_bw(algo_gbps, world_size):
    n = world_size
    return algo_gbps * 2 * (n - 1) / n


def format_bytes(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ---- Section 1: AllReduce by inference phase ----


def bench_allreduce_phase(model_name, cfg, tp, device, dtype,
                          decode_batches, prefill_lengths, warmup, iters):
    hidden = cfg["hidden"]
    results = []

    for batch in decode_batches:
        tensor = torch.randn(batch, hidden, dtype=dtype, device=device)

        def fn():
            dist.all_reduce(tensor)

        reset_nccl_tuning(fn, warmup=warmup)
        s = bench(fn, warmup=warmup, iters=iters)
        nelems = batch * hidden
        nbytes = nelems * dtype.itemsize

        results.append({
            "section": "allreduce_decode",
            "model": model_name,
            "batch_size": batch,
            "hidden": hidden,
            "nelems": nelems,
            "nbytes": nbytes,
            "stats": s,
            "algo_bw_gbps": round(algo_bw(nbytes, s["p50_us"]), 2),
            "bus_bw_gbps": round(bus_bw(algo_bw(nbytes, s["p50_us"]), tp), 2),
        })

    for seq_len in prefill_lengths:
        tensor = torch.randn(seq_len, hidden, dtype=dtype, device=device)

        def fn():
            dist.all_reduce(tensor)

        reset_nccl_tuning(fn, warmup=warmup)
        s = bench(fn, warmup=warmup, iters=iters)
        nelems = seq_len * hidden
        nbytes = nelems * dtype.itemsize

        results.append({
            "section": "allreduce_prefill",
            "model": model_name,
            "seq_len": seq_len,
            "hidden": hidden,
            "nelems": nelems,
            "nbytes": nbytes,
            "stats": s,
            "algo_bw_gbps": round(algo_bw(nbytes, s["p50_us"]), 2),
            "bus_bw_gbps": round(bus_bw(algo_bw(nbytes, s["p50_us"]), tp), 2),
        })

    return results


# ---- Section 2: TP layer (vLLM-style) ----


def bench_tp_layer(model_name, cfg, tp, device, dtype,
                   batch, seq_len, warmup, iters):
    hidden = cfg["hidden"]
    intermediate = cfg["intermediate"]

    if hidden % tp != 0 or intermediate % tp != 0:
        return None

    shard_h = hidden // tp
    shard_inter = intermediate // tp
    tokens = batch * seq_len

    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    W_qkv = torch.randn(hidden, 3 * shard_h, dtype=dtype, device=device) / math.sqrt(hidden)
    W_o = torch.randn(shard_h, hidden, dtype=dtype, device=device) / math.sqrt(shard_h)
    W_gate = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
    W_up = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
    W_down = torch.randn(shard_inter, hidden, dtype=dtype, device=device) / math.sqrt(shard_inter)

    def tp_layer():
        qkv = torch.mm(x, W_qkv)
        o = torch.mm(qkv[:, :shard_h], W_o)
        dist.all_reduce(o)
        gate = torch.mm(o, W_gate)
        up = torch.mm(o, W_up)
        act = torch.nn.functional.silu(gate) * up
        down = torch.mm(act, W_down)
        dist.all_reduce(down)

    reset_nccl_tuning(tp_layer, warmup=warmup)
    s = bench(tp_layer, warmup=warmup, iters=iters)

    return {
        "section": "tp_layer",
        "model": model_name,
        "batch_size": batch,
        "seq_len": seq_len,
        "tokens": tokens,
        "scenario": "decode" if seq_len == 1 else "prefill",
        "hidden": hidden,
        "intermediate": intermediate,
        "tp": tp,
        "step": s,
    }


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="vLLM-style TP inference benchmark: standard AllReduce "
                    "at decode and prefill tensor sizes")
    parser.add_argument("--section", default="all",
                        choices=["all", "allreduce", "layer"])
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--decode-batches", type=int, nargs="+",
                        default=DECODE_BATCHES)
    parser.add_argument("--prefill-lengths", type=int, nargs="+",
                        default=PREFILL_LENGTHS)
    parser.add_argument("--dtype", default=DTYPES[0],
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to PATH (rank 0 only)")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    dist.init_process_group(backend="nccl", timeout=BENCH_NCCL_TIMEOUT)
    rank = dist.get_rank()
    tp = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    peaks = get_gpu_peak_bandwidth()
    nvlink_peak = peaks["nvlink_unidir_gbps"]

    json_results = []

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"vLLM-Style TP Inference Benchmark")
        print(f"  TP: {tp}  |  GPU: {torch.cuda.get_device_name(device)}"
              f"  |  dtype: {args.dtype}")
        if nvlink_peak > 0:
            print(f"  NVLink unidir peak: {nvlink_peak} GB/s")
        print(f"{'=' * 80}")

    # ---- Section 1: AllReduce by inference phase ----

    if args.section in ("all", "allreduce"):
        for model_name in args.models:
            cfg = MODELS[model_name]

            if rank == 0:
                print(f"\n{'=' * 80}")
                print(f"  {model_name}: TP AllReduce (hidden={cfg['hidden']})")
                print(f"{'=' * 80}")

                print(f"\n--- Decode phase (batch × hidden) ---")
                hdr = (f"{'batch':>8} {'nbytes':>10}"
                       f" | {'p50_us':>10} {'algo_GB/s':>10} {'bus_GB/s':>10}")
                if nvlink_peak > 0:
                    hdr += f" {'eff_%':>7}"
                print(hdr)
                print("-" * len(hdr))

            results = bench_allreduce_phase(
                model_name, cfg, tp, device, dtype,
                args.decode_batches, args.prefill_lengths,
                args.warmup, args.iters,
            )

            printed_prefill_header = False
            for r in results:
                eff_pct = (round(r["bus_bw_gbps"] / nvlink_peak * 100, 1)
                           if nvlink_peak > 0 else None)
                if eff_pct is not None:
                    r["efficiency_pct"] = eff_pct

                if rank == 0:
                    if r["section"] == "allreduce_prefill" and not printed_prefill_header:
                        print(f"\n--- Prefill phase (seq_len × hidden) ---")
                        hdr = (f"{'seq_len':>8} {'nbytes':>10}"
                               f" | {'p50_us':>10} {'algo_GB/s':>10} {'bus_GB/s':>10}")
                        if nvlink_peak > 0:
                            hdr += f" {'eff_%':>7}"
                        print(hdr)
                        print("-" * len(hdr))
                        printed_prefill_header = True

                    size_key = r.get("batch_size", r.get("seq_len"))
                    line = (
                        f"{size_key:>8} {format_bytes(r['nbytes']):>10}"
                        f" | {r['stats']['p50_us']:>8.1f}us"
                        f" {r['algo_bw_gbps']:>9.1f} {r['bus_bw_gbps']:>9.1f}"
                    )
                    if eff_pct is not None:
                        line += f" {eff_pct:>6.1f}%"
                    print(line)

                json_results.append(r)

    # ---- Section 2: TP layer ----

    if args.section in ("all", "layer"):
        scenarios = [
            ("decode", 32, 1),
            ("prefill", 1, 2048),
        ]

        for model_name in args.models:
            cfg = MODELS[model_name]

            if cfg["hidden"] % tp != 0 or cfg["intermediate"] % tp != 0:
                if rank == 0:
                    print(f"\nSkipping {model_name}: dims not divisible by TP={tp}")
                continue

            if rank == 0:
                print(f"\n{'=' * 80}")
                print(f"  {model_name}: TP Layer (4 GEMMs + 2 AllReduces)")
                print(f"  hidden={cfg['hidden']}  intermediate={cfg['intermediate']}")
                print(f"{'=' * 80}")
                hdr = (f"{'scenario':>10} {'tokens':>8}"
                       f" | {'p50_us':>10} {'step_ms':>10}")
                print(hdr)
                print("-" * len(hdr))

            for scenario_name, batch, seq_len in scenarios:
                try:
                    r = bench_tp_layer(
                        model_name, cfg, tp, device, dtype,
                        batch, seq_len, args.warmup, args.iters,
                    )
                    if r is None:
                        continue

                    if rank == 0:
                        print(
                            f"{scenario_name:>10} {r['tokens']:>8}"
                            f" | {r['step']['p50_us']:>8.1f}us"
                            f" {r['step']['p50_us'] / 1000:>8.1f}ms"
                        )

                    json_results.append(r)
                except Exception as e:
                    if rank == 0:
                        print(f"{scenario_name:>10}  FAILED: {e}")
                    torch.cuda.empty_cache()

    # ---- JSON output ----

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0 and args.json:
        output = collect_metadata("inference_tp_vllm", tp=tp, dtype=args.dtype)
        output["gpu_mem_delta_bytes"] = mem_after - mem_before
        output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
        output["results"] = json_results
        write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
