"""
Benchmark: Full TP inference layer — fused vs unfused.

Simulates the attention + MLP blocks of one transformer layer under TP:

  Attention block (AG + RS pair #1):
    AG(x) -> GEMM(qkv) -> [attention omitted] -> GEMM(o_proj) -> RS

  MLP block (AG + RS pair #2):
    AG(x) -> GEMM(gate) + GEMM(up) -> SiLU*mul -> GEMM(down) -> RS

Each transformer layer has TWO AG+RS pairs. We benchmark each block
separately and report the combined layer time. Attention computation
(softmax, rotary) is omitted — we measure only the TP-collective-bound
portion of each block.

Fused path uses symmetric memory ops to overlap communication with compute:
  fused_all_gather_matmul  (AG + GEMM)
  fused_matmul_reduce_scatter  (GEMM + RS)

Usage:
  torchrun --nproc_per_node=2 bench_inference_tp_layer.py
  torchrun --nproc_per_node=8 bench_inference_tp_layer.py --json results/tp_layer.json
"""

import argparse
import math

import torch
import torch.distributed as dist
try:
    import torch.distributed._symmetric_memory as symm_mem
except (ImportError, ModuleNotFoundError):
    raise SystemExit(
        "bench_inference_tp_layer requires torch.distributed._symmetric_memory "
        "(not available in this PyTorch build)"
    )

from bench_utils import BENCH_NCCL_TIMEOUT, bench, collect_metadata, reset_nccl_tuning, verify_close, write_json

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# Fused symm-mem GEMMs serve 16-bit inference; fp32 at 405B/S=32K is ~10x
# slower per iter and overruns run_all's timeout. Force with --dtype fp32.
DTYPES = ("bf16", "fp16")


MODELS = {
    "Llama-8B":   {"hidden": 4096, "intermediate": 14336, "num_layers": 32,
                   "num_kv_heads": 8},
    "Llama-70B":  {"hidden": 8192, "intermediate": 28672, "num_layers": 80,
                   "num_kv_heads": 8},
    "Llama-405B": {"hidden": 16384, "intermediate": 53248, "num_layers": 126,
                   "num_kv_heads": 8},
}

SEQ_LENGTHS = [128, 512, 2048, 8192, 32768]



def bench_layer(group_name, rank, tp, seq_len, hidden, intermediate,
                num_kv_heads, dtype=torch.bfloat16, warmup=50, iters=200):
    """Benchmark one full transformer layer (attention + MLP TP projections).

    Attention block: AG -> GEMM(qkv) -> [attn omitted] -> GEMM(o_proj) -> RS
    MLP block:       AG -> GEMM(gate)+GEMM(up) -> SiLU*mul -> GEMM(down) -> RS

    Returns (unfused_stats, fused_stats) for the combined layer.
    """
    device = torch.device(f"cuda:{rank}")
    shard_s = seq_len // tp
    shard_inter = intermediate // tp
    qkv_out = hidden // tp  # per-rank output of qkv_proj
    o_in = hidden // tp      # per-rank input of o_proj

    x = torch.randn(shard_s, hidden, dtype=dtype, device=device)

    # Kaiming-scaled init to prevent value explosion through chained matmuls
    W_qkv = torch.randn(hidden, qkv_out, dtype=dtype, device=device) / math.sqrt(hidden)
    W_o = torch.randn(o_in, hidden, dtype=dtype, device=device) / math.sqrt(o_in)
    W_gate = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
    W_up = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
    W_down = torch.randn(shard_inter, hidden, dtype=dtype, device=device) / math.sqrt(shard_inter)

    # Pre-allocated buffers for unfused path
    gathered = torch.empty(seq_len, hidden, dtype=dtype, device=device)
    out_attn_rs = torch.empty(shard_s, hidden, dtype=dtype, device=device)
    out_mlp_rs = torch.empty(shard_s, hidden, dtype=dtype, device=device)

    def unfused_layer():
        # --- Attention block ---
        dist.all_gather_into_tensor(gathered, x, group=dist.group.WORLD)
        qkv = torch.mm(gathered, W_qkv)
        # Attention computation omitted — not TP-collective-bound
        o_proj_in = qkv  # placeholder: same shape as real o_proj input
        proj = torch.mm(o_proj_in, W_o)
        dist.reduce_scatter_tensor(out_attn_rs, proj, group=dist.group.WORLD)

        # --- MLP block ---
        dist.all_gather_into_tensor(gathered, out_attn_rs, group=dist.group.WORLD)
        gate = torch.mm(gathered, W_gate)
        up = torch.mm(gathered, W_up)
        act = torch.nn.functional.silu(gate) * up
        proj = torch.mm(act, W_down)
        dist.reduce_scatter_tensor(out_mlp_rs, proj, group=dist.group.WORLD)

    def fused_layer():
        # --- Attention block ---
        _, (qkv,) = symm_mem._fused_all_gather_matmul(
            x, [W_qkv], gather_dim=0, group_name=group_name,
        )
        o_proj_in = qkv
        attn_out = symm_mem._fused_matmul_reduce_scatter(
            o_proj_in, W_o, "sum", scatter_dim=0, group_name=group_name,
        )

        # --- MLP block ---
        _, (gate, up) = symm_mem._fused_all_gather_matmul(
            attn_out, [W_gate, W_up], gather_dim=0, group_name=group_name,
        )
        act = torch.nn.functional.silu(gate) * up
        return symm_mem._fused_matmul_reduce_scatter(
            act, W_down, "sum", scatter_dim=0, group_name=group_name,
        )

    reset_nccl_tuning(unfused_layer)

    unfused_layer()
    torch.cuda.synchronize()
    ref = out_mlp_rs.clone()
    fused_out = fused_layer()
    torch.cuda.synchronize()
    verify_close("tp_layer", ref, fused_out)

    return bench(unfused_layer, warmup=warmup, iters=iters), bench(fused_layer, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser(
        description="Full TP inference layer benchmark: fused vs unfused")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=SEQ_LENGTHS)
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

    group_name = dist.group.WORLD.group_name

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"TP Inference Layer Benchmark (Attention + MLP)")
        print(f"  TP: {tp}  |  GPU: {torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"{'=' * 95}")
        print()
        print(f"  Attention: AG(x) -> GEMM(qkv) -> [attn omitted] -> GEMM(o_proj) -> RS")
        print(f"  MLP:       AG(x) -> GEMM(gate)+GEMM(up) -> SiLU*mul -> GEMM(down) -> RS")
        print(f"  Each layer has 2 AG+RS pairs. Fused ops overlap each with its GEMM.")
        print()
        hdr = (f"{'model':>12} {'S':>6} {'layers':>6}"
               f" | {'unfused':>10} {'fused':>10} {'speedup':>8}"
               f" | {'delta/layer':>12} {'delta*layers':>13}")
        print(hdr)
        print("-" * len(hdr))

    for model_name in args.models:
        cfg = MODELS[model_name]
        if cfg["hidden"] % tp != 0 or cfg["intermediate"] % tp != 0:
            continue
        for seq_len in args.seq_lengths:
            if seq_len < tp or seq_len % tp != 0:
                continue
            try:
                s_unfused, s_fused = bench_layer(
                    group_name, rank, tp, seq_len,
                    cfg["hidden"], cfg["intermediate"],
                    cfg["num_kv_heads"], dtype,
                    warmup=args.warmup, iters=args.iters,
                )
                if rank == 0:
                    speedup = s_unfused["p50_us"] / max(s_fused["p50_us"], 0.1)
                    delta_us = s_unfused["p50_us"] - s_fused["p50_us"]
                    total_delta_ms = delta_us * cfg["num_layers"] / 1000

                    print(
                        f"{model_name:>12} {seq_len:>6} {cfg['num_layers']:>6}"
                        f" | {s_unfused['p50_us']:>8.1f}us {s_fused['p50_us']:>8.1f}us"
                        f" {speedup:>7.2f}x"
                        f" | {delta_us:>10.1f}us {total_delta_ms:>11.1f}ms"
                    )

                    json_results.append({
                        "model": model_name,
                        "seq_len": seq_len,
                        "num_layers": cfg["num_layers"],
                        "hidden": cfg["hidden"],
                        "intermediate": cfg["intermediate"],
                        "unfused_layer": s_unfused,
                        "fused_layer": s_fused,
                        "layer_speedup": round(speedup, 3),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>12} {seq_len:>6} FAILED: {e}")

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print()
        print("  'delta/layer': unfused - fused per layer (positive = fused is faster)")
        print("  'delta*layers': estimated total improvement across all layers")
        print("    Measures TP projection GEMMs + collectives (attention + MLP).")
        print("    Attention dot-product/softmax/rotary are omitted (not TP-collective-bound).")
        print("    For prefill TTFT: delta*layers approximates the collective-side TTFT reduction.")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata("inference_tp_layer", tp=tp, dtype=args.dtype)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
