"""
Benchmark: MoE expert-parallel all-to-all dispatch.

Measures the all-to-all collective that routes tokens to experts in
Mixture-of-Experts models. This is the dominant communication primitive
for expert parallelism (EP), distinct from TP's AG/RS pattern.

Two phases per MoE layer:
  1. Dispatch: all-to-all sends tokens to the rank that owns each expert
  2. Combine:  all-to-all returns expert outputs back to original ranks

Benchmarks both equal-split (perfectly balanced routing) and unequal-split
(realistic skewed routing where popular experts get more tokens) patterns.

Uses dist.all_to_all_single (standard NCCL path). PyTorch also has
symmetric-memory variants (all_to_all_nd via NCCL LSA, all_to_all_vdev_2d
via NVSHMEM) that require NCCL 2.29+ and may not be available in all builds.

Model shapes from:
  Mixtral-8x7B:   8 experts, top_k=2, hidden=4096, intermediate=14336
  DeepSeek-V2:  160 experts, top_k=6, hidden=5120, intermediate=1536
  DeepSeek-V3:  256 experts, top_k=8, hidden=7168, intermediate=2048

Usage:
  torchrun --nproc_per_node=2 bench_moe_alltoall.py
  torchrun --nproc_per_node=8 bench_moe_alltoall.py --json results/moe.json
"""

import argparse

import torch
import torch.distributed as dist

from bench_utils import BENCH_NCCL_TIMEOUT, bench, collect_metadata, reset_nccl_tuning, write_json

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# all_to_all_single moves bytes; the sweep records each dtype's message sizes.
DTYPES = ("bf16", "fp16", "fp32")


MODELS = {
    "Mixtral-8x7B": {
        "num_experts": 8,
        "top_k": 2,
        "hidden": 4096,
        "intermediate": 14336,
    },
    "DeepSeek-V2": {
        "num_experts": 160,
        "top_k": 6,
        "hidden": 5120,
        "intermediate": 1536,
    },
    "DeepSeek-V3": {
        "num_experts": 256,
        "top_k": 8,
        "hidden": 7168,
        "intermediate": 2048,
    },
}

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]



def verify_alltoall(rank, world_size, device, dtype):
    """Verify all-to-all correctness with both small and realistic tensors.

    1. Small check: each rank sends its rank value, expects values from all ranks.
    2. Roundtrip check: dispatch then combine should recover original data.
    """
    # Small element check
    send = torch.full((world_size, 4), rank, dtype=dtype, device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()

    for src in range(world_size):
        expected = torch.full((4,), src, dtype=dtype, device=device)
        if not torch.equal(recv[src], expected):
            raise RuntimeError(
                f"all_to_all correctness failed: rank {rank} expected "
                f"{expected} from rank {src}, got {recv[src]}")

    # Roundtrip check: dispatch -> combine should recover original data
    hidden = 128
    tokens_per_rank = 32
    original = torch.randn(tokens_per_rank * world_size, hidden,
                           dtype=dtype, device=device)
    dispatched = torch.empty_like(original)
    recovered = torch.empty_like(original)
    dist.all_to_all_single(dispatched, original)
    dist.all_to_all_single(recovered, dispatched)
    torch.cuda.synchronize()
    if not torch.equal(original, recovered):
        max_diff = (original - recovered).abs().max().item()
        raise RuntimeError(
            f"all_to_all roundtrip failed: max_diff={max_diff:.6f}")


def make_routing(num_tokens, top_k, num_experts, world_size, rank,
                 balanced=True):
    """Generate token-to-expert routing assignments.

    Returns (input_splits, output_splits) for all_to_all_single.
    Each rank owns num_experts // world_size experts.
    Routing is synthetic (static), not gating-derived.

    balanced=True:  tokens distributed evenly across experts
    balanced=False: zipf-like skew (some experts get many more tokens)

    Symmetric routing: every rank sends the same input_splits pattern.
    Therefore rank r receives input_splits[r] from each sender, so
    output_splits = [input_splits[r]] * world_size.
    """
    total_routed = num_tokens * top_k

    if balanced:
        base = total_routed // world_size
        remainder = total_routed % world_size
        input_splits = [base + (1 if i < remainder else 0)
                        for i in range(world_size)]
    else:
        weights = [1.0 / (i + 1) for i in range(world_size)]
        total_w = sum(weights)
        raw = [int(total_routed * w / total_w) for w in weights]
        diff = total_routed - sum(raw)
        for i in range(abs(diff)):
            raw[i % world_size] += 1 if diff > 0 else -1
        input_splits = raw

    if balanced:
        output_splits = input_splits[:]
    else:
        # Symmetric routing: every rank sends input_splits[dest] to dest,
        # so rank r receives input_splits[r] from each of the P senders.
        output_splits = [input_splits[rank]] * world_size

    return input_splits, output_splits


def bench_equal_split(world_size, num_tokens, hidden, top_k, num_experts,
                      dtype, device, warmup, iters):
    """Benchmark all-to-all with equal splits (balanced routing)."""
    total_routed = num_tokens * top_k
    if total_routed % world_size != 0:
        raise ValueError(
            f"total_routed ({total_routed}) not divisible by world_size ({world_size})")

    send_buf = torch.randn(total_routed, hidden, dtype=dtype, device=device)
    recv_buf = torch.empty(total_routed, hidden, dtype=dtype, device=device)

    def dispatch():
        dist.all_to_all_single(recv_buf, send_buf)

    def combine():
        dist.all_to_all_single(send_buf, recv_buf)

    reset_nccl_tuning(dispatch)
    s_dispatch = bench(dispatch, warmup=warmup, iters=iters)
    s_combine = bench(combine, warmup=warmup, iters=iters)

    return s_dispatch, s_combine, total_routed * hidden * dtype.itemsize


def bench_unequal_split(rank, world_size, num_tokens, hidden, top_k,
                        num_experts, dtype, device, warmup, iters):
    """Benchmark all-to-all with unequal splits (skewed routing)."""
    input_splits, output_splits = make_routing(
        num_tokens, top_k, num_experts, world_size, rank, balanced=False)

    total_send = sum(input_splits)
    total_recv = sum(output_splits)

    send_buf = torch.randn(total_send, hidden, dtype=dtype, device=device)
    recv_buf = torch.empty(total_recv, hidden, dtype=dtype, device=device)

    def dispatch():
        dist.all_to_all_single(recv_buf, send_buf, output_splits, input_splits)

    def combine():
        dist.all_to_all_single(send_buf, recv_buf, input_splits, output_splits)

    reset_nccl_tuning(dispatch)
    s_dispatch = bench(dispatch, warmup=warmup, iters=iters)
    s_combine = bench(combine, warmup=warmup, iters=iters)

    return (s_dispatch, s_combine,
            total_send * hidden * dtype.itemsize,
            input_splits, output_splits)


def main():
    parser = argparse.ArgumentParser(
        description="MoE expert-parallel all-to-all benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: all compatible)")
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKEN_COUNTS)
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
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Filter models: num_experts must be >= world_size for expert parallelism
    model_names = args.models or list(MODELS.keys())
    models_to_run = {}
    for name in model_names:
        if name not in MODELS:
            if rank == 0:
                print(f"Unknown model: {name}")
            continue
        cfg = MODELS[name]
        if cfg["num_experts"] < world_size:
            if rank == 0:
                print(f"Skipping {name}: {cfg['num_experts']} experts < {world_size} ranks")
            continue
        models_to_run[name] = cfg

    if not models_to_run:
        if rank == 0:
            print("No compatible models for this world size.")
        dist.destroy_process_group()
        return

    # Correctness check
    verify_alltoall(rank, world_size, device, dtype)

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 100}")
        print(f"MoE All-to-All Benchmark (Expert Parallel)")
        print(f"  EP: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"{'=' * 100}")

    # ---- Section 1: Balanced routing (equal split) ----
    if rank == 0:
        print(f"\n--- Balanced Routing (equal split across ranks) ---")
        print(f"  Best case: tokens evenly distributed across experts")
        print()
        hdr = (f"{'model':>15} {'tokens':>6} {'top_k':>5} {'routed':>6} {'hidden':>6} {'nbytes':>10}"
               f" | {'dispatch':>10} {'combine':>10} {'roundtrip':>11}")
        print(hdr)
        print("-" * len(hdr))

    for model_name, cfg in models_to_run.items():
        for num_tokens in args.tokens:
            try:
                s_dispatch, s_combine, nbytes = bench_equal_split(
                    world_size, num_tokens, cfg["hidden"],
                    cfg["top_k"], cfg["num_experts"],
                    dtype, device, args.warmup, args.iters,
                )
                if rank == 0:
                    total_routed = num_tokens * cfg["top_k"]
                    roundtrip = s_dispatch["p50_us"] + s_combine["p50_us"]
                    print(
                        f"{model_name:>15} {num_tokens:>6}"
                        f" {cfg['top_k']:>5} {total_routed:>6}"
                        f" {cfg['hidden']:>6} {nbytes:>10}"
                        f" | {s_dispatch['p50_us']:>8.1f}us"
                        f" {s_combine['p50_us']:>8.1f}us"
                        f" {roundtrip:>9.1f}us"
                    )
                    json_results.append({
                        "routing": "balanced",
                        "model": model_name,
                        "num_tokens": num_tokens,
                        "top_k": cfg["top_k"],
                        "num_experts": cfg["num_experts"],
                        "hidden": cfg["hidden"],
                        "total_routed": total_routed,
                        "nbytes": nbytes,
                        "dispatch": s_dispatch,
                        "combine": s_combine,
                        "roundtrip_p50_us": round(roundtrip, 1),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>15} {num_tokens:>6} FAILED: {e}")

    # ---- Section 2: Skewed routing (unequal split) ----
    if rank == 0:
        print(f"\n--- Skewed Routing (Zipf-like imbalance) ---")
        print(f"  Realistic case: popular experts get more tokens")
        print()
        hdr = (f"{'model':>15} {'tokens':>6} {'top_k':>5} {'skew_ratio':>10}"
               f" | {'dispatch':>10} {'combine':>10} {'roundtrip':>11}"
               f" | {'vs_balanced':>11}")
        print(hdr)
        print("-" * len(hdr))

    for model_name, cfg in models_to_run.items():
        for num_tokens in args.tokens:
            try:
                (s_dispatch, s_combine, nbytes,
                 in_splits, out_splits) = bench_unequal_split(
                    rank, world_size, num_tokens, cfg["hidden"],
                    cfg["top_k"], cfg["num_experts"],
                    dtype, device, args.warmup, args.iters,
                )
                if rank == 0:
                    roundtrip = s_dispatch["p50_us"] + s_combine["p50_us"]
                    skew = max(in_splits) / max(min(in_splits), 1)

                    # Find balanced roundtrip for comparison
                    balanced_rt = None
                    for r in json_results:
                        if (r.get("routing") == "balanced"
                                and r["model"] == model_name
                                and r["num_tokens"] == num_tokens):
                            balanced_rt = r["roundtrip_p50_us"]
                            break

                    line = (
                        f"{model_name:>15} {num_tokens:>6}"
                        f" {cfg['top_k']:>5} {skew:>9.1f}x"
                        f" | {s_dispatch['p50_us']:>8.1f}us"
                        f" {s_combine['p50_us']:>8.1f}us"
                        f" {roundtrip:>9.1f}us"
                    )
                    if balanced_rt:
                        overhead = (roundtrip - balanced_rt) / balanced_rt * 100
                        line += f" | {overhead:>+9.1f}%"
                    else:
                        line += f" | {'N/A':>11}"
                    print(line)

                    json_results.append({
                        "routing": "skewed",
                        "model": model_name,
                        "num_tokens": num_tokens,
                        "top_k": cfg["top_k"],
                        "num_experts": cfg["num_experts"],
                        "hidden": cfg["hidden"],
                        "nbytes": nbytes,
                        "input_splits": in_splits,
                        "output_splits": out_splits,
                        "skew_ratio": round(skew, 2),
                        "dispatch": s_dispatch,
                        "combine": s_combine,
                        "roundtrip_p50_us": round(roundtrip, 1),
                        "overhead_vs_balanced_pct": (
                            round((roundtrip - balanced_rt) / balanced_rt * 100, 1)
                            if balanced_rt else None
                        ),
                    })
            except Exception as e:
                if rank == 0:
                    print(f"{model_name:>15} {num_tokens:>6} FAILED: {e}")

    # ---- Section 3: Scaling with message size ----
    if rank == 0:
        print(f"\n--- Message Size Scaling (fixed 1024 tokens, varying hidden) ---")
        print(f"  Shows all-to-all bandwidth saturation point")
        print()
        hdr = (f"{'hidden':>8} {'nbytes':>10} {'top_k':>5}"
               f" | {'dispatch':>10} {'combine':>10}"
               f" | {'algo_GB/s':>10}")
        print(hdr)
        print("-" * len(hdr))

    hidden_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    fixed_tokens = 1024
    fixed_top_k = 2

    for hidden in hidden_sizes:
        try:
            total_routed = fixed_tokens * fixed_top_k
            send_buf = torch.randn(total_routed, hidden, dtype=dtype, device=device)
            recv_buf = torch.empty_like(send_buf)

            def dispatch():
                dist.all_to_all_single(recv_buf, send_buf)

            reset_nccl_tuning(dispatch)
            s = bench(dispatch, warmup=args.warmup, iters=args.iters)
            nbytes = total_routed * hidden * dtype.itemsize
            bw = nbytes / (s["p50_us"] / 1e6) / 1e9

            s_combine = bench(
                lambda: dist.all_to_all_single(send_buf, recv_buf),
                warmup=args.warmup, iters=args.iters,
            )

            if rank == 0:
                print(
                    f"{hidden:>8} {nbytes:>10} {fixed_top_k:>5}"
                    f" | {s['p50_us']:>8.1f}us"
                    f" {s_combine['p50_us']:>8.1f}us"
                    f" | {bw:>8.1f}"
                )
                json_results.append({
                    "routing": "scaling",
                    "hidden": hidden,
                    "num_tokens": fixed_tokens,
                    "top_k": fixed_top_k,
                    "total_routed": total_routed,
                    "nbytes": nbytes,
                    "dispatch": s,
                    "combine": s_combine,
                    "algo_bw_gbps": round(bw, 2),
                })
        except Exception as e:
            if rank == 0:
                print(f"{hidden:>8} FAILED: {e}")

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 100}")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"  Scope: intra-node only (NVLink). Inter-node IB/RoCE not covered.")
        print(f"  Note: real MoE routing is dynamic (gating network). These are static/synthetic.")
        print(f"  algo_GB/s = total_bytes / time (per-rank throughput, not NVLink bus BW).")
        print(f"{'=' * 100}\n")

        if args.json:
            output = collect_metadata("moe_alltoall", ep=world_size, dtype=args.dtype)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
