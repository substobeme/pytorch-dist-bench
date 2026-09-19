"""
Benchmark: FSDP2 training step — forward + backward + optimizer.

Measures a complete FSDP2 training iteration using fully_shard() on a
stack of MLP layers with Llama-70B dimensions. Tests what
bench_training_fsdp_collectives.py does NOT: FSDP2's overlap scheduling,
pre-forward AllGather pipelining, and post-backward ReduceScatter overlap.

The model is a stack of MLP blocks (Linear->ReLU->Linear), each wrapped
with fully_shard() independently. This gives FSDP2 the opportunity to
overlap AllGather of layer N+1 with compute of layer N — the same pattern
used in real Llama training.

Usage:
  torchrun --nproc_per_node=2 bench_fsdp2_training.py
  torchrun --nproc_per_node=8 bench_fsdp2_training.py --json results/fsdp2.json
"""

import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, fsdp_mp_policy,
    reset_nccl_tuning, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# FSDP2 step per dtype with fp32 master weights and dtype all-gather/reduce-
# scatter.
DTYPES = ("bf16", "fp16", "fp32")


class MLPBlock(nn.Module):
    """Single MLP block matching Llama gate_proj + down_proj pattern."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class FSDPBenchModel(nn.Module):
    """Stack of MLP blocks for FSDP2 benchmarking."""

    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def build_and_shard(hidden, intermediate, num_layers, device, dtype):
    """Build model, move to device, apply fully_shard() bottom-up."""
    model = FSDPBenchModel(hidden, intermediate, num_layers).to(device=device)

    mp_policy = fsdp_mp_policy(dtype)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="FSDP2 training step benchmark")
    parser.add_argument("--hidden", type=int, default=8192,
                        help="Hidden dimension (default: 8192, Llama-70B)")
    parser.add_argument("--intermediate", type=int, default=28672,
                        help="Intermediate dimension (default: 28672, Llama-70B)")
    parser.add_argument("--num-layers", type=int, nargs="+", default=[4, 8, 16],
                        help="Number of MLP layers to stack")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16],
                        help="Batch sizes to sweep")
    parser.add_argument("--dtype", default=DTYPES[0],
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
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

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"FSDP2 Training Step Benchmark (fully_shard)")
        print(f"  DP: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}")
        print(f"  Measures: zero_grad + forward + backward + optimizer.step")
        print(f"{'=' * 95}")
        print()
        hdr = (f"{'layers':>6} {'batch':>5} {'params':>10} {'sharded':>10}"
               f" | {'step_us':>10} {'step_ms':>10}"
               f" | {'peak_MB':>10}")
        print(hdr)
        print("-" * len(hdr))

    for num_layers in args.num_layers:
        for batch_size in args.batch_sizes:
            try:
                torch.cuda.reset_peak_memory_stats(device)
                mem_pre = torch.cuda.memory_allocated(device)

                model = build_and_shard(args.hidden, args.intermediate,
                                        num_layers, device, dtype)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                inp = torch.randn(batch_size, args.hidden, dtype=dtype,
                                  device=device)

                total_params = sum(p.numel() for p in model.parameters())
                # Sharded params are fp32 master weights regardless of dtype.
                param_bytes = sum(p.numel() * p.element_size()
                                  for p in model.parameters())
                shard_bytes = param_bytes // world_size

                def step():
                    optimizer.zero_grad()
                    loss = model(inp).sum()
                    loss.backward()
                    optimizer.step()

                reset_nccl_tuning(step)
                s = bench(step, warmup=args.warmup, iters=args.iters)

                peak_mb = (torch.cuda.max_memory_allocated(device) - mem_pre) / 1024**2

                if rank == 0:
                    print(
                        f"{num_layers:>6} {batch_size:>5}"
                        f" {total_params:>10,}"
                        f" {shard_bytes // 1024**2:>8} MB"
                        f" | {s['p50_us']:>8.0f}us {s['p50_us'] / 1000:>8.1f}ms"
                        f" | {peak_mb:>8.0f} MB"
                    )

                    json_results.append({
                        "num_layers": num_layers,
                        "batch_size": batch_size,
                        "hidden": args.hidden,
                        "intermediate": args.intermediate,
                        "total_params": total_params,
                        "param_bytes": param_bytes,
                        "shard_bytes": shard_bytes,
                        "step": s,
                        "peak_mem_mb": round(peak_mb, 1),
                    })

                del model, optimizer, inp
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"{num_layers:>6} {batch_size:>5}  OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"{num_layers:>6} {batch_size:>5}  FAILED: {e}")
                torch.cuda.empty_cache()

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  FSDP2 fully_shard() applied per-layer (bottom-up)")
        print(f"  Each layer's AllGather overlaps with previous layer's compute")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata("fsdp2_training", dp=world_size,
                                      dtype=args.dtype,
                                      hidden=args.hidden,
                                      intermediate=args.intermediate)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
