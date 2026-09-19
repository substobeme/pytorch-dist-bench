"""
Benchmark: FSDP2 training collectives — standard vs symmetric memory.

Simulates FSDP2's communication pattern:
  AllGather to unshard parameters before forward
  ReduceScatter to shard gradients after backward

Compares:
  1. Standard NCCL: dist.all_gather_into_tensor / dist.reduce_scatter_tensor
  2. Symmetric memory AG: _low_contention_all_gather (P2P copy engine, zero SM usage)
  3. NVLS AllReduce: multimem_all_reduce for DDP-style gradient sync

Tensor shapes match FSDP shard sizes for Llama-70B layer weights.

Usage:
  torchrun --nproc_per_node=2 bench_training_fsdp_collectives.py
  torchrun --nproc_per_node=8 bench_training_fsdp_collectives.py --json results/fsdp.json
"""

import argparse

import torch
import torch.distributed as dist
try:
    import torch.distributed._symmetric_memory as symm_mem
except (ImportError, ModuleNotFoundError):
    raise SystemExit(
        "bench_training_fsdp_collectives requires torch.distributed._symmetric_memory "
        "(not available in this PyTorch build)"
    )

from bench_utils import BENCH_NCCL_TIMEOUT, bench, collect_metadata, reset_nccl_tuning, write_json

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# AG/RS per dtype; NVLS all-reduce runs for bf16/fp32 (no fp16 kernel).
DTYPES = ("bf16", "fp16", "fp32")


# Llama-70B parameter shapes (the training-relevant model)
PARAMS = [
    ("q_proj",    8192,  8192),
    ("k_proj",    8192,  1024),
    ("v_proj",    8192,  1024),
    ("o_proj",    1024,  8192),  # GQA head dim -> hidden
    ("gate_proj", 8192,  28672),
    ("up_proj",   8192,  28672),
    ("down_proj", 28672, 8192),
]



def main():
    parser = argparse.ArgumentParser(
        description="FSDP2 training collectives benchmark")
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
    dp = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    group_name = dist.group.WORLD.group_name

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    json_results = []

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"FSDP2 Training Collectives Benchmark")
        print(f"  DP: {dp}  |  GPU: {torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"  Model: Llama-70B parameter shapes")
        print(f"{'=' * 95}")

    # ---- Section 1: AllGather (parameter unshard) ----
    if rank == 0:
        print(f"\n--- AllGather: Parameter Unshard ---")
        print(f"  FSDP2 gathers full parameters from shards before forward pass")
        print()
        hdr = (f"{'param':>12} {'full_shape':>16} {'shard_shape':>16}"
               f" | {'standard':>10} {'low_cont':>10} {'speedup':>8}")
        print(hdr)
        print("-" * len(hdr))

    for param_name, rows, cols in PARAMS:
        if cols % dp != 0 or cols < dp:
            continue
        shard_cols = cols // dp

        shard = torch.randn(rows, shard_cols, dtype=dtype, device=device)
        full_standard = torch.empty(rows, cols, dtype=dtype, device=device)

        def standard_ag():
            dist.all_gather_into_tensor(full_standard, shard, group=dist.group.WORLD)
        reset_nccl_tuning(standard_ag)
        s_standard = bench(standard_ag, warmup=args.warmup, iters=args.iters)

        # Low-contention AG via symmetric memory (P2P copy engine)
        symm_shard = symm_mem.empty(rows * shard_cols, dtype=dtype, device=device)
        symm_mem.rendezvous(symm_shard, group_name)
        symm_shard_2d = symm_shard.view(rows, shard_cols)

        s_low_cont = None
        try:
            symm_shard_2d.copy_(shard)
            def low_contention_ag():
                torch.ops.symm_mem._low_contention_all_gather(symm_shard, group_name)
            s_low_cont = bench(low_contention_ag, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            if rank == 0:
                print(f"  low_contention_all_gather not available: {e}")

        if rank == 0:
            line = (f"{param_name:>12}"
                    f" [{rows:>5}x{cols:<5}]"
                    f" [{rows:>5}x{shard_cols:<5}]"
                    f" | {s_standard['p50_us']:>8.1f}us")

            if s_low_cont is not None:
                speedup = s_standard["p50_us"] / max(s_low_cont["p50_us"], 0.1)
                line += f" {s_low_cont['p50_us']:>8.1f}us {speedup:>7.2f}x"
            else:
                line += f" {'N/A':>10} {'N/A':>8}"
            print(line)

            json_results.append({
                "op": "allgather_unshard",
                "param_name": param_name,
                "param_shape": [rows, cols],
                "shard_shape": [rows, shard_cols],
                "standard": s_standard,
                "low_contention": s_low_cont,
                "speedup": round(s_standard["p50_us"] / max(s_low_cont["p50_us"], 0.1), 3) if s_low_cont else None,
            })

    # ---- Section 2: ReduceScatter (gradient shard) ----
    if rank == 0:
        print(f"\n--- ReduceScatter: Gradient Shard ---")
        print(f"  FSDP2 reduces and scatters gradients after backward pass")
        print()
        hdr = (f"{'param':>12} {'grad_shape':>16} {'shard_shape':>16}"
               f" | {'standard':>10}")
        print(hdr)
        print("-" * len(hdr))

    for param_name, rows, cols in PARAMS:
        if cols % dp != 0 or cols < dp:
            continue
        shard_cols = cols // dp

        grad = torch.randn(rows, cols, dtype=dtype, device=device)
        out = torch.empty(rows, shard_cols, dtype=dtype, device=device)

        def standard_rs():
            dist.reduce_scatter_tensor(out, grad, group=dist.group.WORLD)
        reset_nccl_tuning(standard_rs)
        s_standard = bench(standard_rs, warmup=args.warmup, iters=args.iters)

        if rank == 0:
            print(
                f"{param_name:>12}"
                f" [{rows:>5}x{cols:<5}]"
                f" [{rows:>5}x{shard_cols:<5}]"
                f" | {s_standard['p50_us']:>8.1f}us"
            )
            json_results.append({
                "op": "reducescatter_gradient",
                "param_name": param_name,
                "grad_shape": [rows, cols],
                "shard_shape": [rows, shard_cols],
                "standard": s_standard,
            })

    # ---- Section 3: AllReduce (DDP gradient sync) ----
    if rank == 0:
        print(f"\n--- AllReduce: DDP Gradient Sync (NVLS vs standard) ---")
        print(f"  multimem_all_reduce (NVSwitch) vs standard NCCL all_reduce")
        print()
        hdr = (f"{'param':>12} {'shape':>16} {'nbytes':>10}"
               f" | {'standard':>10} {'nvls':>10} {'speedup':>8}")
        print(hdr)
        print("-" * len(hdr))

    for param_name, rows, cols in PARAMS:
        nelems = rows * cols
        regular = torch.randn(nelems, dtype=dtype, device=device)
        buf = symm_mem.empty(nelems, dtype=dtype, device=device)
        symm_mem.rendezvous(buf, group_name)

        def standard_ar():
            dist.all_reduce(regular, group=dist.group.WORLD)
        reset_nccl_tuning(standard_ar)
        s_standard = bench(standard_ar, warmup=args.warmup, iters=args.iters)

        s_nvls = None
        try:
            def nvls_ar():
                buf.copy_(regular)
                if dp in (4, 6, 8):
                    torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", group_name)
                else:
                    torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", group_name)
            s_nvls = bench(nvls_ar, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            if rank == 0:
                print(f"  NVLS all_reduce not available: {e}")

        if rank == 0:
            nbytes = nelems * dtype.itemsize
            line = (f"{param_name:>12}"
                    f" [{rows:>5}x{cols:<5}]"
                    f" {nbytes:>10}")

            line += f" | {s_standard['p50_us']:>8.1f}us"
            if s_nvls is not None:
                speedup = s_standard["p50_us"] / max(s_nvls["p50_us"], 0.1)
                line += f" {s_nvls['p50_us']:>8.1f}us {speedup:>7.2f}x"
            else:
                line += f" {'N/A':>10} {'N/A':>8}"
            print(line)

            json_results.append({
                "op": "allreduce_gradient",
                "param_name": param_name,
                "param_shape": [rows, cols],
                "nbytes": nelems * dtype.itemsize,
                "standard": s_standard,
                "nvls": s_nvls,
                "speedup": round(s_standard["p50_us"] / max(s_nvls["p50_us"], 0.1), 3) if s_nvls else None,
            })

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  GPU memory delta: {(mem_after - mem_before) / 1024**2:.1f} MB")
        print(f"  Scope: intra-node only (NVLink). Inter-node IB/RoCE not covered.")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata("training_fsdp_collectives", dp=dp, dtype=args.dtype)
            output["gpu_mem_delta_bytes"] = mem_after - mem_before
            output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
