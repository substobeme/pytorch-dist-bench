"""
Benchmark: Multi-node distributed — collectives by topology + 2D training.

Section 1: Raw collectives decomposed by network topology.
  Measures AllReduce, AllGather, and ReduceScatter on four groups:
    - intra_node:  ranks on the same node (NVLink)
    - inter_node:  one rank per node (raw IB/RoCE link bandwidth)
    - inter_agg:   all ranks' inter-node groups simultaneously (aggregate
                   IB/RoCE bandwidth under contention — what DP sees)
    - world:       all ranks (hierarchical, mixed NVLink + IB/RoCE)

  Isolates whether the bottleneck is intra-node (NVLink) or inter-node
  (IB/RoCE) and measures link contention from multiple DP streams.

Section 2: 2D parallelism training step.
  TP intra-node + FSDP2 DP inter-node using DeviceMesh.
  TP via ColwiseParallel/RowwiseParallel, DP via fully_shard().
  Measures a complete training step with realistic overlap scheduling.

Multi-node only — requires torchrun with --nnodes > 1 or Kubernetes
PyTorchJob. Single-node smoke test works but inter-node sections are
skipped (1-member group = no-op).

Usage:
  # On each node (or via pdsh/srun/PyTorchJob):
  torchrun --nnodes=3 --nproc_per_node=2 \\
    --rdzv_backend=c10d --rdzv_endpoint=master:29500 \\
    bench_multinode.py --json results/multinode_3n2g.json

  # Sections can be run independently:
  torchrun ... bench_multinode.py --section collectives
  torchrun ... bench_multinode.py --section training
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, fsdp_mp_policy,
    get_gpu_peak_bandwidth, reset_nccl_tuning, sizes_in_elems, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# Opt-in multi-node; collectives in bytes (SIZES), 2D training with fp32 master
# weights.
DTYPES = ("bf16", "fp16", "fp32")


# ---- Collective helpers (same as bench_collectives.py) ----

# Message sizes in bytes, 1 KB .. 1 GB.
SIZES = [1 << n for n in range(10, 31, 2)]


def algo_bw(nbytes, p50_us):
    if p50_us <= 0:
        return 0.0
    return nbytes / (p50_us * 1e-6) / 1e9


def bus_bw(algo_gbps, group_size, collective):
    n = group_size
    if collective == "all_reduce":
        factor = 2 * (n - 1) / n
    elif collective in ("all_gather", "reduce_scatter"):
        factor = (n - 1) / n
    else:
        factor = 1.0
    return algo_gbps * factor


def format_bytes(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


# ---- Topology setup ----


def setup_groups(world_size, rank, local_world_size, local_rank):
    """Create intra-node and inter-node process groups.

    All ranks must call new_group() in the same order even if they are
    not members — NCCL communicator creation is a collective operation.

    Returns (intra_group, inter_group, num_nodes, node_id).
    """
    num_nodes = world_size // local_world_size
    node_id = rank // local_world_size

    # Intra-node: ranks on the same physical node
    intra_group = None
    for n in range(num_nodes):
        ranks = list(range(n * local_world_size, (n + 1) * local_world_size))
        g = dist.new_group(ranks)
        if n == node_id:
            intra_group = g

    # Inter-node: one group per local_rank, spanning all nodes
    inter_group = None
    for lr in range(local_world_size):
        ranks = [lr + n * local_world_size for n in range(num_nodes)]
        g = dist.new_group(ranks)
        if lr == local_rank:
            inter_group = g

    return intra_group, inter_group, num_nodes, node_id


# ---- Collective benchmarking ----


def bench_collective(collective, group, device, dtype, sizes,
                     group_size, warmup, iters):
    """Benchmark a single collective type across message sizes."""
    results = []
    for nelems in sizes:
        if collective == "all_reduce":
            tensor = torch.randn(nelems, dtype=dtype, device=device)

            def fn():
                dist.all_reduce(tensor, group=group)

        elif collective == "all_gather":
            shard_size = nelems // group_size
            if shard_size < 1:
                continue
            actual_nelems = shard_size * group_size
            inp = torch.randn(shard_size, dtype=dtype, device=device)
            out = torch.empty(actual_nelems, dtype=dtype, device=device)

            def fn():
                dist.all_gather_into_tensor(out, inp, group=group)

            nelems = actual_nelems

        elif collective == "reduce_scatter":
            shard_size = nelems // group_size
            if shard_size < 1:
                continue
            actual_nelems = shard_size * group_size
            inp = torch.randn(actual_nelems, dtype=dtype, device=device)
            out = torch.empty(shard_size, dtype=dtype, device=device)

            def fn():
                dist.reduce_scatter_tensor(out, inp, group=group)

            nelems = actual_nelems

        reset_nccl_tuning(fn, warmup=warmup, group=group)
        s = bench(fn, warmup=warmup, iters=iters)
        results.append((collective, nelems, s))

    return results


def run_collectives_section(args, rank, world_size, local_rank,
                            local_world_size, device, dtype,
                            intra_group, inter_group, num_nodes):
    """Section 1: Collectives decomposed by topology."""
    peaks = get_gpu_peak_bandwidth()
    nvlink_peak = peaks["nvlink_unidir_gbps"]
    json_results = []

    topologies = [
        ("intra_node", intra_group, local_world_size, None),
        ("world", dist.group.WORLD, world_size, None),
    ]

    if num_nodes > 1:
        topologies.insert(1, (
            "inter_node", inter_group, num_nodes, "local_rank==0"))
        topologies.insert(2, (
            "inter_agg", inter_group, num_nodes, "all"))

    collectives = ["all_reduce", "all_gather", "reduce_scatter"]

    for topo_name, group, group_size, mode in topologies:
        skip_this_rank = False
        if mode == "local_rank==0" and local_rank != 0:
            skip_this_rank = True

        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"  Topology: {topo_name}  |  group_size: {group_size}"
                  f"  |  mode: {mode or 'all ranks'}")
            print(f"{'=' * 80}")

        if skip_this_rank:
            dist.barrier()
            continue

        for coll_name in collectives:
            if rank == 0:
                print(f"\n--- {topo_name}: {coll_name} (group_size={group_size}) ---")
                hdr = (f"{'nelems':>12} {'nbytes':>10}"
                       f" | {'p50_us':>10} {'algo_GB/s':>10} {'bus_GB/s':>10}")
                if nvlink_peak > 0 and topo_name == "intra_node":
                    hdr += f" {'eff_%':>7}"
                print(hdr)
                print("-" * len(hdr))

            results = bench_collective(
                coll_name, group, device, dtype, sizes_in_elems(SIZES, dtype),
                group_size, args.warmup, args.iters,
            )

            for coll, nelems, s in results:
                nbytes = nelems * dtype.itemsize
                a_bw = algo_bw(nbytes, s["p50_us"])
                b_bw = bus_bw(a_bw, group_size, coll)

                show_eff = nvlink_peak > 0 and topo_name == "intra_node"
                eff_pct = round(b_bw / nvlink_peak * 100, 1) if show_eff else None

                if rank == 0:
                    line = (
                        f"{nelems:>12} {format_bytes(nbytes):>10}"
                        f" | {s['p50_us']:>8.1f}us {a_bw:>9.1f} {b_bw:>9.1f}"
                    )
                    if eff_pct is not None:
                        line += f" {eff_pct:>6.1f}%"
                    print(line)

                entry = {
                    "section": "collectives",
                    "topology": topo_name,
                    "group_size": group_size,
                    "collective": coll,
                    "nelems": nelems,
                    "nbytes": nbytes,
                    "stats": s,
                    "algo_bw_gbps": round(a_bw, 2),
                    "bus_bw_gbps": round(b_bw, 2),
                }
                if eff_pct is not None:
                    entry["efficiency_pct"] = eff_pct
                json_results.append(entry)

        if mode == "local_rank==0":
            dist.barrier()

    dist.barrier()
    return json_results


# ---- 2D Training ----


class MLPBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class FSDPBenchModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def build_2d_model(hidden, intermediate, num_layers, tp_mesh, dp_mesh,
                   device, dtype):
    """Build model with TP (intra-node) + FSDP2 (inter-node DP)."""
    model = FSDPBenchModel(hidden, intermediate, num_layers).to(device=device)

    for layer in model.layers:
        parallelize_module(layer, tp_mesh, {
            "up": ColwiseParallel(),
            "down": RowwiseParallel(),
        })

    mp_policy = fsdp_mp_policy(dtype)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

    return model


def run_training_section(args, rank, world_size, local_world_size,
                         num_nodes, device, dtype):
    """Section 2: 2D parallelism training (TP intra-node + FSDP2 DP)."""
    json_results = []

    mesh = init_device_mesh(
        "cuda", (num_nodes, local_world_size),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"  2D Parallelism Training: TP={local_world_size} (intra-node)"
              f" + FSDP2 DP={num_nodes} (inter-node)")
        print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}")
        print(f"  Measures: zero_grad + forward + backward + optimizer.step")
        print(f"{'=' * 80}")
        print()
        hdr = (f"{'layers':>6} {'batch':>5} {'params':>10}"
               f" | {'step_us':>10} {'step_ms':>10}"
               f" | {'peak_MB':>10}")
        print(hdr)
        print("-" * len(hdr))

    for num_layers in args.num_layers:
        for batch_size in args.batch_sizes:
            torch.cuda.empty_cache()
            setup_ok = False
            try:
                torch.cuda.reset_peak_memory_stats(device)
                mem_pre = torch.cuda.memory_allocated(device)

                model = build_2d_model(
                    args.hidden, args.intermediate, num_layers,
                    tp_mesh, dp_mesh, device, dtype,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                inp = torch.randn(
                    batch_size, args.hidden, dtype=dtype, device=device,
                )
                total_params = sum(p.numel() for p in model.parameters())
                setup_ok = True
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"{num_layers:>6} {batch_size:>5}  OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"{num_layers:>6} {batch_size:>5}  FAILED: {e}")
                torch.cuda.empty_cache()

            ok = torch.tensor([1.0 if setup_ok else 0.0], device=device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() < 1.0:
                torch.cuda.empty_cache()
                continue

            def step():
                optimizer.zero_grad()
                loss = model(inp).sum()
                loss.backward()
                optimizer.step()

            reset_nccl_tuning(step)
            s = bench(step, warmup=20, iters=args.iters)

            peak_mb = (
                torch.cuda.max_memory_allocated(device) - mem_pre
            ) / 1024**2

            if rank == 0:
                print(
                    f"{num_layers:>6} {batch_size:>5}"
                    f" {total_params:>10,}"
                    f" | {s['p50_us']:>8.0f}us"
                    f" {s['p50_us'] / 1000:>8.1f}ms"
                    f" | {peak_mb:>8.0f} MB"
                )

            json_results.append({
                "section": "2d_training",
                "num_layers": num_layers,
                "batch_size": batch_size,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "tp": local_world_size,
                "dp": num_nodes,
                "total_params": total_params,
                "step": s,
                "peak_mem_mb": round(peak_mb, 1),
            })

            del model, optimizer, inp
            torch.cuda.empty_cache()

    return json_results


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node benchmark: collectives by topology + "
                    "2D parallelism training")
    parser.add_argument("--section", default="all",
                        choices=["all", "collectives", "training"])
    parser.add_argument("--hidden", type=int, default=8192,
                        help="Hidden dimension (default: 8192, Llama-70B)")
    parser.add_argument("--intermediate", type=int, default=28672,
                        help="MLP intermediate dim (default: 28672, Llama-70B)")
    parser.add_argument("--num-layers", type=int, nargs="+", default=[4, 8],
                        help="Layer counts to sweep (training section)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4],
                        help="Batch sizes to sweep (training section)")
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) must be divisible by "
            f"local_world_size ({local_world_size}) — rectangular mesh required"
        )

    num_nodes = world_size // local_world_size

    intra_group, inter_group, _, _ = setup_groups(
        world_size, rank, local_world_size, local_rank,
    )

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Multi-Node Distributed Benchmark")
        print(f"  Nodes: {num_nodes}  |  GPUs/node: {local_world_size}"
              f"  |  World: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(device)}"
              f"  |  dtype: {args.dtype}")
        print(f"{'=' * 80}")

    all_results = []

    if args.section in ("all", "collectives"):
        results = run_collectives_section(
            args, rank, world_size, local_rank, local_world_size,
            device, dtype, intra_group, inter_group, num_nodes,
        )
        all_results.extend(results)

    if args.section in ("all", "training"):
        results = run_training_section(
            args, rank, world_size, local_world_size,
            num_nodes, device, dtype,
        )
        all_results.extend(results)

    if rank == 0:
        print(f"\n{'=' * 80}")
        if num_nodes > 1:
            print(f"  Collectives measured on 4 topologies:")
            print(f"    intra_node:  NVLink within each node")
            print(f"    inter_node:  1 rank/node, raw IB/RoCE link bandwidth")
            print(f"    inter_agg:   all ranks' inter-node groups, aggregate"
                  f" bandwidth under contention")
            print(f"    world:       all {world_size} ranks, hierarchical")
        else:
            print(f"  Single-node mode: inter-node sections skipped")
        print(f"{'=' * 80}\n")

        if args.json:
            output = collect_metadata(
                "multinode",
                world_size=world_size, num_nodes=num_nodes,
                local_world_size=local_world_size,
                dtype=args.dtype, hidden=args.hidden,
                intermediate=args.intermediate,
            )
            output["results"] = all_results
            write_json(args.json, output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
