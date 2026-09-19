"""
Benchmark pipeline parallelism: P2P communication and pipeline training steps.

Four sections:

  1. P2P Send/Recv sweep
     Bidirectional point-to-point between adjacent ranks at 11 message sizes.
     Measures NVLink P2P (single node) or IB P2P (multi-node) latency and
     bandwidth.

  2. Pipeline training step (PP only)
     GPipe-style pipeline: each GPU is one stage holding a subset of layers.
     All microbatches forward through the pipeline, then backward in reverse.
     Reports theoretical pipeline bubble fraction.

  3. FSDP2 + PP combined
     2D parallelism: PP stages × DP replicas = world_size. Each stage's model
     is FSDP2-sharded across its DP group. Reports bubble fraction.

  4. All-pairs P2P latency matrix
     Measures P2P between every GPU pair using tournament scheduling.
     Detects topology asymmetries (degraded NVLink, PCIe bottleneck).
     Reports NxN matrix at two sizes (latency-dominated and BW-dominated).

Usage:
  torchrun --nproc_per_node=8 bench_pipeline_parallel.py
  torchrun --nproc_per_node=2 bench_pipeline_parallel.py --section p2p
  torchrun --nproc_per_node=4 bench_pipeline_parallel.py --section fsdp2_pp --pp-stages 2
  torchrun --nproc_per_node=8 bench_pipeline_parallel.py --section all_pairs
"""

import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, fsdp_mp_policy,
    get_gpu_peak_bandwidth, reset_nccl_tuning, sizes_in_elems, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default. P2P
# in bytes (SIZES); training sections use fp32 master weights with dtype
# compute.
DTYPES = ("bf16", "fp16", "fp32")


# Message sizes in bytes, 1 KB .. 1 GB.
SIZES = [1 << n for n in range(10, 31, 2)]


def format_bytes(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def p2p_bw(nbytes, p50_us):
    if p50_us <= 0:
        return 0.0
    return nbytes / (p50_us * 1e-6) / 1e9


# ---- Model ----


class MLPBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class StageModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# ---- Section 1: P2P Send/Recv sweep ----


def bench_p2p_sweep(rank, world_size, device, dtype, warmup, iters):
    if world_size < 2:
        if rank == 0:
            print("  P2P requires at least 2 ranks, skipping")
        return []

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    results = []
    for nelems in sizes_in_elems(SIZES, dtype):
        send_buf = torch.randn(nelems, dtype=dtype, device=device)
        recv_buf = torch.empty(nelems, dtype=dtype, device=device)

        def fn():
            ops = [
                dist.P2POp(dist.isend, send_buf, next_rank),
                dist.P2POp(dist.irecv, recv_buf, prev_rank),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        reset_nccl_tuning(fn, warmup=warmup)
        s = bench(fn, warmup=warmup, iters=iters)
        results.append((nelems, s))

    return results


# ---- Pipeline step ----


def pipeline_step(stage_model, optimizer, inp_or_none, target_or_none,
                  pp_rank, pp_stages, prev_rank, next_rank,
                  num_microbatches, mb_size, hidden, device, dtype):
    saved = []
    optimizer.zero_grad()

    for mb in range(num_microbatches):
        if pp_rank == 0:
            x = inp_or_none[mb * mb_size : (mb + 1) * mb_size]
        else:
            x = torch.empty(mb_size, hidden, dtype=dtype, device=device)
            dist.recv(x, src=prev_rank)

        x = x.detach().requires_grad_(True)
        # Master weights are fp32; compute in dtype so activations and the
        # P2P buffers match. Under FSDP2 the mp_policy already casts params
        # and autocast is a no-op.
        with torch.autocast("cuda", dtype=dtype,
                            enabled=dtype != torch.float32):
            out = stage_model(x)
        saved.append((x, out))

        if next_rank is not None:
            dist.send(out.detach(), dst=next_rank)

    for mb in reversed(range(num_microbatches)):
        x, out = saved[mb]
        if next_rank is None:
            tgt = target_or_none[mb * mb_size : (mb + 1) * mb_size]
            loss = torch.nn.functional.mse_loss(out, tgt) / num_microbatches
            loss.backward()
        else:
            grad = torch.empty_like(out)
            dist.recv(grad, src=next_rank)
            out.backward(grad)

        if prev_rank is not None:
            dist.send(x.grad, dst=prev_rank)

    optimizer.step()


# ---- Section 2: Pipeline training (PP only) ----


def bench_pipeline(rank, world_size, device, dtype,
                   hidden, intermediate, total_layers,
                   batch_size, num_microbatches, warmup, iters):
    pp_stages = world_size
    pp_rank = rank
    layers_per_stage = total_layers // pp_stages
    if layers_per_stage < 1:
        return None

    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < pp_stages - 1 else None

    mb_size = batch_size // num_microbatches
    if mb_size < 1:
        return None

    stage_model = StageModel(hidden, intermediate, layers_per_stage).to(
        device=device)
    optimizer = torch.optim.Adam(stage_model.parameters(), lr=1e-4)

    inp = None
    if pp_rank == 0:
        inp = torch.randn(batch_size, hidden, dtype=dtype, device=device)

    target = None
    if next_rank is None:
        target = torch.randn(batch_size, hidden, dtype=dtype, device=device)

    def step():
        pipeline_step(stage_model, optimizer, inp, target,
                      pp_rank, pp_stages, prev_rank, next_rank,
                      num_microbatches, mb_size, hidden, device, dtype)

    reset_nccl_tuning(step, warmup=warmup)
    s = bench(step, warmup=warmup, iters=iters)

    total_params = sum(p.numel() for p in stage_model.parameters())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    bubble_pct = round(
        (pp_stages - 1) / (pp_stages - 1 + num_microbatches) * 100, 1)

    del stage_model, optimizer, inp
    torch.cuda.empty_cache()

    return {
        "pp_stages": pp_stages,
        "layers_per_stage": layers_per_stage,
        "total_params_per_stage": total_params,
        "num_microbatches": num_microbatches,
        "bubble_pct": bubble_pct,
        "step": s,
        "peak_mem_mb": round(peak_mb, 1),
    }


# ---- Section 3: FSDP2 + PP combined ----


def bench_fsdp2_pp(rank, world_size, device, dtype,
                   pp_stages, hidden, intermediate, total_layers,
                   batch_size, num_microbatches, warmup, iters):
    if world_size % pp_stages != 0:
        return None
    dp_size = world_size // pp_stages
    if dp_size < 2:
        return None

    layers_per_stage = total_layers // pp_stages
    if layers_per_stage < 1:
        return None

    mb_size = batch_size // num_microbatches
    if mb_size < 1:
        return None

    mesh_2d = init_device_mesh(
        "cuda", (pp_stages, dp_size), mesh_dim_names=("pp", "dp"))
    dp_mesh = mesh_2d["dp"]

    pp_rank = mesh_2d.get_local_rank("pp")
    dp_rank = mesh_2d.get_local_rank("dp")

    pp_col = mesh_2d.mesh[:, dp_rank].tolist()
    prev_rank = pp_col[pp_rank - 1] if pp_rank > 0 else None
    next_rank = pp_col[pp_rank + 1] if pp_rank < pp_stages - 1 else None

    stage_model = StageModel(hidden, intermediate, layers_per_stage).to(
        device=device)

    mp_policy = fsdp_mp_policy(dtype)
    for layer in stage_model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(stage_model, mesh=dp_mesh, mp_policy=mp_policy)

    optimizer = torch.optim.Adam(stage_model.parameters(), lr=1e-4)

    inp = None
    if pp_rank == 0:
        inp = torch.randn(batch_size, hidden, dtype=dtype, device=device)

    target = None
    if next_rank is None:
        target = torch.randn(batch_size, hidden, dtype=dtype, device=device)

    def step():
        pipeline_step(stage_model, optimizer, inp, target,
                      pp_rank, pp_stages, prev_rank, next_rank,
                      num_microbatches, mb_size, hidden, device, dtype)

    reset_nccl_tuning(step, warmup=warmup)
    s = bench(step, warmup=warmup, iters=iters)

    total_params = sum(p.numel() for p in stage_model.parameters())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    bubble_pct = round(
        (pp_stages - 1) / (pp_stages - 1 + num_microbatches) * 100, 1)

    del stage_model, optimizer, inp
    torch.cuda.empty_cache()

    return {
        "pp_stages": pp_stages,
        "dp_size": dp_size,
        "layers_per_stage": layers_per_stage,
        "total_params_per_stage": total_params,
        "num_microbatches": num_microbatches,
        "bubble_pct": bubble_pct,
        "step": s,
        "peak_mem_mb": round(peak_mb, 1),
    }


# ---- Section 4: All-pairs P2P latency matrix ----


ALL_PAIRS_SIZES = [16 << 10, 64 << 20]  # bytes


def bench_all_pairs_p2p(rank, world_size, device, dtype, warmup, iters):
    """Measure P2P between all GPU pairs using tournament scheduling.

    In round k (k=1..N-1), rank i sends to (i+k)%N and receives from
    (i-k+N)%N. All ranks participate simultaneously. Completes all
    N*(N-1) directed pairs in N-1 rounds.

    Returns list of result dicts (one per message size) with the full
    NxN latency/bandwidth matrix.
    """
    if world_size < 2:
        return []

    json_results = []

    for nelems in sizes_in_elems(ALL_PAIRS_SIZES, dtype):
        nbytes = nelems * dtype.itemsize

        my_times = torch.zeros(world_size, device=device)

        for k in range(1, world_size):
            dst = (rank + k) % world_size
            src = (rank - k + world_size) % world_size

            send_buf = torch.randn(nelems, dtype=dtype, device=device)
            recv_buf = torch.empty(nelems, dtype=dtype, device=device)

            def fn():
                ops = [
                    dist.P2POp(dist.isend, send_buf, dst),
                    dist.P2POp(dist.irecv, recv_buf, src),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

            reset_nccl_tuning(fn, warmup=max(5, warmup // 4))
            s = bench(fn, warmup=max(5, warmup // 4), iters=max(20, iters // 4))
            my_times[dst] = s["p50_us"]

        all_times = [torch.zeros(world_size, device=device)
                     for _ in range(world_size)]
        dist.all_gather(all_times, my_times)

        pairs = []
        all_p50 = []
        all_bw = []
        for src_r in range(world_size):
            for dst_r in range(world_size):
                if src_r == dst_r:
                    continue
                p50 = all_times[src_r][dst_r].item()
                bw = p2p_bw(nbytes, p50)
                pairs.append({
                    "src": src_r, "dst": dst_r,
                    "p50_us": round(p50, 1),
                    "bw_gbps": round(bw, 1),
                })
                all_p50.append(p50)
                all_bw.append(bw)

        entry = {
            "section": "all_pairs_p2p",
            "nelems": nelems,
            "nbytes": nbytes,
            "pairs": pairs,
        }
        if all_p50:
            mn, mx = min(all_p50), max(all_p50)
            entry["latency_max_min_ratio"] = round(mx / mn, 2) if mn > 0 else None
            mn_bw, mx_bw = min(all_bw), max(all_bw)
            entry["bw_max_min_ratio"] = round(mx_bw / mn_bw, 2) if mn_bw > 0 else None

        json_results.append(entry)

    return json_results


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline parallelism benchmark: P2P sweep, pipeline "
                    "training step, and FSDP2+PP combined")
    parser.add_argument("--section", default="all",
                        choices=["all", "p2p", "pipeline", "fsdp2_pp",
                                 "all_pairs"])
    parser.add_argument("--pp-stages", type=int, default=0,
                        help="PP stages (0 = world_size for p2p/pipeline, "
                             "2 for fsdp2_pp)")
    parser.add_argument("--num-microbatches", type=int, nargs="+",
                        default=[1, 4])
    parser.add_argument("--hidden", type=int, default=4096,
                        help="Hidden dimension (default: 4096, Llama-8B)")
    parser.add_argument("--intermediate", type=int, default=14336,
                        help="MLP intermediate dim (default: 14336, Llama-8B)")
    parser.add_argument("--num-layers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 16])
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

    peaks = get_gpu_peak_bandwidth()
    nvlink_peak = peaks["nvlink_unidir_gbps"]

    json_results = []

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Pipeline Parallelism Benchmark")
        print(f"  World: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}"
              f"  |  dtype: {args.dtype}")
        if nvlink_peak > 0:
            print(f"  NVLink unidir peak: {nvlink_peak} GB/s")
        print(f"{'=' * 80}")

    # ---- Section 1: P2P sweep ----

    if args.section in ("all", "p2p"):
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"  P2P Send/Recv Sweep (bidirectional ring)")
            print(f"{'=' * 80}")
            hdr = (f"{'nelems':>12} {'nbytes':>10}"
                   f" | {'p50_us':>10} {'GB/s':>10}")
            if nvlink_peak > 0:
                hdr += f" {'eff_%':>7}"
            print(hdr)
            print("-" * len(hdr))

        results = bench_p2p_sweep(rank, world_size, device, dtype,
                                  args.warmup, args.iters)

        for nelems, s in results:
            nbytes = nelems * dtype.itemsize
            bw = p2p_bw(nbytes, s["p50_us"])
            eff_pct = round(bw / nvlink_peak * 100, 1) if nvlink_peak > 0 else None

            if rank == 0:
                line = (
                    f"{nelems:>12} {format_bytes(nbytes):>10}"
                    f" | {s['p50_us']:>8.1f}us {bw:>9.1f}"
                )
                if eff_pct is not None:
                    line += f" {eff_pct:>6.1f}%"
                print(line)

            entry = {
                "section": "p2p",
                "nelems": nelems,
                "nbytes": nbytes,
                "stats": s,
                "bw_gbps": round(bw, 2),
            }
            if eff_pct is not None:
                entry["efficiency_pct"] = eff_pct
            json_results.append(entry)

    # ---- Section 2: Pipeline training ----

    if args.section in ("all", "pipeline"):
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"  Pipeline Training (GPipe, PP={world_size})")
            print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}")
            print(f"{'=' * 80}")
            print()
            hdr = (f"{'layers':>6} {'batch':>5} {'mbs':>4} {'params/stg':>12}"
                   f" | {'step_us':>10} {'step_ms':>10}"
                   f" | {'bubble%':>8} {'peak_MB':>10}")
            print(hdr)
            print("-" * len(hdr))

        for num_layers in args.num_layers:
            for batch_size in args.batch_sizes:
                for num_mbs in args.num_microbatches:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    ok = torch.tensor([1.0], device=device)
                    result = None
                    try:
                        result = bench_pipeline(
                            rank, world_size, device, dtype,
                            args.hidden, args.intermediate, num_layers,
                            batch_size, num_mbs, args.warmup, args.iters,
                        )
                    except torch.cuda.OutOfMemoryError:
                        if rank == 0:
                            print(f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}  OOM")
                        ok.zero_()
                        torch.cuda.empty_cache()
                    except Exception as e:
                        if rank == 0:
                            print(f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}  FAILED: {e}")
                        ok.zero_()
                        torch.cuda.empty_cache()

                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    if ok.item() < 1.0 or result is None:
                        continue

                    if rank == 0:
                        print(
                            f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}"
                            f" {result['total_params_per_stage']:>12,}"
                            f" | {result['step']['p50_us']:>8.0f}us"
                            f" {result['step']['p50_us'] / 1000:>8.1f}ms"
                            f" | {result['bubble_pct']:>6.1f}%"
                            f" {result['peak_mem_mb']:>8.0f} MB"
                        )

                    json_results.append({
                        "section": "pipeline",
                        "num_layers": num_layers,
                        "batch_size": batch_size,
                        **result,
                    })

    # ---- Section 3: FSDP2 + PP ----

    if args.section in ("all", "fsdp2_pp"):
        pp_stages = args.pp_stages if args.pp_stages > 0 else 2

        if world_size < pp_stages * 2:
            if rank == 0:
                print(f"\n  FSDP2+PP requires at least {pp_stages * 2} ranks "
                      f"(PP={pp_stages} × DP>=2), skipping")
        else:
            dp_size = world_size // pp_stages

            if rank == 0:
                print(f"\n{'=' * 80}")
                print(f"  FSDP2 + PP (PP={pp_stages}, DP={dp_size})")
                print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}")
                print(f"{'=' * 80}")
                print()
                hdr = (f"{'layers':>6} {'batch':>5} {'mbs':>4} {'params/stg':>12}"
                       f" | {'step_us':>10} {'step_ms':>10}"
                       f" | {'bubble%':>8} {'peak_MB':>10}")
                print(hdr)
                print("-" * len(hdr))

            for num_layers in args.num_layers:
                for batch_size in args.batch_sizes:
                    for num_mbs in args.num_microbatches:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)

                        ok = torch.tensor([1.0], device=device)
                        result = None
                        try:
                            result = bench_fsdp2_pp(
                                rank, world_size, device, dtype,
                                pp_stages, args.hidden, args.intermediate,
                                num_layers, batch_size, num_mbs,
                                args.warmup, args.iters,
                            )
                            if result is None:
                                ok.zero_()
                        except torch.cuda.OutOfMemoryError:
                            if rank == 0:
                                print(f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}  OOM")
                            ok.zero_()
                            torch.cuda.empty_cache()
                        except Exception as e:
                            if rank == 0:
                                print(f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}  FAILED: {e}")
                            ok.zero_()
                            torch.cuda.empty_cache()

                        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                        if ok.item() < 1.0:
                            torch.cuda.empty_cache()
                            continue

                        if rank == 0:
                            print(
                                f"{num_layers:>6} {batch_size:>5} {num_mbs:>4}"
                                f" {result['total_params_per_stage']:>12,}"
                                f" | {result['step']['p50_us']:>8.0f}us"
                                f" {result['step']['p50_us'] / 1000:>8.1f}ms"
                                f" | {result['bubble_pct']:>6.1f}%"
                                f" {result['peak_mem_mb']:>8.0f} MB"
                            )

                        json_results.append({
                            "section": "fsdp2_pp",
                            "num_layers": num_layers,
                            "batch_size": batch_size,
                            **result,
                        })

    # ---- Section 4: All-pairs P2P ----

    if args.section in ("all", "all_pairs"):
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"  All-Pairs P2P Latency Matrix (tournament scheduling)")
            print(f"  Bidirectional: send to one partner, recv from another")
            if nvlink_peak > 0:
                print(f"  NVLink unidir peak: {nvlink_peak} GB/s")
            print(f"{'=' * 80}")

        ap_results = bench_all_pairs_p2p(
            rank, world_size, device, dtype,
            args.warmup, args.iters)

        for entry in ap_results:
            nbytes = entry["nbytes"]
            label = ("latency-dominated" if nbytes < 1 << 20
                     else "bandwidth-dominated")

            if rank == 0:
                print(f"\n--- {format_bytes(nbytes)} ({label}) ---")
                hdr = "     " + "".join(f"{'GPU' + str(d):>9}" for d in range(world_size))
                print(hdr)

                for src_r in range(world_size):
                    row = f"GPU{src_r} "
                    for dst_r in range(world_size):
                        if src_r == dst_r:
                            row += f"{'---':>9}"
                        else:
                            pair = next(
                                p for p in entry["pairs"]
                                if p["src"] == src_r and p["dst"] == dst_r)
                            if nbytes < 1 << 20:
                                row += f"{pair['p50_us']:>7.1f}us"
                            else:
                                row += f"{pair['bw_gbps']:>6.1f}G/s"
                    print(row)

                ratio = entry.get("latency_max_min_ratio")
                bw_ratio = entry.get("bw_max_min_ratio")
                if ratio is not None:
                    sym = "symmetric" if ratio < 1.15 else "ASYMMETRIC"
                    print(f"  Latency max/min ratio: {ratio:.2f} ({sym})")
                if bw_ratio is not None and nbytes >= 1 << 20:
                    sym = "symmetric" if bw_ratio < 1.15 else "ASYMMETRIC"
                    print(f"  BW max/min ratio: {bw_ratio:.2f} ({sym})")

        json_results.extend(ap_results)

    # ---- JSON output ----

    mem_after = torch.cuda.memory_allocated(device)

    if rank == 0 and args.json:
        output = collect_metadata(
            "pipeline_parallel",
            dtype=args.dtype,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )
        output["gpu_mem_delta_bytes"] = mem_after - mem_before
        output["gpu_mem_peak_bytes"] = torch.cuda.max_memory_allocated(device)
        output["results"] = json_results
        write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
