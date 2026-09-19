"""
End-to-end distributed workload benchmarks.

Measures throughput (samples/sec, tokens/sec) of representative workloads
that chain multiple distributed operations, as real training and inference
pipelines do. Complements the microbenchmarks by capturing overlap,
scheduling, and pipeline effects that isolated op benchmarks miss.

Uses synthetic data, no external model dependencies.

Three sections:

  1. FSDP2 training throughput
     Multi-step FSDP2 training at transformer MLP dimensions.
     Reports samples/sec, step time, and loss validation.

  2. TP inference throughput
     Multi-iteration TP inference at Llama-70B dimensions.
     Decode (batch=32, seq=1) and prefill (batch=1, seq=2048).
     Reports tokens/sec and latency.

  3. FSDP2+PP training throughput (requires >= 4 GPUs)
     Combined pipeline + data parallelism. PP=2, DP=world/2.
     Reports samples/sec and step time.

Usage:
  torchrun --nproc_per_node=8 bench_e2e.py
  torchrun --nproc_per_node=2 bench_e2e.py --section training --json results/e2e.json
  torchrun --nproc_per_node=4 bench_e2e.py --section fsdp2_pp --pp-stages 2
"""

import argparse
import math
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from bench_utils import (
    BENCH_NCCL_TIMEOUT, collect_metadata, fsdp_mp_policy, stats, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# Throughput per dtype; FSDP2 sections keep fp32 master weights
# (fsdp_mp_policy).
DTYPES = ("bf16", "fp16", "fp32")


# ---- Models ----


class MLPBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class TransformerMLPModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# ---- Section 1: FSDP2 training throughput ----


def bench_fsdp2_training(rank, world_size, device, dtype,
                         hidden, intermediate, num_layers,
                         batch_size, warmup, iters):
    model = TransformerMLPModel(hidden, intermediate, num_layers).to(device=device)

    mp_policy = fsdp_mp_policy(dtype)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    inp = torch.randn(batch_size, hidden, dtype=dtype, device=device)
    target = torch.randn(batch_size, hidden, dtype=dtype, device=device)

    total_params = sum(p.numel() for p in model.parameters())

    for _ in range(warmup):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inp), target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    times = []
    first_loss = None
    last_loss = None
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inp), target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)
        if i == 0:
            first_loss = loss.item()
        if i == iters - 1:
            last_loss = loss.item()

    s = stats(times)
    step_s = s["p50_us"] / 1e6
    samples_per_sec = batch_size / step_s if step_s > 0 else 0

    del model, optimizer, inp
    torch.cuda.empty_cache()

    return {
        "section": "fsdp2_training",
        "hidden": hidden,
        "intermediate": intermediate,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "dp": world_size,
        "total_params": total_params,
        "samples_per_sec": round(samples_per_sec, 1),
        "step": s,
        "first_loss": round(first_loss, 4) if first_loss is not None else None,
        "last_loss": round(last_loss, 4) if last_loss is not None else None,
        "loss_decreased": last_loss < first_loss if first_loss is not None else None,
    }


# ---- Section 2: TP inference throughput ----


def bench_tp_inference(rank, world_size, device, dtype,
                       hidden, intermediate, batch, seq_len,
                       num_layers, warmup, iters):
    tp = world_size
    if hidden % tp != 0 or intermediate % tp != 0:
        return None

    shard_h = hidden // tp
    shard_inter = intermediate // tp
    tokens = batch * seq_len

    x = torch.randn(tokens, hidden, dtype=dtype, device=device)

    weights = []
    for _ in range(num_layers):
        W_qkv = torch.randn(hidden, 3 * shard_h, dtype=dtype, device=device) / math.sqrt(hidden)
        W_o = torch.randn(shard_h, hidden, dtype=dtype, device=device) / math.sqrt(shard_h)
        W_gate = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
        W_up = torch.randn(hidden, shard_inter, dtype=dtype, device=device) / math.sqrt(hidden)
        W_down = torch.randn(shard_inter, hidden, dtype=dtype, device=device) / math.sqrt(shard_inter)
        weights.append((W_qkv, W_o, W_gate, W_up, W_down))

    def inference_step():
        h = x
        for W_qkv, W_o, W_gate, W_up, W_down in weights:
            qkv = torch.mm(h, W_qkv)
            o = torch.mm(qkv[:, :shard_h], W_o)
            dist.all_reduce(o)
            gate = torch.mm(o, W_gate)
            up = torch.mm(o, W_up)
            act = torch.nn.functional.silu(gate) * up
            down = torch.mm(act, W_down)
            dist.all_reduce(down)
            h = h + down
        return h

    for _ in range(warmup):
        inference_step()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        inference_step()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)

    s = stats(times)
    step_s = s["p50_us"] / 1e6
    tokens_per_sec = tokens / step_s if step_s > 0 else 0

    return {
        "section": "tp_inference",
        "scenario": "decode" if seq_len == 1 else "prefill",
        "hidden": hidden,
        "intermediate": intermediate,
        "num_layers": num_layers,
        "batch_size": batch,
        "seq_len": seq_len,
        "tokens": tokens,
        "tp": tp,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "step": s,
    }


# ---- Section 3: FSDP2+PP training throughput ----


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


def bench_fsdp2_pp_training(rank, world_size, device, dtype,
                            pp_stages, hidden, intermediate, num_layers,
                            batch_size, num_microbatches, warmup, iters):
    from torch.distributed.device_mesh import init_device_mesh

    if world_size % pp_stages != 0:
        return None
    dp_size = world_size // pp_stages
    if dp_size < 2:
        return None

    layers_per_stage = num_layers // pp_stages
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

    stage_model = TransformerMLPModel(hidden, intermediate, layers_per_stage).to(
        device=device)

    mp_policy = fsdp_mp_policy(dtype)
    for layer in stage_model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(stage_model, mesh=dp_mesh, mp_policy=mp_policy)

    optimizer = torch.optim.Adam(stage_model.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in stage_model.parameters())

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

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)

    s = stats(times)
    step_s = s["p50_us"] / 1e6
    samples_per_sec = batch_size / step_s if step_s > 0 else 0

    del stage_model, optimizer, inp
    torch.cuda.empty_cache()

    bubble_pct = round(
        (pp_stages - 1) / (pp_stages - 1 + num_microbatches) * 100, 1)

    return {
        "section": "fsdp2_pp_training",
        "pp_stages": pp_stages,
        "dp_size": dp_size,
        "hidden": hidden,
        "intermediate": intermediate,
        "num_layers": num_layers,
        "layers_per_stage": layers_per_stage,
        "batch_size": batch_size,
        "num_microbatches": num_microbatches,
        "bubble_pct": bubble_pct,
        "total_params_per_stage": total_params,
        "samples_per_sec": round(samples_per_sec, 1),
        "step": s,
    }


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end distributed workload benchmarks: "
                    "FSDP2 training, TP inference, FSDP2+PP training")
    parser.add_argument("--section", default="all",
                        choices=["all", "training", "inference", "fsdp2_pp"])
    parser.add_argument("--pp-stages", type=int, default=2,
                        help="PP stages for FSDP2+PP section (default: 2)")
    parser.add_argument("--hidden", type=int, default=8192,
                        help="Hidden dimension (default: 8192, Llama-70B)")
    parser.add_argument("--intermediate", type=int, default=28672,
                        help="MLP intermediate dim (default: 28672, Llama-70B)")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-microbatches", type=int, default=2)
    parser.add_argument("--dtype", default=DTYPES[0],
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=10)
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

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"End-to-End Distributed Workload Benchmarks")
        print(f"  World: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}"
              f"  |  dtype: {args.dtype}")
        print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}"
              f"  |  Layers: {args.num_layers}")
        print(f"{'=' * 70}")

    json_results = []

    # ---- Section 1: FSDP2 training ----

    if args.section in ("all", "training"):
        if rank == 0:
            print(f"\n--- FSDP2 Training Throughput (DP={world_size}) ---")

        ok = torch.tensor([1.0], device=device)
        r = None
        try:
            r = bench_fsdp2_training(
                rank, world_size, device, dtype,
                args.hidden, args.intermediate, args.num_layers,
                args.batch_size, args.warmup, args.iters,
            )
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print("  OOM — skipping")
            ok.zero_()
            torch.cuda.empty_cache()
        except Exception as e:
            if rank == 0:
                print(f"  FAILED: {e}")
            ok.zero_()
            torch.cuda.empty_cache()

        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() >= 1.0 and r is not None:
            if rank == 0:
                print(f"  Step p50:       {r['step']['p50_us'] / 1000:.1f} ms")
                print(f"  Samples/sec:    {r['samples_per_sec']:.1f}")
                print(f"  Loss:           {r['first_loss']:.4f} → {r['last_loss']:.4f}"
                      f"  ({'decreased' if r['loss_decreased'] else 'DID NOT decrease'})")
                print(f"  Params:         {r['total_params']:,}")
            json_results.append(r)

    # ---- Section 2: TP inference ----

    if args.section in ("all", "inference"):
        scenarios = [
            ("decode", 32, 1),
            ("prefill", 1, 2048),
        ]

        if rank == 0:
            print(f"\n--- TP Inference Throughput (TP={world_size},"
                  f" hidden={args.hidden}) ---")

        for name, batch, seq_len in scenarios:
            ok = torch.tensor([1.0], device=device)
            r = None
            try:
                r = bench_tp_inference(
                    rank, world_size, device, dtype,
                    args.hidden, args.intermediate, batch, seq_len,
                    args.num_layers, args.warmup, args.iters,
                )
                if r is None:
                    if rank == 0:
                        print(f"  {name}: skipped (dims not divisible by TP={world_size})")
                    continue
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"  {name}: OOM — skipping")
                ok.zero_()
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"  {name}: FAILED: {e}")
                ok.zero_()
                torch.cuda.empty_cache()

            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() >= 1.0 and r is not None:
                if rank == 0:
                    print(f"  {name}:")
                    print(f"    Latency p50:  {r['step']['p50_us'] / 1000:.2f} ms")
                    print(f"    Tokens/sec:   {r['tokens_per_sec']:.1f}")
                json_results.append(r)

    # ---- Section 3: FSDP2+PP ----

    if args.section in ("all", "fsdp2_pp"):
        pp_stages = args.pp_stages

        if world_size < pp_stages * 2:
            if rank == 0:
                print(f"\n--- FSDP2+PP: skipped (need >= {pp_stages * 2} GPUs) ---")
        else:
            dp_size = world_size // pp_stages
            if rank == 0:
                print(f"\n--- FSDP2+PP Training Throughput "
                      f"(PP={pp_stages}, DP={dp_size}) ---")

            ok = torch.tensor([1.0], device=device)
            r = None
            try:
                r = bench_fsdp2_pp_training(
                    rank, world_size, device, dtype,
                    pp_stages, args.hidden, args.intermediate,
                    args.num_layers, args.batch_size,
                    args.num_microbatches, args.warmup, args.iters,
                )
                if r is None:
                    ok.zero_()
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print("  OOM — skipping")
                ok.zero_()
                torch.cuda.empty_cache()
            except Exception as e:
                if rank == 0:
                    print(f"  FAILED: {e}")
                ok.zero_()
                torch.cuda.empty_cache()

            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() >= 1.0 and r is not None:
                if rank == 0:
                    print(f"  Step p50:       {r['step']['p50_us'] / 1000:.1f} ms")
                    print(f"  Samples/sec:    {r['samples_per_sec']:.1f}")
                    print(f"  Bubble (GPipe): {r['bubble_pct']:.1f}%")
                    print(f"  Params/stage:   {r['total_params_per_stage']:,}")
                json_results.append(r)

    # ---- JSON output ----

    if rank == 0:
        print(f"\n{'=' * 70}\n")

        if args.json:
            output = collect_metadata(
                "e2e", dtype=args.dtype,
                hidden=args.hidden, intermediate=args.intermediate,
                num_layers=args.num_layers, batch_size=args.batch_size,
            )
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
