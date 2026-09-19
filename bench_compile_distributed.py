"""
Benchmark: torch.compile + distributed — compiled vs eager.

Measures the performance impact of torch.compile (Inductor) on distributed
operations:

  1. FSDP2 training step: fully_shard() model, eager vs compiled.
     Tests Inductor's compute kernel fusion within FSDP2's communication
     overlap schedule. Compiled autograd graph-breaks at FSDP hooks, so
     the benefit comes from fusing kernels within each layer.

  2. TP-style inference: stacked (Linear -> AllReduce -> SiLU) layers,
     eager vs compiled. Tests Inductor's functionalization of in-place
     collectives and potential comm/compute reordering across layers.

torch.compile + distributed is an active development area. This benchmark
tracks whether compile helps, hurts, or breaks across PyTorch releases.

No NVSwitch or symmetric memory required.

Usage:
  torchrun --nproc_per_node=2 bench_compile_distributed.py
  torchrun --nproc_per_node=8 bench_compile_distributed.py --json results/compile.json
"""

import argparse

import torch
try:
    import torch._dynamo
    import torch._inductor.config
except (ImportError, ModuleNotFoundError):
    raise SystemExit(
        "bench_compile_distributed requires torch._dynamo and torch._inductor "
        "(not available in this PyTorch build)"
    )
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from bench_utils import (
    BENCH_NCCL_TIMEOUT, bench, collect_metadata, fsdp_mp_policy,
    reset_nccl_tuning, verify_close, write_json,
)

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# Inductor codegen differs per dtype; FSDP2 params are fp32 master weights
# under mp_policy.
DTYPES = ("bf16", "fp16", "fp32")


def verify_compiled_output(eager_model, compiled_model, inp):
    """Check that compiled model produces the same output as eager."""
    with torch.no_grad():
        ref = eager_model(inp)
        out = compiled_model(inp)
    verify_close("compiled_vs_eager", ref, out)


# ---- Models ----


class MLPBlock(nn.Module):
    """MLP block matching Llama gate_proj + down_proj pattern."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class FSDPModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class TPLayer(nn.Module):
    """Simulates TP column-parallel: Linear -> AllReduce -> SiLU."""

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        x = self.linear(x)
        dist.all_reduce(x)
        return torch.nn.functional.silu(x)


class TPModel(nn.Module):
    def __init__(self, hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [TPLayer(hidden) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_fsdp_model(hidden, intermediate, num_layers, device, dtype):
    model = FSDPModel(hidden, intermediate, num_layers).to(device=device)
    mp_policy = fsdp_mp_policy(dtype)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="torch.compile + distributed benchmark"
    )
    parser.add_argument("--hidden", type=int, default=4096,
                        help="Hidden dimension (default: 4096, Llama-7B)")
    parser.add_argument("--intermediate", type=int, default=11008,
                        help="MLP intermediate dim (default: 11008, Llama-7B)")
    parser.add_argument("--num-layers", type=int, nargs="+", default=[4, 8],
                        help="Layer counts to sweep")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for TP section")
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

    json_results = []

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"torch.compile + Distributed Benchmark")
        print(f"  World size: {world_size}  |  GPU: "
              f"{torch.cuda.get_device_name(device)}  |  dtype: {args.dtype}")
        print(f"{'=' * 95}")

    # Disable cudagraphs — incompatible with FSDP and async collectives
    torch._inductor.config.triton.cudagraphs = False

    # ---- Section 1: FSDP2 Training Step ----

    if rank == 0:
        print(f"\n--- FSDP2 Training Step: compiled vs eager ---")
        print(f"  Hidden: {args.hidden}  |  Intermediate: {args.intermediate}"
              f"  |  Batch: {args.batch_size}")
        hdr = (f"{'layers':>6}"
               f" | {'eager_us':>10} {'compiled_us':>12} {'speedup':>10}"
               f" | {'status':>8}")
        print(hdr)
        print("-" * len(hdr))

    for num_layers in args.num_layers:
        eager_s = None
        compiled_s = None

        # --- Eager ---
        try:
            torch._dynamo.reset()
            torch.cuda.empty_cache()

            model_e = build_fsdp_model(
                args.hidden, args.intermediate, num_layers, device, dtype
            )
            opt_e = torch.optim.Adam(model_e.parameters(), lr=1e-4)
            inp_e = torch.randn(
                args.batch_size, args.hidden, dtype=dtype, device=device
            )

            def eager_step():
                opt_e.zero_grad()
                model_e(inp_e).sum().backward()
                opt_e.step()

            reset_nccl_tuning(eager_step)
            eager_s = bench(eager_step, warmup=args.warmup, iters=args.iters)

            del model_e, opt_e, inp_e
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"{num_layers:>6} | eager FAILED: {e}")
            torch.cuda.empty_cache()
            json_results.append({
                "section": "fsdp2_training", "num_layers": num_layers,
                "batch_size": args.batch_size, "error": f"eager: {e}",
            })
            continue

        # --- Compiled ---
        try:
            torch._dynamo.reset()
            torch.cuda.empty_cache()

            model_c = build_fsdp_model(
                args.hidden, args.intermediate, num_layers, device, dtype
            )
            compiled_model_c = torch.compile(model_c, backend="inductor")
            opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-4)
            inp_c = torch.randn(
                args.batch_size, args.hidden, dtype=dtype, device=device
            )

            def compiled_step():
                opt_c.zero_grad()
                compiled_model_c(inp_c).sum().backward()
                opt_c.step()

            # Extra warmup to absorb compilation latency
            reset_nccl_tuning(compiled_step, warmup=30)
            compiled_s = bench(
                compiled_step, warmup=args.warmup, iters=args.iters
            )

            del model_c, compiled_model_c, opt_c, inp_c
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"{num_layers:>6}"
                      f" | {eager_s['p50_us']:>8.0f}us {'FAILED':>12}"
                      f" {'---':>10} | compile: {e}")
            torch.cuda.empty_cache()
            json_results.append({
                "section": "fsdp2_training", "num_layers": num_layers,
                "batch_size": args.batch_size,
                "eager": eager_s, "error": f"compile: {e}",
            })
            continue

        speedup = (eager_s["p50_us"] / max(compiled_s["p50_us"], 0.1) - 1) * 100

        if rank == 0:
            print(
                f"{num_layers:>6}"
                f" | {eager_s['p50_us']:>8.0f}us"
                f" {compiled_s['p50_us']:>10.0f}us"
                f" {speedup:>+8.1f}%"
                f" |       OK"
            )

        json_results.append({
            "section": "fsdp2_training",
            "num_layers": num_layers,
            "batch_size": args.batch_size,
            "eager": eager_s,
            "compiled": compiled_s,
            "speedup_pct": round(speedup, 1),
        })

    # ---- Section 2: TP Inference (forward only) ----

    if rank == 0:
        print(f"\n--- TP Inference (Linear -> AllReduce -> SiLU): "
              f"compiled vs eager ---")
        print(f"  Hidden: {args.hidden}  |  Seq len: {args.seq_len}")
        hdr = (f"{'layers':>6}"
               f" | {'eager_us':>10} {'compiled_us':>12} {'speedup':>10}"
               f" | {'status':>8}")
        print(hdr)
        print("-" * len(hdr))

    for num_layers in args.num_layers:
        eager_s = None
        compiled_s = None

        # --- Eager ---
        try:
            torch._dynamo.reset()
            torch.cuda.empty_cache()

            tp_model_e = TPModel(args.hidden, num_layers).to(
                device=device, dtype=dtype
            )
            tp_inp_e = torch.randn(
                args.seq_len, args.hidden, dtype=dtype, device=device
            )

            def eager_fwd():
                with torch.no_grad():
                    tp_model_e(tp_inp_e)

            reset_nccl_tuning(eager_fwd)
            eager_s = bench(eager_fwd, warmup=args.warmup, iters=args.iters)

            del tp_model_e, tp_inp_e
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"{num_layers:>6} | eager FAILED: {e}")
            torch.cuda.empty_cache()
            json_results.append({
                "section": "tp_inference", "num_layers": num_layers,
                "seq_len": args.seq_len, "error": f"eager: {e}",
            })
            continue

        # --- Compiled ---
        try:
            torch._dynamo.reset()
            torch.cuda.empty_cache()

            tp_model_c = TPModel(args.hidden, num_layers).to(
                device=device, dtype=dtype
            )
            compiled_tp = torch.compile(tp_model_c, backend="inductor")
            tp_inp_c = torch.randn(
                args.seq_len, args.hidden, dtype=dtype, device=device
            )

            verify_compiled_output(tp_model_c, compiled_tp, tp_inp_c)

            def compiled_fwd():
                with torch.no_grad():
                    compiled_tp(tp_inp_c)

            reset_nccl_tuning(compiled_fwd, warmup=30)
            compiled_s = bench(
                compiled_fwd, warmup=args.warmup, iters=args.iters
            )

            del tp_model_c, compiled_tp, tp_inp_c
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"{num_layers:>6}"
                      f" | {eager_s['p50_us']:>8.0f}us {'FAILED':>12}"
                      f" {'---':>10} | compile: {e}")
            torch.cuda.empty_cache()
            json_results.append({
                "section": "tp_inference", "num_layers": num_layers,
                "seq_len": args.seq_len,
                "eager": eager_s, "error": f"compile: {e}",
            })
            continue

        speedup = (eager_s["p50_us"] / max(compiled_s["p50_us"], 0.1) - 1) * 100

        if rank == 0:
            print(
                f"{num_layers:>6}"
                f" | {eager_s['p50_us']:>8.0f}us"
                f" {compiled_s['p50_us']:>10.0f}us"
                f" {speedup:>+8.1f}%"
                f" |       OK"
            )

        json_results.append({
            "section": "tp_inference",
            "num_layers": num_layers,
            "seq_len": args.seq_len,
            "eager": eager_s,
            "compiled": compiled_s,
            "speedup_pct": round(speedup, 1),
        })

    # ---- Summary ----

    if rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  FSDP2: compiled autograd graph-breaks at FSDP hooks —")
        print(f"  benefit comes from Inductor fusing compute kernels within "
              f"each layer")
        print(f"  TP: Inductor functionalizes in-place collectives — benefit")
        print(f"  comes from kernel fusion and comm/compute reordering")
        print(f"  cudagraphs disabled (incompatible with async collectives)")
        print(f"{'=' * 95}\n")

        if args.json:
            output = collect_metadata(
                "compile_distributed",
                world_size=world_size, dtype=args.dtype,
                hidden=args.hidden, intermediate=args.intermediate,
            )
            output["results"] = json_results
            write_json(args.json, output)

    if rank == 0 and not json_results:
        raise SystemExit("ERROR: all configs failed — 0 results collected")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
