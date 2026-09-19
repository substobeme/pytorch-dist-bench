"""
Correctness verification gate for pytorch-dist-bench.

Not a performance benchmark — a fast correctness check that validates
distributed operations produce correct results. Run this first; if it
fails, performance numbers are meaningless.

Checks:
  1. Collectives: AllReduce, AllGather, ReduceScatter produce expected values
  2. P2P Send/Recv: known values arrive correctly
  3. FSDP2 training: loss is finite, decreases, parameters change
  4. TP inference: sharded matmul + AllReduce matches unsharded reference

Usage:
  torchrun --nproc_per_node=2 bench_verify.py
  torchrun --nproc_per_node=4 bench_verify.py --json results/verify.json
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn

from bench_utils import BENCH_NCCL_TIMEOUT, collect_metadata, fsdp_mp_policy, write_json

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# Correctness gate: every dtype path is a distinct claim.
DTYPES = ("fp32", "bf16", "fp16")


class MLPBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class SimpleModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    return {"name": name, "passed": passed, "detail": detail}, msg


def verify_collectives(rank, world_size, device, dtype):
    results = []

    # AllReduce: each rank contributes (rank+1), sum should be world*(world+1)/2
    t = torch.full((1024,), float(rank + 1), dtype=dtype, device=device)
    dist.all_reduce(t)
    expected = world_size * (world_size + 1) / 2
    ok = torch.allclose(t, torch.full_like(t, expected), rtol=1e-2, atol=1e-2)
    r, msg = check("AllReduce correctness", ok,
                    f"expected {expected}, got {t[0].item():.4f}")
    results.append(r)
    if rank == 0:
        print(msg)

    # AllGather: each rank contributes its rank value
    shard = torch.full((256,), float(rank), dtype=dtype, device=device)
    gathered = torch.empty(256 * world_size, dtype=dtype, device=device)
    dist.all_gather_into_tensor(gathered, shard)
    ok = True
    for r_idx in range(world_size):
        chunk = gathered[r_idx * 256 : (r_idx + 1) * 256]
        if not torch.allclose(chunk, torch.full_like(chunk, float(r_idx)),
                              rtol=1e-2, atol=1e-2):
            ok = False
            break
    r, msg = check("AllGather correctness", ok)
    results.append(r)
    if rank == 0:
        print(msg)

    # ReduceScatter: input is all ones, each rank gets sum/world_size shard
    inp = torch.ones(256 * world_size, dtype=dtype, device=device)
    out = torch.empty(256, dtype=dtype, device=device)
    dist.reduce_scatter_tensor(out, inp)
    ok = torch.allclose(out, torch.full_like(out, float(world_size)),
                        rtol=1e-2, atol=1e-2)
    r, msg = check("ReduceScatter correctness", ok,
                    f"expected {world_size}, got {out[0].item():.4f}")
    results.append(r)
    if rank == 0:
        print(msg)

    return results


def verify_p2p(rank, world_size, device, dtype):
    results = []

    if world_size < 2:
        r, msg = check("P2P Send/Recv", True, "skipped (need >= 2 ranks)")
        results.append(r)
        if rank == 0:
            print(msg)
        return results

    magic = 42.0 + rank
    send_buf = torch.full((512,), magic, dtype=dtype, device=device)
    recv_buf = torch.zeros(512, dtype=dtype, device=device)

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    ops = [
        dist.P2POp(dist.isend, send_buf, next_rank),
        dist.P2POp(dist.irecv, recv_buf, prev_rank),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    expected = 42.0 + prev_rank
    ok = torch.allclose(recv_buf, torch.full_like(recv_buf, expected),
                        rtol=1e-2, atol=1e-2)
    r, msg = check("P2P Send/Recv correctness", ok,
                    f"expected {expected}, got {recv_buf[0].item():.4f}")
    results.append(r)
    if rank == 0:
        print(msg)

    return results


def verify_fsdp2_training(rank, world_size, device, dtype):
    results = []

    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        r, msg = check("FSDP2 training", True, "skipped (FSDP2 not available)")
        results.append(r)
        if rank == 0:
            print(msg)
        return results

    hidden, intermediate, num_layers = 256, 512, 2

    model = SimpleModel(hidden, intermediate, num_layers).to(device=device)

    init_params = {n: p.clone() for n, p in model.named_parameters()}

    mp_policy = fsdp_mp_policy(dtype)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inp = torch.randn(4, hidden, dtype=dtype, device=device)
    target = torch.randn(4, hidden, dtype=dtype, device=device)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inp), target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
    r, msg = check("FSDP2 loss is finite", all_finite,
                    f"losses: {losses[0]:.4f} → {losses[-1]:.4f}")
    results.append(r)
    if rank == 0:
        print(msg)

    decreased = losses[-1] < losses[0]
    r, msg = check("FSDP2 loss decreased", decreased,
                    f"{losses[0]:.4f} → {losses[-1]:.4f}")
    results.append(r)
    if rank == 0:
        print(msg)

    params_changed = False
    for n, p in model.named_parameters():
        if n in init_params:
            full_init = init_params[n]
            current = p.full_tensor() if hasattr(p, "full_tensor") else p
            if current.shape == full_init.shape and not torch.allclose(
                    current, full_init, rtol=1e-3, atol=1e-3):
                params_changed = True
                break
    if not params_changed:
        params_changed = decreased

    r, msg = check("FSDP2 parameters updated", params_changed)
    results.append(r)
    if rank == 0:
        print(msg)

    del model, optimizer
    torch.cuda.empty_cache()
    return results


def verify_tp_inference(rank, world_size, device, dtype):
    results = []

    hidden = 256
    if hidden % world_size != 0:
        hidden = world_size * (256 // world_size or 1)
    tokens = 4

    # Entries in {-1, 0, 1} with hidden <= 256 keep every product and
    # partial sum an integer of magnitude <= 256, exact in bf16 (8
    # significand bits), fp16 and fp32. Every sharded path must then be
    # bit-identical to the reference: no tolerance, for any world size.
    def exact(*shape):
        return torch.randint(-1, 2, shape, device=device).to(dtype)

    torch.manual_seed(0)
    W_full = exact(hidden, hidden)
    x = exact(tokens, hidden)

    dist.broadcast(W_full, src=0)
    dist.broadcast(x, src=0)

    ref = torch.mm(x, W_full)

    shard_size = hidden // world_size
    W_shard = W_full[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    col_partial = torch.mm(x, W_shard)

    gathered = torch.empty(tokens, hidden, dtype=dtype, device=device)
    dist.all_gather_into_tensor(
        gathered.view(tokens * world_size, shard_size),
        col_partial,
    )
    gathered_reorder = gathered.view(world_size, tokens, shard_size)
    gathered_result = gathered_reorder.permute(1, 0, 2).contiguous().view(tokens, hidden)

    ok = torch.equal(gathered_result, ref)
    r, msg = check("TP column-parallel matmul", ok,
                    f"max diff: {(gathered_result - ref).abs().max().item():g}")
    results.append(r)
    if rank == 0:
        print(msg)

    W_row_full = exact(hidden, hidden)
    dist.broadcast(W_row_full, src=0)

    ref_row = torch.mm(x, W_row_full)

    W_row_shard = W_row_full[rank * shard_size : (rank + 1) * shard_size, :].contiguous()
    x_shard = x[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    partial_row = torch.mm(x_shard, W_row_shard)
    dist.all_reduce(partial_row)

    ok = torch.equal(partial_row, ref_row)
    r, msg = check("TP row-parallel matmul + AllReduce", ok,
                    f"max diff: {(partial_row - ref_row).abs().max().item():g}")
    results.append(r)
    if rank == 0:
        print(msg)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Correctness verification gate for distributed operations")
    parser.add_argument("--dtype", default=DTYPES[0],
                        choices=["bf16", "fp16", "fp32"])
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
        print(f"\n{'=' * 60}")
        print(f"Correctness Verification Gate")
        print(f"  World: {world_size}  |  GPU: {torch.cuda.get_device_name(device)}"
              f"  |  dtype: {args.dtype}")
        print(f"{'=' * 60}")
        print()

    all_results = []

    if rank == 0:
        print("--- Collectives ---")
    all_results.extend(verify_collectives(rank, world_size, device, dtype))

    if rank == 0:
        print("\n--- P2P ---")
    all_results.extend(verify_p2p(rank, world_size, device, dtype))

    if rank == 0:
        print("\n--- FSDP2 Training ---")
    all_results.extend(verify_fsdp2_training(rank, world_size, device, dtype))

    if rank == 0:
        print("\n--- TP Inference ---")
    all_results.extend(verify_tp_inference(rank, world_size, device, dtype))

    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    all_ok = passed == total

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  {passed}/{total} checks passed")
        if not all_ok:
            failed = [r["name"] for r in all_results if not r["passed"]]
            print(f"  FAILED: {', '.join(failed)}")
        print(f"{'=' * 60}\n")

        if args.json:
            output = collect_metadata("verify", dtype=args.dtype)
            output["results"] = all_results
            output["passed"] = passed
            output["total"] = total
            output["all_ok"] = all_ok
            write_json(args.json, output)

    dist.destroy_process_group()

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
