---
name: benchmark-authoring
description: Write new benchmarks for the pytorch-dist-bench suite following established conventions. Use when the user mentions new benchmark, add benchmark, write benchmark, benchmark template, benchmark convention, or needs to extend the suite with a new measurement.
---

# Benchmark Authoring

Conventions and patterns for writing new benchmarks in this suite. Every benchmark should produce valid, comparable JSON output and integrate cleanly with `run_all.sh` and `compare_results.py`.

## File Conventions

- Name: `bench_<descriptive_name>.py` in the repo root
- All benchmarks are standalone scripts launched with `torchrun --nproc_per_node=N`
- Single file per benchmark — no submodules

## Required Imports from bench_utils.py

```python
from bench_utils import bench, collect_metadata, reset_nccl_tuning, write_json

# Dtypes run_all.sh sweeps (read from this line); the first is the default.
# One line on why these dtypes measure something distinct.
DTYPES = ("bf16", "fp16", "fp32")
```

**`bench(fn, *, warmup=50, iters=200)`** — CUDA-synchronous timing. Returns stats dict with `p50_us`, `mean_us`, `p5_us`, `p95_us`, `min_us`, `max_us`, `iqr_us`, `iters`. Flags high variance (IQR/median > 10%). This is the only timing function — never roll your own `time.time()` loop.

**`reset_nccl_tuning(fn, warmup=20, group=None)`** — Barrier + warmup between configurations. Call once before `bench()` when switching tensor sizes or collectives. NCCL's runtime tuner explores algorithms on size changes; without this reset, the first measured iterations use a suboptimal algorithm. Pass `group` for sub-group benchmarks.

**`collect_metadata(benchmark_name, **kwargs)`** — JSON metadata envelope. Captures PyTorch version, commit, CUDA/NCCL versions, GPU model, hostname, world_size, num_nodes, NCCL env vars. Pass parallelism config as kwargs (e.g., `tp=8`, `dp=4`).

**`write_json(path, data)`** — Write JSON to path. Call from rank 0 only.

## Argparse Conventions

Every benchmark should accept:

```python
parser.add_argument("--section", default="all", choices=["all", "foo", "bar"])
parser.add_argument("--dtype", default=DTYPES[0], choices=["bf16", "fp16", "fp32"])
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--json", metavar="PATH", help="Write JSON results to PATH (rank 0 only)")
```

Use `--section` to allow running subsets independently. Pattern:

```python
if args.section in ("all", "foo"):
    # run section foo
```

For sweeps over model dimensions, add `--hidden`, `--intermediate`. For training sweeps, add `--num-layers`, `--batch-sizes` (nargs="+"). For pipeline parallelism, add `--num-microbatches`, `--pp-stages`.

## Main Function Pattern

```python
def main():
    # argparse setup
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    json_results = []

    # ... sections ...

    if rank == 0 and args.json:
        output = collect_metadata("benchmark_name", dtype=args.dtype, ...)
        output["results"] = json_results
        write_json(args.json, output)

    dist.destroy_process_group()
```

## Bandwidth Helpers

For collective benchmarks, compute and report algorithm and bus bandwidth:

```python
def algo_bw(nbytes, p50_us):
    if p50_us <= 0:
        return 0.0
    return nbytes / (p50_us * 1e-6) / 1e9  # GB/s

def bus_bw(algo_gbps, world_size):
    n = world_size
    return algo_gbps * 2 * (n - 1) / n  # AllReduce correction

def format_bytes(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"
```

Bus bandwidth correction factors differ by collective: AllReduce uses `2*(n-1)/n`, AllGather and ReduceScatter use `(n-1)/n`.

## Message Size Sweep

The standard sweep covers 11 message sizes from 1 KB to 1 GB, defined in
bytes so every dtype moves the same messages:

```python
SIZES = [1 << n for n in range(10, 31, 2)]      # bytes
...
for nelems in sizes_in_elems(SIZES, dtype):      # bench_utils
```

Use this so `compare_results.py` labels (`nelems`) line up with the existing
benchmarks; for bf16 the element counts are 512 .. 536870912.

## OOM Resilience Pattern

For benchmarks with model allocation (training steps, FSDP2, PP), use the allreduce success flag to synchronize all ranks when one rank OOMs:

```python
ok = torch.tensor([1.0], device=device)
result = None
try:
    # model setup and benchmarking
    result = bench_something(...)
    if result is None:
        ok.zero_()
except torch.cuda.OutOfMemoryError:
    ok.zero_()
    torch.cuda.empty_cache()

dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if ok.item() < 1.0:
    torch.cuda.empty_cache()
    continue
```

This prevents deadlocks where one rank OOMs during local setup while other ranks proceed to collective operations. Note: OOM during a collective operation itself (inside `fully_shard()` or `parallelize_module()`) cannot be caught at the Python level — that requires NCCL timeouts (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`).

## Model Pattern

Use the standard MLP block with residual connections, matching Llama dimensions:

```python
class MLPBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))

class BenchModel(nn.Module):
    def __init__(self, hidden, intermediate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPBlock(hidden, intermediate) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # residual
        return x
```

Default dimensions: Llama-8B (hidden=4096, intermediate=14336), Llama-70B (hidden=8192, intermediate=28672).

## Integration Checklist

After writing the benchmark:

1. **`run_all.sh`** — add to the `BENCHMARKS` array in logical order; the dtypes it sweeps come from the script's `DTYPES` line (it aborts if a script takes `--dtype` but declares none)
2. **`compare_results.py`** — if your JSON entries need a new field to tell them apart (e.g. `num_microbatches`), add it to `VALUE_KEYS` (or `LABEL_KEYS` for string fields); both entry matching and labels use those tuples
3. **`README.md`** — update the benchmark count, add a row to the benchmarks table (include NVSwitch requirement), update the portable subset count if applicable, add to the project structure listing, and add the benchmark to the dtype sweep table
4. **Syntax check** — `python -m py_compile bench_new.py`
5. **Smoke test** — `torchrun --nproc_per_node=2 bench_new.py --iters 3 --warmup 2`
6. **Full run** — `./run_all.sh 2` to verify the new benchmark integrates without breaking others

## Anti-Patterns

**Missing synchronize**: `bench()` handles this — but if you write manual timing, always `torch.cuda.synchronize()` before each clock read. `time.time()` around a CUDA call measures launch overhead, not execution.

**No `reset_nccl_tuning` between sizes**: NCCL explores algorithms when tensor sizes change. Without the reset, the first iterations at a new size use a suboptimal algorithm, injecting multi-millisecond spikes that corrupt your p50.

**Barrier mismatches**: Every rank must call the same number of barriers and collectives. If rank 0 skips a section but other ranks don't, deadlock. When ranks diverge (e.g., idle ranks in a topology section), make sure idle ranks call exactly the same barriers as active ranks.

**Forgetting `dist.destroy_process_group()`**: Omitting this causes NCCL to leak resources and print warnings on exit.

**Using `dist.barrier()` inside timing loops**: Barriers serialize all ranks and add latency. `bench()` intentionally does NOT barrier between iterations — each rank measures its own local view. Only use barriers in `reset_nccl_tuning` between configurations.

**Allocating inside timing loops**: Memory allocation is slow. Pre-allocate all tensors before `bench()`. If you need different sizes, allocate once at the maximum and slice.
