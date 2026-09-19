---
name: run-and-compare
description: Run pytorch-dist-bench benchmarks and compare results across versions. Use when the user mentions run benchmarks, compare results, A/B test, baseline, before/after, compare versions, run_all, compare_results, ab_test, or needs to produce benchmark data for analysis.
---

# Run and Compare

How to run the benchmark suite, produce valid comparisons, and avoid statistical pitfalls.

## Running Benchmarks

### Full single-node suite

```bash
./run_all.sh 8                              # 8 GPUs, JSON to ./results/
./run_all.sh 8 --json-dir /path/to/output   # custom output directory
./run_all.sh 8 --dtypes "bf16"              # subset of each benchmark's sweep
```

Runs all 14 benchmarks sequentially (prevents GPU contention), starting with `bench_verify` (correctness gate). Each benchmark runs once per dtype in the `DTYPES = (...)` line at the top of its script (the README "Data types" table says why) and writes `bench_NAME_tpN_<dtype>.json`; the three benchmarks without a `--dtype` option (`allreduce_dispatch`, `fp8_fused_ops`, `migration_path`) write `bench_NAME_tpN.json`.

### Individual benchmarks

```bash
torchrun --nproc_per_node=8 bench_collectives.py --json results/collectives.json
torchrun --nproc_per_node=8 bench_inference_tp_vllm.py --section allreduce --json results/tp_ar.json
torchrun --nproc_per_node=4 bench_pipeline_parallel.py --section fsdp2_pp --pp-stages 2 --json results/pp.json
```

All benchmarks accept `--json PATH` (rank 0 writes); all but `bench_verify` accept `--warmup N` and `--iters N`. All but `allreduce_dispatch`, `fp8_fused_ops` and `migration_path` accept `--dtype bf16|fp16|fp32` (any dtype can be forced, even one outside the benchmark's declared sweep). `e2e`, `inference_tp_vllm`, `multinode` and `pipeline_parallel` accept `--section` to run a subset.

### Multi-node

```bash
# Via helper script (each node)
./run_multinode.sh <nnodes> <nproc_per_node> <master-ip> <port>
DTYPE=fp16 ./run_multinode.sh <nnodes> <nproc_per_node> <master-ip> <port>   # one dtype per run

# Via torchrun directly
torchrun --nnodes=3 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=master:29500 \
  bench_multinode.py --json results/multinode_3n8g_bf16.json --dtype bf16

# Via Kubernetes
kubectl apply -f k8s/pytorchjob.yaml
```

## Comparing Results

```bash
python compare_results.py results/baseline/ results/test/
python compare_results.py results/baseline/ results/test/ --threshold 10
```

The tool matches JSON files by filename between directories, extracts all `p50_us` values (and pass/fail for `bench_verify`), and reports % change. Exit code 1 if any regression exceeds the threshold; 2 if the comparison is incomplete (baseline files or entries missing from the test run, `benchmark`/`dtype` mismatch between paired files, no comparable metrics); 0 otherwise.

**Reading the output**:
- Each row shows: label, metric path, baseline p50_us → test p50_us, (% change), flag
- `REGRESSION` = slower by more than threshold
- `IMPROVED` = faster by more than threshold
- No flag = within threshold (unchanged)

Labels and match keys come from `LABEL_KEYS` and `VALUE_KEYS` in `compare_results.py` (section, topology, collective, op, routing, model, param_name, name; nelems, seq_len, num_tokens, num_layers, batch_size, num_microbatches, dtype, hidden).

## A/B Testing a PyTorch PR

```bash
./ab_test_pytorch_pr.sh 187642 8                       # Test PR #187642 on 8 GPUs
./ab_test_pytorch_pr.sh abc1234 4 /opt/pytorch         # Commit SHA, custom source path
PYTORCH_DIR=/opt/pytorch ./ab_test_pytorch_pr.sh 187642 8
```

Workflow: build baseline (current HEAD) → run all benchmarks → apply PR → rebuild → run all benchmarks → restore baseline. Results land in `results/<timestamp>_baseline_<sha>/` and `results/<timestamp>_test_<sha>/`.

Requires a PyTorch source checkout. Set `PYTORCH_DIR`, pass as 3rd arg, or install from source (auto-detected).

## Environment Setup

### NCCL timeout

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300  # default 1800; surfaces hangs faster
```

### GPU clocks (for reproducible results)

```bash
nvidia-smi -pm 1                    # persistence mode
nvidia-smi -lgc <max_clock>         # lock graphics clock
nvidia-smi -lmc <max_mem_clock>     # lock memory clock
```

### Useful NCCL env vars

```bash
NCCL_DEBUG=INFO                     # topology and algorithm selection
NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log # per-process log files
NCCL_ALGO=Ring,Tree                 # restrict algorithms (debugging only)
```

## JSON Format

Every benchmark writes through `bench_utils.write_json()`:

```json
{
  "benchmark": "collectives",
  "timestamp": "...",
  "pytorch_version": "2.8.0a0+git...",
  "pytorch_commit": "abc1234",
  "cuda_version": "12.6",
  "nccl_version": "2.25.1",
  "gpu": "NVIDIA H200",
  "gpu_count": 8,
  "gpu_driver": "550.54.15",
  "gpu_peak_nvlink_gbps": 450,
  "gpu_peak_hbm_gbps": 4800.0,
  "os": "Linux",
  "kernel": "5.14.0-615.el9.x86_64",
  "os_distro": "Red Hat Enterprise Linux 9.4",
  "arch": "x86_64",
  "hostname": "...",
  "world_size": 8,
  "num_nodes": 1,
  "results": [
    {
      "section": "...",
      "stats": {"p50_us": 1234.5, "mean_us": ..., "iqr_us": ..., "iters": 200},
      ...
    }
  ]
}
```

The `p50_us` field (median latency in microseconds) is the primary regression metric. The `iqr_us` field indicates measurement stability — high IQR/median ratios (>10%) trigger a warning in the stats.

## Measurement Validity

**Single-run comparisons are directional, not conclusive.** A single `compare_results.py` run compares one p50 value against another. If the IQR is large relative to the delta, the apparent regression may be noise. For reportable claims (impact reports, release decisions):

1. Run each configuration at least 3 times independently (separate `torchrun` invocations, not just more `--iters`)
2. Check that the delta consistently exceeds the pooled IQR across runs
3. Lock GPU clocks to reduce run-to-run variance

**Attribution requires isolation.** `ab_test_pytorch_pr.sh` changes exactly one variable (a single PR applied to a fixed baseline) — that supports direct attribution. A release-over-release comparison changes PyTorch, NCCL, driver, and possibly firmware simultaneously. To attribute a release-level improvement to a specific change, decompose: test each variable change in isolation with `ab_test_pytorch_pr.sh` or equivalent controlled builds.
