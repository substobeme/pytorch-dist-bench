# pytorch-dist-bench

Microbenchmark suite for PyTorch distributed operations. Tracks performance across PyTorch releases on GPU clusters, with a focus on the collective operations that dominate tensor-parallel inference and FSDP2 training, including `torch.compile` interactions.

Designed for single-node multi-GPU systems (tested on 8×H200 NVSwitch), with opt-in multi-node benchmarking for inter-node (IB/RoCE) performance. All benchmarks produce structured JSON for automated regression detection.

## Quick start

```bash
# Run all 14 single-node benchmarks on 8 GPUs, write JSON to ./results/
./run_all.sh 8

# Run a single benchmark
torchrun --nproc_per_node=8 bench_collectives.py --json results/collectives.json

# Compare two runs
python compare_results.py results/baseline/ results/test/ --threshold 5
```

### Data types

Each benchmark declares the dtypes for which it measures something distinct
in a `DTYPES = (...)` line at the top of its script, with the reason beside
it; the first entry is the script's default. `run_all.sh` reads that line,
runs one torchrun job per dtype and writes `<bench>_tp<N>_<dtype>.json`;
`--dtypes "bf16"` restricts the sweep. Any dtype can still be forced on a
single run with `--dtype`.

| Sweep | Benchmarks | Why |
|---|---|---|
| bf16 fp16 fp32 | all dtype-aware benchmarks except `inference_tp_layer` (order is default-first; `verify` defaults to fp32) | each dtype has its own kernels (NCCL reduction, cuBLAS GEMM, Inductor codegen); training benchmarks keep fp32 master weights and run the collectives in the sweep dtype |
| bf16 fp16 | `inference_tp_layer` | fused symm-mem GEMMs are 16-bit inference paths; fp32 at 405B/S=32K is ~10x slower per iteration and overruns the per-run timeout (`--dtype fp32` still works) |

Byte-based size sweeps (`collectives`, `pipeline_parallel`, `multinode`) move
the same messages in every dtype; element counts scale with `itemsize`.
NVLS `multimem_all_reduce_` has no fp16 kernel, so those rows record the
NCCL number with the symm-mem field null.

Training benchmarks hold fp32 master weights and optimizer state, with
`MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)` under FSDP2
and `autocast` for the non-FSDP GPipe stage (which re-casts the fp32
weights to `dtype` on every microbatch, a cost real GPipe users also pay);
pure fp16 parameters with Adam diverge on the first step. `reduce_dtype=dtype` is deliberate so the
gradient reduce-scatter runs in the sweep dtype (torchtitan's default
reduces in fp32 and moves 2x the bytes). Under fp16, FSDP2 reduces with SUM
plus pre/post-scaling kernels rather than a single AVG, so fp16-vs-bf16
step-time deltas are not purely the NCCL dtype.

## Requirements

- PyTorch 2.6+ (for `fully_shard`, symmetric memory, FP8 fused ops)
- CUDA 12.x
- NCCL 2.21+
- 2+ NVIDIA GPUs (8× with NVSwitch recommended)

Some benchmarks require NVSwitch and symmetric memory support (see table below). The remainder run on any multi-GPU system.

## Benchmarks

| Benchmark | What it measures | NVSwitch required? |
|---|---|---|
| `bench_verify` | Correctness gate: validates AllReduce, AllGather, ReduceScatter, P2P Send/Recv, FSDP2 training (loss finite + decreasing + params updated), and TP inference (sharded matmul matches reference). Run first — if correctness fails, performance numbers are meaningless. | No |
| `bench_collectives` | AllReduce, AllGather, ReduceScatter at 11 message sizes (1KB–1GB) through `torch.distributed`. The nccl-tests equivalent through the ProcessGroup stack. Reports efficiency % vs NVLink peak. | No |
| `bench_symm_mem_fused_ops` | Fused GEMM+ReduceScatter, AllGather+GEMM, NVLS AllReduce via `torch.distributed._symmetric_memory`. The TP inference fast path. | Yes |
| `bench_fp8_fused_ops` | FP8 scaled fused ops (`_fused_all_gather_scaled_matmul`, `_fused_scaled_matmul_reduce_scatter`) vs unfused equivalents. The quantized inference path. | Yes |
| `bench_migration_path` | pynccl → `torch.distributed` → fused ops progression. Validates that migrating dispatch paths doesn't regress and that fused ops improve latency. | Yes |
| `bench_inference_tp_layer` | Full TP transformer layer (attention + MLP) with fused vs unfused collectives. Composite benchmark at real Llama-70B dimensions. | Yes |
| `bench_inference_tp_vllm` | vLLM-style TP inference with standard `dist.all_reduce`: AllReduce at decode (latency-dominated) and prefill (bandwidth-dominated) tensor sizes, plus a full TP layer (4 GEMMs + 2 AllReduces) for Llama-8B/70B/405B. | No |
| `bench_training_fsdp_collectives` | FSDP2-shaped AllGather/ReduceScatter, low-contention AllGather (copy engine), NVLS AllReduce. Raw collective ops at FSDP parameter shard sizes. | Yes |
| `bench_fsdp2_training` | Complete FSDP2 training step (`fully_shard()` → zero_grad → forward → backward → optimizer.step) on MLP blocks at Llama-70B dimensions. Tests FSDP2's overlap scheduling end-to-end. | No |
| `bench_pipeline_parallel` | P2P Send/Recv sweep (NVLink/IB point-to-point bandwidth), GPipe pipeline training step, and FSDP2+PP combined (2D mesh with PP stages × DP replicas). | No |
| `bench_moe_alltoall` | MoE expert-parallel all-to-all dispatch with balanced and skewed (Zipf) routing. Mixtral-8x7B and DeepSeek-V2 shapes. | No |
| `bench_allreduce_dispatch` | CPU dispatch overhead: pynccl vs ProcessGroupNCCL, with CUDA event timing and CUDA graph variants. | No |
| `bench_compile_distributed` | `torch.compile` (Inductor) vs eager on FSDP2 training steps and TP-style inference. Tracks whether compile helps, hurts, or breaks distributed workloads across releases. | No |
| `bench_e2e` | End-to-end distributed workloads: FSDP2 training throughput (samples/sec), TP inference throughput (tokens/sec at Llama-70B dims), and FSDP2+PP combined training. Reports throughput metrics rather than individual op latency. | No |
| `bench_multinode` | Multi-node collectives decomposed by topology (intra-node NVLink, inter-node IB/RoCE raw + aggregate, WORLD) plus 2D parallelism training (TP intra-node + FSDP2 DP inter-node). **Opt-in** — not run by `run_all.sh`. | No |

### Portable subset

9 single-node benchmarks run on any multi-GPU system without NVSwitch or symmetric memory: `bench_verify`, `bench_collectives`, `bench_inference_tp_vllm`, `bench_fsdp2_training`, `bench_pipeline_parallel`, `bench_moe_alltoall`, `bench_allreduce_dispatch`, `bench_compile_distributed`, `bench_e2e`.

`bench_multinode` also runs without NVSwitch but requires a multi-node setup (see below).

## Multi-node benchmarks

`bench_multinode` is opt-in and requires a multi-node cluster. It is **not** included in `run_all.sh`. Run single-node benchmarks first (`run_all.sh`) for NVLink baselines, then add multi-node measurements.

### Launch with torchrun (bare-metal / SLURM)

Run on each node (or fan out via `pdsh` / `srun`):

```bash
./run_multinode.sh 3 2 <master-ip> 29500
```

Or directly:

```bash
torchrun --nnodes=3 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=<master-ip>:29500 \
  bench_multinode.py --json results/multinode_3n2g_bf16.json --dtype bf16
```

Run only one section:

```bash
# Collectives only (skip 2D training)
./run_multinode.sh 3 2 <master-ip> 29500 -- --section collectives

# Training only (skip raw collectives)
./run_multinode.sh 3 2 <master-ip> 29500 -- --section training

# Custom model dimensions and sweep
./run_multinode.sh 3 2 <master-ip> 29500 -- --section training \
  --hidden 4096 --intermediate 11008 --num-layers 2 4 8 --batch-sizes 1 2
```

### Launch on Kubernetes (PyTorchJob)

Requires the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator). Edit `k8s/pytorchjob.yaml` to set your container image and NCCL/RDMA configuration, then:

```bash
kubectl apply -f k8s/pytorchjob.yaml
kubectl logs -f pytorch-dist-bench-multinode-master-0
```

For IB/RoCE bandwidth (not TCP socket fallback), pods need RDMA device access. See comments in the YAML.

### NCCL timeout

Set a heartbeat timeout so OOM or network failures surface as errors instead of silent hangs:

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300  # default 1800 (30 min)
```

### What it measures

**Collectives by topology** — AllReduce, AllGather, ReduceScatter at the same 11 message sizes as `bench_collectives`, but on four process groups:

| Group | Ranks (3n2g) | Measures |
|---|---|---|
| `intra_node` | {0,1}, {2,3}, {4,5} | NVLink bandwidth within each node |
| `inter_node` | {0,2,4} (local_rank=0 only) | Raw single-flow IB/RoCE link bandwidth |
| `inter_agg` | {0,2,4} + {1,3,5} simultaneously | Aggregate IB/RoCE bandwidth under contention |
| `world` | {0,1,2,3,4,5} | Hierarchical (NVLink + IB/RoCE combined) |

**2D parallelism training** — TP=2 intra-node (ColwiseParallel/RowwiseParallel) + FSDP2 DP=3 inter-node, with a training step sweep over layer counts and batch sizes. Default dimensions match Llama-70B (hidden=8192, intermediate=28672); override with `--hidden` and `--intermediate`.

## JSON output

Every benchmark writes structured JSON through `bench_utils.write_json()`:

```json
{
  "benchmark": "collectives",
  "timestamp": "2026-07-14T18:30:00+00:00",
  "pytorch_version": "2.8.0a0+git1234abc",
  "pytorch_commit": "1234abc",
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
  "dp": 8,
  "dtype": "bf16",
  "results": [
    {
      "collective": "all_reduce",
      "nelems": 536870912,
      "nbytes": 1073741824,
      "stats": {
        "p50_us": 1234.5,
        "mean_us": 1250.3,
        "p5_us": 1200.1,
        "p95_us": 1310.2,
        "min_us": 1195.0,
        "max_us": 1450.8,
        "iqr_us": 45.2,
        "iters": 200
      },
      "algo_bw_gbps": 810.5,
      "bus_bw_gbps": 709.2
    }
  ]
}
```

The `p50_us` field (median latency in microseconds) is the primary metric used for regression detection.

## Comparing results

`compare_results.py` matches JSON files by filename between two result directories, extracts `p50_us` metrics, and flags regressions. Files are named `<bench>_tp<N>_<dtype>.json` (or `<bench>_tp<N>.json` for benchmarks without a dtype option), and paired files must agree on `benchmark` and `dtype`; a pair with no comparable metrics is reported and the exit code is 2.

Baselines produced before the dtype suffix: for `verify`, `collectives`, `symm_mem_fused_ops`, `inference_tp_layer`, `inference_tp_vllm`, `moe_alltoall` and `training_fsdp_collectives`, rename `<bench>_tp<N>.json` to `<bench>_tp<N>_<dtype>.json` using the file's own top-level `dtype` field (bf16 for all of these except `verify`, which defaulted to fp32); `compare_results.py` refuses a pair whose `dtype` fields differ, so a wrong guess is caught. The training benchmarks (`fsdp2_training`, `e2e`, `compile_distributed`, `pipeline_parallel`, `multinode`) changed configuration — fp32 master weights instead of parameters in the run dtype — so step times and `peak_MB` moved and their baselines must be regenerated, not renamed (`pipeline_parallel` is one file, so its P2P rows go with it).

```bash
python compare_results.py results/baseline/ results/test/
```

```
=== bench_collectives_tp8_bf16.json ===
  all_reduce  nelems=536870912      stats    1234.5 ->  1298.7  (+5.2%)  REGRESSION
  all_gather  nelems=536870912      stats    1100.2 ->  1045.1  (-5.0%)  IMPROVED
  ...

Summary: 42 metrics compared
  1 REGRESSIONS (>5.0% slower)
  1 improvements (<-5.0% faster)
  40 unchanged (within +/-5.0%)
```

Exit code: 1 when any regression exceeds the threshold (default: 5%); 2 when the comparison is incomplete or inconsistent (a baseline file or entry missing from the test run, paired files disagreeing on `benchmark`/`dtype`, or no comparable metrics); 0 otherwise. A regression takes precedence over an incomplete run.

## A/B testing a PyTorch PR

`ab_test_pytorch_pr.sh` automates the full workflow: build baseline → run benchmarks → apply PR → rebuild → run benchmarks → compare.

```bash
./ab_test_pytorch_pr.sh 187642 8                       # Test PR #187642 on 8 GPUs
./ab_test_pytorch_pr.sh abc1234 4 /opt/pytorch         # Custom source path
PYTORCH_DIR=/opt/pytorch ./ab_test_pytorch_pr.sh 187642 8  # Via env var
```

Requires a PyTorch source checkout. Set `PYTORCH_DIR`, pass as the 3rd argument, or have PyTorch installed from source (auto-detected).

## Measurement methodology

All benchmarks share infrastructure through `bench_utils.py`:

- **`bench(fn, warmup=50, iters=200)`** — CUDA-synchronous timing with `torch.cuda.synchronize()` before each clock read. Reports p50, p5, p95, IQR. Flags runs where IQR/median exceeds 10%.
- **`reset_nccl_tuning(fn, warmup=20, group=None)`** — Barrier + warmup between configurations. NCCL's runtime tuner explores algorithms when tensor sizes change; without this reset, the first iterations at a new size use a suboptimal algorithm and inject multi-millisecond spikes. Pass `group` for sub-group benchmarks (e.g. intra-node or inter-node topologies).
- **`collect_metadata(name, **kwargs)`** — Captures PyTorch version, commit SHA, CUDA/NCCL versions, GPU model, GPU driver, peak NVLink/HBM bandwidth, OS distro, kernel version, hostname, world size, node count, NCCL environment variables, and parallelism configuration.
- **Sequential execution** — `run_all.sh` runs benchmarks one at a time to prevent GPU contention from corrupting measurements.

## Project structure

```
bench_utils.py                  # Shared timing, stats, metadata, JSON output
bench_verify.py                 # Correctness gate (collectives, P2P, FSDP2, TP)
bench_collectives.py            # Raw collective sweep (AR/AG/RS × 11 sizes)
bench_symm_mem_fused_ops.py     # Fused ops (symmetric memory)
bench_fp8_fused_ops.py          # FP8 scaled fused ops
bench_migration_path.py         # pynccl → dist → fused migration
bench_inference_tp_layer.py     # Full TP layer (attention + MLP)
bench_inference_tp_vllm.py      # vLLM-style TP inference (standard AllReduce)
bench_training_fsdp_collectives.py  # FSDP2-shaped raw collectives
bench_fsdp2_training.py         # FSDP2 training step (fully_shard)
bench_pipeline_parallel.py      # P2P sweep, GPipe pipeline, FSDP2+PP
bench_moe_alltoall.py           # MoE expert-parallel all-to-all
bench_allreduce_dispatch.py     # Dispatch overhead comparison
bench_compile_distributed.py    # torch.compile vs eager (FSDP2 + TP)
bench_e2e.py                    # End-to-end workloads (FSDP2 training, TP inference, FSDP2+PP)
bench_multinode.py              # Multi-node: topology-decomposed collectives + 2D training
run_all.sh                      # Sequential runner for single-node benchmarks
run_multinode.sh                # Multi-node launcher (torchrun with rendezvous)
ab_test_pytorch_pr.sh           # A/B test harness for PyTorch PRs
compare_results.py              # JSON regression detector
k8s/pytorchjob.yaml             # Kubernetes PyTorchJob manifest (3 nodes × 2 GPUs)
```

## License

Apache-2.0
