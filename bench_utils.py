"""Shared benchmarking infrastructure for pytorch-dist-bench.

Every benchmark imports from here instead of duplicating timing,
metadata, and statistics code. This is the single place to fix
measurement methodology.
"""

import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone

import torch
import torch.distributed as dist

BENCH_NCCL_TIMEOUT = timedelta(seconds=120)

def sizes_in_elems(sizes_bytes, dtype):
    """Element counts for a byte-based size sweep, so every dtype moves the
    same messages."""
    return [n // dtype.itemsize for n in sizes_bytes]


NVLINK_UNIDIR_GBPS = {
    "H200": 450,
    "H100": 450,
    "A100": 300,
    "A30":  100,
    "V100": 150,
}


def get_gpu_peak_bandwidth():
    """Return theoretical peak bandwidth for the current GPU.

    Returns dict with nvlink_unidir_gbps (from lookup) and hbm_gbps
    (computed from device properties). Returns 0 for unknown GPUs.
    """
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name

    nvlink = 0
    for prefix, bw in NVLINK_UNIDIR_GBPS.items():
        if prefix in gpu_name:
            nvlink = bw
            break

    hbm = 0
    if props.memory_bus_width > 0 and props.memory_clock_rate > 0:
        hbm = round(
            props.memory_bus_width * props.memory_clock_rate * 2 / 8 / 1e6, 1)

    return {"nvlink_unidir_gbps": nvlink, "hbm_gbps": hbm}


def _get_gpu_driver():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True)
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown"


def _get_os_distro():
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME", "unknown")
    except Exception:
        return "unknown"


def fit_alpha_beta(sweep):
    """Fit alpha-beta model (time = alpha + nbytes/beta) to sweep data.

    Input: list of (nbytes, p50_us) tuples from a message size sweep.
    Returns: {"alpha_us": float, "beta_gbps": float, "r_squared": float}
             or None if fewer than 3 points or fit is degenerate.

    alpha_us  = latency (us): startup cost independent of message size.
    beta_gbps = bandwidth (GB/s): sustained throughput at large messages.
    r_squared = coefficient of determination (1.0 = perfect linear fit).
    """
    n = len(sweep)
    if n < 3:
        return None

    xs = [float(x) for x, _ in sweep]
    ys = [float(y) for _, y in sweep]

    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return None

    slope = (n * sxy - sx * sy) / denom
    alpha = (sy - slope * sx) / n

    if slope <= 0:
        return None

    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (alpha + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    beta_gbps = 1 / (slope * 1e3)

    return {
        "alpha_us": round(max(0, alpha), 2),
        "beta_gbps": round(beta_gbps, 1),
        "r_squared": round(r_sq, 4),
    }


def bench(fn, *, warmup=50, iters=200):
    """Time a CUDA-synchronous op with proper statistical reporting.

    Uses device-level synchronize before each clock read, which correctly
    measures NCCL collective completion on the local GPU. Does NOT insert
    cross-rank barriers — each rank independently measures its local view.

    Returns a stats dict with median, IQR, percentiles, and a variance flag.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)

    return stats(times)


def stats(times):
    """Compute stats from a list of microsecond timings.

    Same output format as bench(), for use by benchmarks that manage
    their own timing loop (e.g. bench_allreduce_dispatch.py).
    """
    times = sorted(times)
    n = len(times)
    q25 = times[n // 4]
    q75 = times[3 * n // 4]
    iqr = q75 - q25
    p50 = times[n // 2]

    result = {
        "p50_us": round(p50, 1),
        "mean_us": round(sum(times) / n, 1),
        "p5_us": round(times[max(0, int(n * 0.05))], 1),
        "p95_us": round(times[int(n * 0.95)], 1),
        "min_us": round(times[0], 1),
        "max_us": round(times[-1], 1),
        "iqr_us": round(iqr, 1),
        "iters": n,
    }

    if p50 > 0 and iqr / p50 > 0.10:
        result["warning"] = f"high variance: IQR/median={iqr / p50:.0%}"

    return result


def reset_nccl_tuning(fn, warmup=20, group=None):
    """Barrier + warmup between configs to let NCCL re-stabilize.

    NCCL's runtime tuner explores algorithms/protocols when tensor size
    changes. Without this, the first measured iterations at a new size
    may use a suboptimal algorithm, injecting multi-millisecond spikes.

    Call this once before the bench() call when switching tensor sizes.
    bench() does its own warmup for steady-state; this handles the transition.
    """
    dist.barrier(group=group)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()


def collect_metadata(benchmark_name, **kwargs):
    """Standard JSON metadata envelope for benchmark results.

    Pass parallelism degree as kwargs: tp=8, ep=8, dp=8, etc.
    """
    nccl_ver = torch.cuda.nccl.version()
    local_gpu_count = torch.cuda.device_count()
    peaks = get_gpu_peak_bandwidth()
    meta = {
        "benchmark": benchmark_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "pytorch_commit": getattr(torch.version, "git_version", "unknown"),
        "cuda_version": torch.version.cuda or "unknown",
        "nccl_version": f"{nccl_ver[0]}.{nccl_ver[1]}.{nccl_ver[2]}",
        "gpu": torch.cuda.get_device_name(),
        "gpu_count": local_gpu_count,
        "gpu_driver": _get_gpu_driver(),
        "gpu_peak_nvlink_gbps": peaks["nvlink_unidir_gbps"],
        "gpu_peak_hbm_gbps": peaks["hbm_gbps"],
        "os": platform.system(),
        "kernel": platform.release(),
        "os_distro": _get_os_distro(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
    }
    if dist.is_initialized():
        world_size = dist.get_world_size()
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", local_gpu_count))
        meta["world_size"] = world_size
        meta["num_nodes"] = world_size // max(local_ws, 1)
    nccl_env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    if nccl_env:
        meta["nccl_env"] = nccl_env
    meta.update(kwargs)
    return meta


# Loose on purpose: catches a wrong shard or a missing reduce, not ULPs.
# Absolute and scaled by max|out| because reduction outputs near zero carry
# the rounding error of their partials. Sized for <= 8 ranks.
VERIFY_ULPS = {torch.bfloat16: 8, torch.float16: 8, torch.float32: 256}


def verify_close(name, a, b, group=None):
    """Smoke-test that fused and unfused outputs agree; raise on all ranks.

    The verdict is all-reduced over `group` so ranks reaching this call
    raise together; one rank raising alone would desync the collectives
    that follow and hang the rest.
    """
    eps = torch.finfo(a.dtype).eps
    max_mag = max(a.abs().max().item(), b.abs().max().item())
    atol = max(eps, VERIFY_ULPS[a.dtype] * eps * max_mag)
    try:
        torch.testing.assert_close(b, a, rtol=0, atol=atol)
        err = None
    except AssertionError as e:
        err = f"Correctness check failed for {name}: {e}"

    if dist.is_initialized():
        flag = torch.tensor([float(err is not None)], device=a.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if flag.item() > 0:
            raise RuntimeError(err or f"{name}: failed on another rank")
    elif err:
        raise RuntimeError(err)


def fsdp_mp_policy(dtype):
    """FSDP2 policy: fp32 master weights, all-gather and reduce-scatter in
    `dtype`. Pure-dtype params are not a valid fp16 configuration: Adam's
    eps and (1-beta2)*g^2 underflow to 0, so the first step divides by 0.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy
    if dtype == torch.float32:
        return MixedPrecisionPolicy()
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)


def write_json(path, data):
    """Write JSON results to path (call from rank 0 only)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON results written to {path}")
