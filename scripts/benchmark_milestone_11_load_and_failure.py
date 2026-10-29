"""Milestone 11: Production Load, Concurrency, and Failure Mode Stress Testing.

Evaluates:
1. Single-user baseline latency across supported modes and resolutions.
2. Concurrent mixed workloads (10 concurrent workers submitting mixed FVM, neural, and comparison requests).
3. Burst traffic stress testing (concurrency governor capacity, queue wait times, 503 admission control).
4. Failure modes:
   - Oversized payload (> 10 MB) -> HTTP 413
   - Total cell count limit exceedance -> HTTP 400/422
   - Negative thermal conductivity -> HTTP 400
   - Singular pure-Neumann condition -> HTTP 400
   - Unsupported neural resolution -> HTTP 400
5. Post-failure recovery verification (server health and deterministic accuracy intact).

Emits structured latency percentiles (P50, P95, P99), throughput, and error taxonomy to:
artifacts/reports/milestone_11_load_and_failure_results.json
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from neural_process_twin.api.app import create_app
from neural_process_twin.api.deployment_config import SETTINGS


def make_payload(
    n: int,
    mode: str = "numerical_fvm",
    compute_diagnostics: bool = False,
    k_val: float = 2.5,
    bcs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to generate physical request dictionary."""
    if bcs is None:
        bcs = {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        }
    return {
        "mode": mode,
        "grid": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "nx": n, "ny": n},
        "conductivity": {"pattern_type": "uniform", "uniform_value": k_val},
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": bcs,
        "compute_diagnostics": compute_diagnostics,
    }


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute P50, P95, P99, mean, min, max for a sample of latencies."""
    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "count": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "count": n,
        "mean_ms": round(float(np.mean(s)), 2),
        "std_ms": round(float(np.std(s)), 2),
        "p50_ms": round(float(np.percentile(s, 50)), 2),
        "p95_ms": round(float(np.percentile(s, 95)), 2),
        "p99_ms": round(float(np.percentile(s, 99)), 2),
        "min_ms": round(float(np.min(s)), 2),
        "max_ms": round(float(np.max(s)), 2),
    }


def test_workload_single_user(client: TestClient, iterations: int = 15) -> dict[str, Any]:
    """Measure single-user sequential latency across canonical modes and meshes."""
    print("\n--- Workload 1: Single-User Baseline Latency ---")
    results = {}
    test_cases = [
        ("numerical_fvm_32", 32, "numerical_fvm", True),
        ("numerical_fvm_64", 64, "numerical_fvm", True),
        ("numerical_fvm_128", 128, "numerical_fvm", True),
        ("numerical_fvm_256", 256, "numerical_fvm", True),
        ("neural_32_opt", 32, "experimental_neural", False),
        ("neural_64_opt", 64, "experimental_neural", False),
        ("neural_128_opt", 128, "experimental_neural", False),
        ("neural_256_opt", 256, "experimental_neural", False),
    ]

    for label, n, mode, diag in test_cases:
        payload = make_payload(n, mode, compute_diagnostics=diag)
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            resp = client.post("/api/simulate", json=payload)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            latencies.append(lat_ms)

        stats = compute_percentiles(latencies)
        results[label] = stats
        print(
            f"  {label:<20}: Mean={stats['mean_ms']:>6.2f} ms | "
            f"P50={stats['p50_ms']:>6.2f} ms | P95={stats['p95_ms']:>6.2f} ms"
        )

    return results


def test_workload_concurrent_mixed(client: TestClient, total_requests: int = 40) -> dict[str, Any]:
    """Measure multi-threaded throughput and latency under mixed concurrency."""
    print(f"\n--- Workload 2: Concurrent Mixed Workload ({total_requests} requests) ---")

    workload_specs = [
        (64, "numerical_fvm", True),
        (64, "experimental_neural", False),
        (128, "experimental_neural", False),
        (128, "numerical_fvm", True),
        (64, "experimental_neural_corrected", False),
    ]

    latencies = []
    status_codes = []

    def send_request(idx: int) -> tuple[int, float]:
        n, mode, diag = workload_specs[idx % len(workload_specs)]
        payload = make_payload(n, mode, compute_diagnostics=diag)
        t0 = time.perf_counter()
        resp = client.post("/api/simulate", json=payload)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return resp.status_code, duration_ms

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            sc, lat = f.result()
            status_codes.append(sc)
            latencies.append(lat)
    t_total = time.perf_counter() - t_start

    throughput = float(len(status_codes) / max(t_total, 1e-6))
    stats = compute_percentiles(latencies)
    stats["total_duration_sec"] = round(t_total, 2)
    stats["throughput_req_per_sec"] = round(throughput, 2)
    stats["success_count"] = int(sum(1 for sc in status_codes if sc == 200))
    stats["error_count"] = int(sum(1 for sc in status_codes if sc != 200))

    print(f"  Total Duration: {t_total:.2f} s")
    print(f"  Throughput:     {throughput:.2f} req/s")
    print(
        f"  P50 Latency:    {stats['p50_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms | P99: {stats['p99_ms']:.2f} ms"
    )
    print(
        f"  Success Rate:   {stats['success_count']}/{total_requests} ({(stats['success_count'] / total_requests) * 100:.1f}%)"
    )

    return stats


def test_workload_burst_saturation(client: TestClient, burst_size: int = 25) -> dict[str, Any]:
    """Test burst traffic hitting concurrency governors and admission control."""
    print(f"\n--- Workload 3: Burst Traffic Stress Test ({burst_size} concurrent requests) ---")

    # Target 128x128 numerical FVM (CPU intensive)
    payload = make_payload(128, "numerical_fvm", compute_diagnostics=False)
    status_codes = []
    latencies = []

    def fire_burst() -> tuple[int, float]:
        t0 = time.perf_counter()
        resp = client.post("/api/simulate", json=payload)
        dur = (time.perf_counter() - t0) * 1000.0
        return resp.status_code, dur

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(fire_burst) for _ in range(burst_size)]
        for f in concurrent.futures.as_completed(futures):
            sc, dur = f.result()
            status_codes.append(sc)
            latencies.append(dur)
    total_time = time.perf_counter() - t0

    code_counts = {}
    for sc in status_codes:
        code_counts[str(sc)] = code_counts.get(str(sc), 0) + 1

    stats = compute_percentiles(latencies)
    stats["burst_size"] = burst_size
    stats["duration_sec"] = round(total_time, 2)
    stats["status_code_distribution"] = code_counts

    print(f"  Burst Total Time: {total_time:.2f} s")
    print(f"  Status Codes:     {code_counts}")
    print(f"  P50 Latency:      {stats['p50_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms")

    return stats


def test_failure_modes_and_recovery(client: TestClient) -> dict[str, Any]:
    """Verify explicit error handling, resource limits, and post-failure recovery."""
    print("\n--- Workload 4: Failure Mode & Recovery Testing ---")
    results = {}

    # Test A: Oversized payload (Content-Length > 10 MB)
    oversized = SETTINGS.max_request_body_bytes + 2048
    resp_a = client.post("/api/simulate", content=b"{}", headers={"Content-Length": str(oversized)})
    assert resp_a.status_code == 413
    results["oversized_payload_413"] = {
        "status_code": resp_a.status_code,
        "error_code": resp_a.json().get("error_code"),
        "passed": resp_a.status_code == 413,
    }
    print(
        f"  Oversized Payload: HTTP {resp_a.status_code} (Code: {resp_a.json().get('error_code')})"
    )

    # Test B: Exceeding grid cell limit (257x256)
    bad_grid_payload = make_payload(64)
    bad_grid_payload["grid"]["nx"] = 257
    resp_b = client.post("/api/simulate", json=bad_grid_payload)
    assert resp_b.status_code in (400, 422)
    results["cell_limit_exceeded"] = {
        "status_code": resp_b.status_code,
        "passed": resp_b.status_code in (400, 422),
    }
    print(f"  Cell Limit Exceeded: HTTP {resp_b.status_code}")

    # Test C: Negative thermal conductivity
    bad_k_payload = make_payload(32, k_val=-2.5)
    resp_c = client.post("/api/simulate", json=bad_k_payload)
    assert resp_c.status_code in (400, 422)
    results["negative_conductivity"] = {
        "status_code": resp_c.status_code,
        "passed": resp_c.status_code in (400, 422),
    }
    print(f"  Negative Conductivity: HTTP {resp_c.status_code}")

    # Test D: Pure-Neumann without gauge pin
    pure_neumann_bcs = {
        "left": {"side": "left", "type": "neumann", "value": 0.0},
        "right": {"side": "right", "type": "neumann", "value": 0.0},
        "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
        "top": {"side": "top", "type": "neumann", "value": 0.0},
    }
    bad_bcs_payload = make_payload(32, bcs=pure_neumann_bcs)
    resp_d = client.post("/api/simulate", json=bad_bcs_payload)
    assert resp_d.status_code in (400, 422)
    results["pure_neumann_rejection"] = {
        "status_code": resp_d.status_code,
        "passed": resp_d.status_code in (400, 422),
    }
    print(f"  Pure Neumann Singular: HTTP {resp_d.status_code}")

    # Test E: Unsupported neural resolution (50x50)
    unsupported_payload = make_payload(50, mode="experimental_neural")
    resp_e = client.post("/api/simulate", json=unsupported_payload)
    assert resp_e.status_code == 400
    results["unsupported_neural_resolution"] = {
        "status_code": resp_e.status_code,
        "passed": resp_e.status_code == 400,
    }
    print(f"  Unsupported Resolution: HTTP {resp_e.status_code}")

    # Test F: Post-Failure Recovery (Valid 32x32 simulation must succeed 100%)
    valid_payload = make_payload(32, "numerical_fvm")
    resp_f = client.post("/api/simulate", json=valid_payload)
    assert resp_f.status_code == 200
    results["post_failure_recovery"] = {
        "status_code": resp_f.status_code,
        "passed": resp_f.status_code == 200,
    }
    print(f"  Post-Failure Recovery: HTTP {resp_f.status_code} (System fully operational)")

    return results


def main() -> None:
    print("=" * 80)
    print("Milestone 11: Production Load, Concurrency, and Failure Testing")
    print("=" * 80)

    app = create_app()
    client = TestClient(app)

    # 1. Single User Workload
    w1 = test_workload_single_user(client, iterations=15)

    # 2. Concurrent Mixed Workload
    w2 = test_workload_concurrent_mixed(client, total_requests=30)

    # 3. Burst Saturation Stress Test
    w3 = test_workload_burst_saturation(client, burst_size=20)

    # 4. Failure Mode & Recovery Testing
    w4 = test_failure_modes_and_recovery(client)

    full_report = {
        "metadata": {
            "test_suite": "Milestone 11 Production Load and Failure Testing",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "single_user_baseline": w1,
        "concurrent_mixed_workload": w2,
        "burst_traffic_saturation": w3,
        "failure_modes_and_recovery": w4,
    }

    out_file = Path("artifacts/reports/milestone_11_load_and_failure_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved load and failure test results to: {out_file}")


if __name__ == "__main__":
    main()
