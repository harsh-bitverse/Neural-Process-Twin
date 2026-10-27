#!/usr/bin/env python3
"""Clean CPU Latency and Throughput Benchmark for Milestone 08D (Phase 6).

Benchmarks:
- U0: Baseline M3 (Single forward pass)
- U1: Heteroscedastic FNO (Single-pass mean + scale)
- U2: Evidential FNO (Single-pass DER)
- U3: Spatial MC Dropout FNO (T=3, T=5, T=8 stochastic forward passes)
- U4: Distilled Student FNO (Single-pass epistemic student)
- Ens5: Historical 5-member ensemble (evaluated on 32x32)
- FVM: Reference PyAMG solver

Measurement Protocol:
- STRICT GOVERNANCE: NO tracemalloc during latency measurements (prevents 10.8x profiling overhead).
- 5 warm-up cycles, 20 measurement cycles per resolution.
- Latency dissection:
  1. Input preparation & boundary encoding
  2. Neural operator forward execution
  3. Uncertainty extraction (variance/std calculation)
  4. Physical PDE residual computation
  5. Total end-to-end latency
- Resolutions: 32x32, 64x64, 128x128, 256x256.

Persists results to:
- artifacts/reports/milestone_08d_benchmarks.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.models.uncertainty_fno import (
    DistilledStudentFNO2D,
    EvidentialFNO2D,
    HeteroscedasticFNO2D,
    MCDropoutFNO2D,
)
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_08d_latency")


def set_threads(n_threads: int = 8) -> None:
    torch.set_num_threads(n_threads)
    torch.manual_seed(42)
    np.random.seed(42)


def make_dummy_bcs() -> BoundaryConditions:
    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 350.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 290.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )


class LatencyBenchmarker:
    def __init__(
        self,
        checkpoints_dir: Path = Path("artifacts/checkpoints/milestone_08d"),
        m3_checkpoint: Path = Path(
            "artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt"
        ),
        n_warmup: int = 5,
        n_iters: int = 20,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.m3_checkpoint = m3_checkpoint
        self.n_warmup = n_warmup
        self.n_iters = n_iters
        m3_ckpt = torch.load(self.m3_checkpoint, map_location="cpu", weights_only=False)
        self.normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
        self.temp_std = float(self.normalizer.t_stats.std)
        self.fvm_solver = HeatConductionSolver2D()

        set_threads(8)
        self.models = self._load_models()

    def _load_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {}

        # U0
        if self.m3_checkpoint.exists():
            u0 = FourierNeuralOperator2D(
                in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
            )
            st = torch.load(self.m3_checkpoint, map_location="cpu")
            u0.load_state_dict(st.get("model_state_dict", st))
            u0.eval()
            models["U0_baseline_m3"] = u0

        # U1
        u1_p = self.checkpoints_dir / "U1_heteroscedastic" / "best_model.pt"
        if u1_p.exists():
            u1 = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u1_p, map_location="cpu")
            u1.load_state_dict(st.get("model_state_dict", st))
            u1.eval()
            models["U1_heteroscedastic"] = u1

        # U2
        u2_p = self.checkpoints_dir / "U2_evidential" / "best_model.pt"
        if u2_p.exists():
            u2 = EvidentialFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u2_p, map_location="cpu")
            u2.load_state_dict(st.get("model_state_dict", st))
            u2.eval()
            models["U2_evidential"] = u2

        # U3
        u3_p = self.checkpoints_dir / "U3_mc_dropout" / "best_model.pt"
        if u3_p.exists():
            u3 = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
            st = torch.load(u3_p, map_location="cpu")
            u3.load_state_dict(st.get("model_state_dict", st))
            u3.eval()
            models["U3_mcdropout_T3"] = (u3, 3)
            models["U3_mcdropout_T5"] = (u3, 5)
            models["U3_mcdropout_T8"] = (u3, 8)

        # U4
        u4_p = self.checkpoints_dir / "U4_distilled_student" / "best_model.pt"
        if u4_p.exists():
            u4 = DistilledStudentFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u4_p, map_location="cpu")
            u4.load_state_dict(st.get("model_state_dict", st))
            u4.eval()
            models["U4_distilled_student"] = u4

        # Ensemble5
        ens_models = []
        for seed in [42, 43, 44, 45, 46]:
            ckpt = Path(f"artifacts/checkpoints/scaled_fno_supervised_seed{seed}.pt")
            if ckpt.exists():
                m = FourierNeuralOperator2D(
                    in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
                )
                st = torch.load(ckpt, map_location="cpu")
                m.load_state_dict(st.get("model_state_dict", st))
                m.eval()
                ens_models.append(m)
        if len(ens_models) == 5:
            models["Ensemble5"] = ens_models

        return models

    def benchmark_model_at_resolution(
        self,
        model_name: str,
        res: int,
    ) -> dict[str, float]:
        """Dissect and benchmark latency for a single candidate at given resolution."""
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
        bcs = make_dummy_bcs()
        k_field = np.ones((res, res), dtype=np.float32) * 2.0
        q_field = np.zeros((res, res), dtype=np.float32)

        x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
        y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)
        (1.0 - x_norm)[np.newaxis, :, :]
        x_norm[np.newaxis, :, :]
        (1.0 - y_norm)[np.newaxis, :, :]
        y_norm[np.newaxis, :, :]

        model_entry = self.models[model_name]

        # Warm-up cycles
        dummy_x = torch.zeros((1, 12, res, res), dtype=torch.float32)
        with torch.no_grad():
            for _ in range(self.n_warmup):
                if (
                    model_name == "U0_baseline_m3"
                    or model_name == "U1_heteroscedastic"
                    or model_name == "U2_evidential"
                ):
                    _ = model_entry(dummy_x)
                elif model_name.startswith("U3_mcdropout"):
                    u3, t_samples = model_entry
                    u3.train()
                    for _ in range(t_samples):
                        _ = u3(dummy_x)
                    u3.eval()
                elif model_name == "U4_distilled_student":
                    _ = model_entry(dummy_x)
                elif model_name == "Ensemble5":
                    for m in model_entry:
                        _ = m(dummy_x)

        # Measurement loops for dissected latency
        prep_times = []
        forward_times = []
        unc_times = []
        phys_times = []
        total_times = []

        for _ in range(self.n_iters):
            t_start = time.perf_counter()

            # 1. Input preparation
            t_p0 = time.perf_counter()
            x_arr = np.zeros((1, 12, res, res), dtype=np.float32)
            x_arr[0, 0] = (np.log(k_field) - 0.5) / 1.0
            x_arr[0, 1] = q_field / 500.0
            x_arr[0, 2] = x_norm
            x_arr[0, 3] = y_norm
            x_arr[0, 4] = 1.0  # left dirichlet
            x_arr[0, 5] = 1.0  # right dirichlet
            x_arr[0, 8] = 0.0  # bottom neumann
            x_arr[0, 9] = 0.0  # top neumann
            x_tensor = torch.from_numpy(x_arr)
            t_p1 = time.perf_counter()
            prep_times.append((t_p1 - t_p0) * 1000.0)

            # 2. Forward pass
            t_f0 = time.perf_counter()
            with torch.no_grad():
                if model_name == "U0_baseline_m3":
                    out = model_entry(x_tensor)
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    pred_norm = out.numpy()[:, 0]
                    pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
                    np.ones_like(pred_k) * 1.5
                    t_u1 = time.perf_counter()

                elif model_name == "U1_heteroscedastic":
                    mean_n, logvar_n = model_entry(x_tensor)
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    pred_norm = mean_n.numpy()[:, 0]
                    std_norm = torch.exp(0.5 * logvar_n).numpy()[:, 0]
                    pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
                    std_norm * self.temp_std
                    t_u1 = time.perf_counter()

                elif model_name == "U2_evidential":
                    gamma_n, v, alpha, beta = model_entry(x_tensor)
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    pred_norm = gamma_n.numpy()[:, 0]
                    pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
                    v_np = v.numpy()[:, 0]
                    alpha_np = alpha.numpy()[:, 0]
                    beta_np = beta.numpy()[:, 0]
                    denom = np.maximum(alpha_np - 1.0, 1e-4)
                    var_norm = (beta_np / denom) + (beta_np / (np.maximum(v_np, 1e-4) * denom))
                    np.sqrt(np.maximum(var_norm, 1e-8)) * self.temp_std
                    t_u1 = time.perf_counter()

                elif model_name.startswith("U3_mcdropout"):
                    u3, t_samples = model_entry
                    u3.train()
                    preds_list = []
                    for _ in range(t_samples):
                        preds_list.append(u3(x_tensor).numpy()[:, 0])
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    stack = np.stack(preds_list, axis=0)
                    mean_norm = np.mean(stack, axis=0)
                    std_norm = np.std(stack, axis=0, ddof=1)
                    pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
                    std_norm * self.temp_std
                    t_u1 = time.perf_counter()

                elif model_name == "U4_distilled_student":
                    mean_n, std_n = model_entry(x_tensor)
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    pred_norm = mean_n.numpy()[:, 0]
                    pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
                    std_n.numpy()[:, 0] * self.temp_std
                    t_u1 = time.perf_counter()

                elif model_name == "Ensemble5":
                    preds_list = []
                    for m in model_entry:
                        preds_list.append(m(x_tensor).numpy()[:, 0])
                    t_f1 = time.perf_counter()
                    t_u0 = time.perf_counter()
                    stack = np.stack(preds_list, axis=0)
                    mean_norm = np.mean(stack, axis=0)
                    std_norm = np.std(stack, axis=0, ddof=1)
                    pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
                    std_norm * self.temp_std
                    t_u1 = time.perf_counter()

            forward_times.append((t_f1 - t_f0) * 1000.0)
            unc_times.append((t_u1 - t_u0) * 1000.0)

            # 4. Physical residual computation
            t_ph0 = time.perf_counter()
            _ = compute_heat_balance(
                grid,
                pred_k[0],
                k_field,
                q_field,
                bcs,
            )
            t_ph1 = time.perf_counter()
            phys_times.append((t_ph1 - t_ph0) * 1000.0)

            t_end = time.perf_counter()
            total_times.append((t_end - t_start) * 1000.0)

        mean_total = float(np.mean(total_times))
        return {
            "prep_latency_ms": float(np.mean(prep_times)),
            "forward_latency_ms": float(np.mean(forward_times)),
            "unc_latency_ms": float(np.mean(unc_times)),
            "phys_latency_ms": float(np.mean(phys_times)),
            "total_latency_ms": mean_total,
            "std_total_latency_ms": float(np.std(total_times)),
            "throughput_samples_per_sec": float(1000.0 / mean_total) if mean_total > 0 else 0.0,
        }

    def benchmark_fvm_at_resolution(self, res: int) -> dict[str, float]:
        """Benchmark direct FVM PyAMG solver at given resolution."""
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
        bcs = make_dummy_bcs()
        k_field = np.ones((res, res), dtype=np.float32) * 2.0
        q_field = np.zeros((res, res), dtype=np.float32)

        # Warm-up
        for _ in range(self.n_warmup):
            _ = self.fvm_solver.solve(grid, k_field, q_field, bcs)

        times = []
        for _ in range(self.n_iters):
            t0 = time.perf_counter()
            _ = self.fvm_solver.solve(grid, k_field, q_field, bcs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_t = float(np.mean(times))
        return {
            "prep_latency_ms": 0.0,
            "forward_latency_ms": mean_t,
            "unc_latency_ms": 0.0,
            "phys_latency_ms": 0.0,
            "total_latency_ms": mean_t,
            "std_total_latency_ms": float(np.std(times)),
            "throughput_samples_per_sec": float(1000.0 / mean_t) if mean_t > 0 else 0.0,
        }

    def run_all_benchmarks(self) -> dict[str, Any]:
        resolutions = [32, 64, 128, 256]
        results: dict[str, Any] = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_warmup": self.n_warmup,
                "n_iters": self.n_iters,
                "profiler_instrumentation": "NONE (clean CPU timing, tracemalloc disabled)",
                "cpu_threads": 8,
            },
            "benchmarks": {},
            "speedup_vs_fvm": {},
            "speedup_vs_ensemble": {},
        }

        # 1. Benchmark FVM solver
        fvm_by_res: dict[str, Any] = {}
        for res in resolutions:
            logger.info(f"Benchmarking FVM Solver at {res}x{res}...")
            fvm_res = self.benchmark_fvm_at_resolution(res)
            fvm_by_res[f"res_{res}"] = fvm_res
            logger.info(f"FVM {res}x{res}: {fvm_res['total_latency_ms']:.2f} ms")
        results["benchmarks"]["FVM_PyAMG"] = fvm_by_res

        # 2. Benchmark Neural Operators
        for m_name in self.models:
            m_res_dict: dict[str, Any] = {}
            for res in resolutions:
                if m_name == "Ensemble5" and res != 32:
                    continue
                logger.info(f"Benchmarking {m_name} at {res}x{res}...")
                bench = self.benchmark_model_at_resolution(m_name, res)
                m_res_dict[f"res_{res}"] = bench
                logger.info(
                    f"{m_name} {res}x{res}: Total={bench['total_latency_ms']:.2f} ms "
                    f"(Fwd={bench['forward_latency_ms']:.2f} ms, Unc={bench['unc_latency_ms']:.2f} ms, "
                    f"Phys={bench['phys_latency_ms']:.2f} ms)"
                )
            results["benchmarks"][m_name] = m_res_dict

        # 3. Compute relative speedups
        for m_name in self.models:
            results["speedup_vs_fvm"][m_name] = {}
            for res in resolutions:
                res_key = f"res_{res}"
                if res_key in results["benchmarks"][m_name]:
                    m_lat = results["benchmarks"][m_name][res_key]["total_latency_ms"]
                    fvm_lat = results["benchmarks"]["FVM_PyAMG"][res_key]["total_latency_ms"]
                    speedup = float(fvm_lat / m_lat) if m_lat > 0 else 0.0
                    results["speedup_vs_fvm"][m_name][res_key] = speedup

            # Speedup vs Ensemble5 at 32x32
            if "res_32" in results["benchmarks"][m_name] and "Ensemble5" in results["benchmarks"]:
                m_lat = results["benchmarks"][m_name]["res_32"]["total_latency_ms"]
                ens_lat = results["benchmarks"]["Ensemble5"]["res_32"]["total_latency_ms"]
                speedup_ens = float(ens_lat / m_lat) if m_lat > 0 else 0.0
                results["speedup_vs_ensemble"][m_name] = speedup_ens

        out_json = Path("artifacts/reports/milestone_08d_benchmarks.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results to {out_json}")

        return results


def main() -> None:
    benchmarker = LatencyBenchmarker()
    benchmarker.run_all_benchmarks()


if __name__ == "__main__":
    main()
