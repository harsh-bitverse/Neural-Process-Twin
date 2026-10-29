"""Milestone 08G Phases 5 & 6: Selective Physics Correction and Risk-Latency Optimization.

Benchmarks 7 routing policies across multi-resolution grids (32, 64, 128, 256) and subgroups (IID, OOD):
- P0: Direct FVM Solver (Ground-Truth reference, 100% escalation)
- P1: Unchecked A3 (Surrogate alone, 0% escalation)
- P2: Unchecked A4 (A3 + 2 iterations Full-grid Jacobi smoothing)
- P3: A3 + Selective Boundary Jacobi (2 iterations on boundary ring cells only)
- P4: Single-Pass Learned Error Gated Routing (A3 + U_res <= tau)
- P5: Single-Pass Physics-Residual Gated Routing (A3 + U_phys_res <= tau)
- P6: Adaptive 3-Tier Routing (A3 -> Selective Boundary Jacobi -> FVM Fallback)

Computes for each policy and subgroup:
- Total queries, accepted queries, escalated queries
- Neural acceptance and escalation rates
- Unsafe count (>10 K and >5 K)
- False acceptance rate (FAR) over all queries with Wilson 95% CI
- Conditional failure rate (CFR) among accepted predictions with Wilson 95% CI
- Accepted case RMSE and MaxAE
- End-to-end latency percentiles (p50, p90, p99)
- Speedup vs direct FVM

Outputs: artifacts/reports/milestone_08g_routing_economics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.refined_fno import RefinedFNO2D
from neural_process_twin.solvers.assembly import LinearSystem, assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.physics_smoother import JacobiPhysicsSmoother

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_08g_routing")


def compute_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute exact Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total) / denom
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
    bc_grp = grp["boundary_conditions"]
    l_type = bc_grp["left_type"][idx]
    r_type = bc_grp["right_type"][idx]
    b_type = bc_grp["bottom_type"][idx]
    t_type = bc_grp["top_type"][idx]

    l_val = bc_grp["left_value"][idx]
    r_val = bc_grp["right_value"][idx]
    b_val = bc_grp["bottom_value"][idx]
    t_val = bc_grp["top_value"][idx]

    def _to_str(val: Any) -> str:
        return val.decode("utf-8").lower() if isinstance(val, bytes) else str(val).lower()

    def _side_bc(side: BoundarySide, btype_val: Any, val_arr: np.ndarray) -> BoundaryCondition:
        t_str = _to_str(btype_val)
        b_type = BoundaryType.DIRICHLET if t_str == "dirichlet" else BoundaryType.NEUMANN
        if np.allclose(val_arr, val_arr[0]):
            return BoundaryCondition(side=side, bc_type=b_type, value=float(val_arr[0]))
        return BoundaryCondition(side=side, bc_type=b_type, value=val_arr.copy())

    return BoundaryConditions(
        left=_side_bc(BoundarySide.LEFT, l_type, l_val),
        right=_side_bc(BoundarySide.RIGHT, r_type, r_val),
        bottom=_side_bc(BoundarySide.BOTTOM, b_type, b_val),
        top=_side_bc(BoundarySide.TOP, t_type, t_val),
    )


class A3ResidualPredictor(nn.Module):
    def __init__(self, in_channels: int = 13, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Softplus(),
        )

    def forward(self, x_in: torch.Tensor, t_pred_norm: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_in, t_pred_norm], dim=1)
        return self.net(feat)


def apply_selective_boundary_jacobi(
    t_init: np.ndarray,
    system: LinearSystem,
    grid: Grid2D,
    ring_width: int = 2,
    omega: float = 0.67,
    num_iters: int = 2,
) -> np.ndarray:
    """Apply damped Jacobi relaxation selectively to boundary ring cells."""
    ny, nx = grid.ny, grid.nx
    t_flat = t_init.ravel().copy()

    # Identify boundary ring cell indices
    bnd_mask = np.zeros((ny, nx), dtype=bool)
    rw = min(ring_width, ny // 2)
    bnd_mask[:rw, :] = True
    bnd_mask[-rw:, :] = True
    bnd_mask[:, :rw] = True
    bnd_mask[:, -rw:] = True
    bnd_indices = np.where(bnd_mask.ravel())[0]

    # Pre-extract diagonal conductances for boundary indices
    a_csr = system.A.tocsr()
    diag_a = a_csr.diagonal()[bnd_indices]
    inv_diag = np.where(np.abs(diag_a) > 1e-12, 1.0 / diag_a, 0.0)

    # Perform selective iterations
    for _ in range(num_iters):
        # r_bnd = b[bnd] - A[bnd, :] * t_flat
        r_bnd = system.b[bnd_indices] - a_csr[bnd_indices, :].dot(t_flat)
        t_flat[bnd_indices] += omega * inv_diag * r_bnd

    return t_flat.reshape((ny, nx))


def solve_direct_fvm(system: LinearSystem, grid: Grid2D) -> np.ndarray:
    """Solve steady heat conduction exactly using FVM sparse LU solver."""
    t_flat = spla.spsolve(system.A.tocsc(), system.b)
    return t_flat.reshape((grid.ny, grid.nx))


def benchmark_routing(
    dataset_path: Path,
    device: torch.device,
    resolutions: list[int] | None = None,
) -> dict[str, Any]:
    # Load A3
    if resolutions is None:
        resolutions = [32, 64, 128, 256]
    a3_path = Path("artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt")
    a3_ckpt = torch.load(a3_path, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(a3_ckpt["normalizer"])
    log_k_mean = float(a3_ckpt["log_k_mean"])
    log_k_std = float(a3_ckpt["log_k_std"])

    a3_model = RefinedFNO2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        refinement_hidden=32,
        refinement_layers=3,
        activation="gelu",
    ).to(device)
    a3_model.load_state_dict(a3_ckpt.get("model_state_dict", a3_ckpt))
    a3_model.eval()

    # Load U_res
    res_path = Path("artifacts/checkpoints/milestone_08g/A3_residual_predictor/best_model.pt")
    res_ckpt = torch.load(res_path, map_location=device, weights_only=False)
    res_pred = A3ResidualPredictor(in_channels=13, hidden_dim=32).to(device)
    res_pred.load_state_dict(res_ckpt["model_state_dict"])
    res_pred.eval()

    full_smoother = JacobiPhysicsSmoother(omega=0.67, max_iters=2)

    # Thresholds calibrated for safety
    # MaxAE tolerance is 10.0 K; safety buffer targets error <= 8.5 K
    tau_res_safe = 8.5
    tau_phys_res_safe = 0.05  # 5% relative algebraic residual
    tau_res_adapt_1 = 6.0  # Accept immediately
    tau_res_adapt_2 = 12.0  # Attempt selective Jacobi; if > 12.0 escalate to FVM

    all_policies = [
        "P0_direct_fvm",
        "P1_unchecked_a3",
        "P2_unchecked_a4_full_jacobi",
        "P3_selective_bnd_jacobi",
        "P4_learned_gated_a3",
        "P5_physics_gated_a3",
        "P6_adaptive_3tier",
    ]

    results: dict[str, Any] = {
        "milestone": "08G",
        "phase": 6,
        "title": "Routing Economics and Statistical Safety Evaluation",
        "provenance": {
            "a3_sha256": compute_file_sha256(a3_path),
            "res_predictor_sha256": compute_file_sha256(res_path),
        },
        "thresholds": {
            "tau_res_safe_k": tau_res_safe,
            "tau_phys_res_safe": tau_phys_res_safe,
            "tau_res_adapt_1_k": tau_res_adapt_1,
            "tau_res_adapt_2_k": tau_res_adapt_2,
        },
        "resolutions": {},
    }

    with h5py.File(dataset_path, "r") as f:
        for res in resolutions:
            results["resolutions"][f"{res}x{res}"] = {}
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

            for split in ["test_iid", "test_ood"]:
                logger.info(f"Running routing benchmark on {res}x{res} {split}...")
                grp = f[f"res_{res}/{split}"]
                n_samples = grp["temperature"].shape[0]
                k_all = grp["k"][:]
                q_all = grp["q"][:]
                t_all = grp["temperature"][:]
                bcs_all = [parse_bcs_row(grp, i) for i in range(n_samples)]

                # Track per-policy runs
                policy_stats: dict[str, Any] = {
                    p: {
                        "latencies": [],
                        "accepted_errors": [],
                        "accepted_rmses": [],
                        "escalated_count": 0,
                        "accepted_count": 0,
                        "unsafe_10k_count": 0,
                        "unsafe_5k_count": 0,
                    }
                    for p in all_policies
                }

                for i in range(n_samples):
                    k_f = k_all[i]
                    q_f = q_all[i]
                    t_true = t_all[i]
                    bcs = bcs_all[i]

                    # 1. P0: Direct FVM
                    t0 = time.perf_counter()
                    sys = assemble_steady_heat_system(grid, k_f, q_f, bcs)
                    t_fvm = solve_direct_fvm(sys, grid)
                    t_fvm_end = time.perf_counter()
                    fvm_latency = (t_fvm_end - t0) * 1000.0

                    policy_stats["P0_direct_fvm"]["latencies"].append(fvm_latency)
                    err_fvm = np.abs(t_fvm - t_true)
                    policy_stats["P0_direct_fvm"]["accepted_count"] += 1
                    policy_stats["P0_direct_fvm"]["accepted_errors"].append(float(np.max(err_fvm)))
                    policy_stats["P0_direct_fvm"]["accepted_rmses"].append(
                        float(np.sqrt(np.mean(err_fvm**2)))
                    )

                    # Neural Preprocessing & Inference (shared by neural policies)
                    t_prep_start = time.perf_counter()
                    x_in = encode_continuous_physical_inputs(
                        k_field=k_f,
                        q_field=q_f,
                        bcs=bcs,
                        grid=grid,
                        normalizer=normalizer,
                        use_log_k=True,
                        log_k_mean=log_k_mean,
                        log_k_std=log_k_std,
                    )
                    x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out_norm = a3_model(x_tensor)
                        t_a3 = normalizer.inverse_transform_temperature(
                            out_norm.cpu().numpy()[0, 0]
                        )
                    t_neural_end = time.perf_counter()
                    a3_latency = (t_neural_end - t_prep_start) * 1000.0

                    err_a3 = np.abs(t_a3 - t_true)
                    maxae_a3 = float(np.max(err_a3))
                    rmse_a3 = float(np.sqrt(np.mean(err_a3**2)))

                    # 2. P1: Unchecked A3
                    policy_stats["P1_unchecked_a3"]["latencies"].append(a3_latency)
                    policy_stats["P1_unchecked_a3"]["accepted_count"] += 1
                    policy_stats["P1_unchecked_a3"]["accepted_errors"].append(maxae_a3)
                    policy_stats["P1_unchecked_a3"]["accepted_rmses"].append(rmse_a3)
                    if maxae_a3 > 10.0:
                        policy_stats["P1_unchecked_a3"]["unsafe_10k_count"] += 1
                    if maxae_a3 > 5.0:
                        policy_stats["P1_unchecked_a3"]["unsafe_5k_count"] += 1

                    # 3. P2: Unchecked A4 (Full Jacobi)
                    t_p2_start = time.perf_counter()
                    t_a4 = full_smoother.smooth(
                        t_init=t_a3, grid=grid, k_field=k_f, q_field=q_f, bcs=bcs, num_iters=2
                    )
                    t_p2_end = time.perf_counter()
                    p2_latency = a3_latency + (t_p2_end - t_p2_start) * 1000.0
                    err_a4 = np.abs(t_a4 - t_true)
                    maxae_a4 = float(np.max(err_a4))
                    policy_stats["P2_unchecked_a4_full_jacobi"]["latencies"].append(p2_latency)
                    policy_stats["P2_unchecked_a4_full_jacobi"]["accepted_count"] += 1
                    policy_stats["P2_unchecked_a4_full_jacobi"]["accepted_errors"].append(maxae_a4)
                    policy_stats["P2_unchecked_a4_full_jacobi"]["accepted_rmses"].append(
                        float(np.sqrt(np.mean(err_a4**2)))
                    )
                    if maxae_a4 > 10.0:
                        policy_stats["P2_unchecked_a4_full_jacobi"]["unsafe_10k_count"] += 1
                    if maxae_a4 > 5.0:
                        policy_stats["P2_unchecked_a4_full_jacobi"]["unsafe_5k_count"] += 1

                    # 4. P3: Selective Boundary Jacobi
                    t_p3_start = time.perf_counter()
                    t_sel = apply_selective_boundary_jacobi(
                        t_init=t_a3, system=sys, grid=grid, ring_width=2, omega=0.67, num_iters=2
                    )
                    t_p3_end = time.perf_counter()
                    p3_latency = a3_latency + (t_p3_end - t_p3_start) * 1000.0
                    err_sel = np.abs(t_sel - t_true)
                    maxae_sel = float(np.max(err_sel))
                    policy_stats["P3_selective_bnd_jacobi"]["latencies"].append(p3_latency)
                    policy_stats["P3_selective_bnd_jacobi"]["accepted_count"] += 1
                    policy_stats["P3_selective_bnd_jacobi"]["accepted_errors"].append(maxae_sel)
                    policy_stats["P3_selective_bnd_jacobi"]["accepted_rmses"].append(
                        float(np.sqrt(np.mean(err_sel**2)))
                    )
                    if maxae_sel > 10.0:
                        policy_stats["P3_selective_bnd_jacobi"]["unsafe_10k_count"] += 1
                    if maxae_sel > 5.0:
                        policy_stats["P3_selective_bnd_jacobi"]["unsafe_5k_count"] += 1

                    # Compute uncertainty indicators
                    with torch.no_grad():
                        e_hat = res_pred(x_tensor, out_norm).cpu().numpy()[0, 0]
                        u_res = float(np.max(e_hat))

                    # Linear system residual
                    r = sys.A.dot(t_a3.ravel()) - sys.b
                    b_norm = np.linalg.norm(sys.b)
                    u_phys_res = float(np.linalg.norm(r) / (b_norm if b_norm > 1e-12 else 1.0))

                    # 5. P4: Single-Pass Learned Error Gated Routing
                    if u_res <= tau_res_safe:
                        policy_stats["P4_learned_gated_a3"]["accepted_count"] += 1
                        policy_stats["P4_learned_gated_a3"]["accepted_errors"].append(maxae_a3)
                        policy_stats["P4_learned_gated_a3"]["accepted_rmses"].append(rmse_a3)
                        policy_stats["P4_learned_gated_a3"]["latencies"].append(a3_latency)
                        if maxae_a3 > 10.0:
                            policy_stats["P4_learned_gated_a3"]["unsafe_10k_count"] += 1
                        if maxae_a3 > 5.0:
                            policy_stats["P4_learned_gated_a3"]["unsafe_5k_count"] += 1
                    else:
                        policy_stats["P4_learned_gated_a3"]["escalated_count"] += 1
                        policy_stats["P4_learned_gated_a3"]["latencies"].append(
                            a3_latency + fvm_latency
                        )

                    # 6. P5: Single-Pass Physics-Residual Gated Routing
                    if u_phys_res <= tau_phys_res_safe:
                        policy_stats["P5_physics_gated_a3"]["accepted_count"] += 1
                        policy_stats["P5_physics_gated_a3"]["accepted_errors"].append(maxae_a3)
                        policy_stats["P5_physics_gated_a3"]["accepted_rmses"].append(rmse_a3)
                        policy_stats["P5_physics_gated_a3"]["latencies"].append(a3_latency)
                        if maxae_a3 > 10.0:
                            policy_stats["P5_physics_gated_a3"]["unsafe_10k_count"] += 1
                        if maxae_a3 > 5.0:
                            policy_stats["P5_physics_gated_a3"]["unsafe_5k_count"] += 1
                    else:
                        policy_stats["P5_physics_gated_a3"]["escalated_count"] += 1
                        policy_stats["P5_physics_gated_a3"]["latencies"].append(
                            a3_latency + fvm_latency
                        )

                    # 7. P6: Adaptive 3-Tier Routing
                    if u_res <= tau_res_adapt_1:
                        # Tier 1: Fast direct surrogate acceptance
                        policy_stats["P6_adaptive_3tier"]["accepted_count"] += 1
                        policy_stats["P6_adaptive_3tier"]["accepted_errors"].append(maxae_a3)
                        policy_stats["P6_adaptive_3tier"]["accepted_rmses"].append(rmse_a3)
                        policy_stats["P6_adaptive_3tier"]["latencies"].append(a3_latency)
                        if maxae_a3 > 10.0:
                            policy_stats["P6_adaptive_3tier"]["unsafe_10k_count"] += 1
                        if maxae_a3 > 5.0:
                            policy_stats["P6_adaptive_3tier"]["unsafe_5k_count"] += 1
                    elif u_res <= tau_res_adapt_2:
                        # Tier 2: Apply fast selective boundary Jacobi
                        policy_stats["P6_adaptive_3tier"]["accepted_count"] += 1
                        policy_stats["P6_adaptive_3tier"]["accepted_errors"].append(maxae_sel)
                        policy_stats["P6_adaptive_3tier"]["accepted_rmses"].append(
                            float(np.sqrt(np.mean(err_sel**2)))
                        )
                        policy_stats["P6_adaptive_3tier"]["latencies"].append(p3_latency)
                        if maxae_sel > 10.0:
                            policy_stats["P6_adaptive_3tier"]["unsafe_10k_count"] += 1
                        if maxae_sel > 5.0:
                            policy_stats["P6_adaptive_3tier"]["unsafe_5k_count"] += 1
                    else:
                        # Tier 3: Escalate to FVM
                        policy_stats["P6_adaptive_3tier"]["escalated_count"] += 1
                        policy_stats["P6_adaptive_3tier"]["latencies"].append(
                            a3_latency + fvm_latency
                        )

                # Aggregate results for split
                fvm_p50 = float(np.median(policy_stats["P0_direct_fvm"]["latencies"]))
                split_summary: dict[str, Any] = {"n_samples": n_samples, "policies": {}}

                for pname, pdict in policy_stats.items():
                    n_acc = pdict["accepted_count"]
                    n_esc = pdict["escalated_count"]
                    u_10k = pdict["unsafe_10k_count"]
                    u_5k = pdict["unsafe_5k_count"]

                    far_10k = float(u_10k / n_samples)
                    far_10k_ci = list(wilson_score_interval(u_10k, n_samples))

                    cfr_10k = float(u_10k / n_acc) if n_acc > 0 else 0.0
                    cfr_10k_ci = (
                        list(wilson_score_interval(u_10k, n_acc)) if n_acc > 0 else [0.0, 0.0]
                    )

                    far_5k = float(u_5k / n_samples)
                    cfr_5k = float(u_5k / n_acc) if n_acc > 0 else 0.0

                    acc_rmse = float(np.mean(pdict["accepted_rmses"])) if n_acc > 0 else 0.0
                    acc_maxae = float(np.mean(pdict["accepted_errors"])) if n_acc > 0 else 0.0

                    lats = pdict["latencies"]
                    p50 = float(np.percentile(lats, 50))
                    p90 = float(np.percentile(lats, 90))
                    p99 = float(np.percentile(lats, 99))
                    speedup = float(fvm_p50 / p50) if p50 > 0 else 1.0

                    split_summary["policies"][pname] = {
                        "accepted_count": n_acc,
                        "escalated_count": n_esc,
                        "neural_acceptance_rate": float(n_acc / n_samples),
                        "escalation_rate": float(n_esc / n_samples),
                        "unsafe_10k_count": u_10k,
                        "false_accept_rate_10k": far_10k,
                        "false_accept_rate_10k_ci95": far_10k_ci,
                        "conditional_failure_rate_10k": cfr_10k,
                        "conditional_failure_rate_10k_ci95": cfr_10k_ci,
                        "unsafe_5k_count": u_5k,
                        "false_accept_rate_5k": far_5k,
                        "conditional_failure_rate_5k": cfr_5k,
                        "accepted_rmse_mean": acc_rmse,
                        "accepted_maxae_mean": acc_maxae,
                        "latency_p50_ms": p50,
                        "latency_p90_ms": p90,
                        "latency_p99_ms": p99,
                        "speedup_vs_fvm": speedup,
                    }

                    logger.info(
                        f"{res:3d}x{res:3d} {split:8s} | {pname:28s} | "
                        f"Acc: {n_acc:3d}/{n_samples:3d} ({n_acc / n_samples * 100:4.1f}%) | "
                        f"Unsafe10K: {u_10k:2d} | CFR: {cfr_10k * 100:4.1f}% | "
                        f"p50: {p50:6.1f} ms | Speedup: {speedup:4.1f}x"
                    )

                results["resolutions"][f"{res}x{res}"][split] = split_summary

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 08G Selective Routing Benchmarks")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/multires_dataset.h5"),
        help="Path to multires dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_08g_routing_economics.json"),
        help="Path to output routing economics report",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    routing_results = benchmark_routing(args.dataset, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(routing_results, f, indent=2)
    logger.info(f"Routing benchmark report saved to {args.output}")


if __name__ == "__main__":
    main()
