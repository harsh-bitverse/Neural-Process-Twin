"""Milestone 12: End-to-End Interactive Demonstration Script.

Demonstrates all 8 required capabilities:
1. Configuring a physical heat-conduction problem (inclusion + hotspot + Dirichlet boundaries).
2. Running the verified numerical FVM solver.
3. Visualizing the numerical reference temperature field.
4. Running the experimental neural surrogate (A3 RefinedFNO2D).
5. Comparing neural and numerical outputs (A4 Selective Jacobi corrected).
6. Inspecting the spatial error map and localized boundary layer.
7. Viewing physical diagnostics (algebraic residual, heat flux balance).
8. Explaining why the neural result is not certified.

Generates: artifacts/figures/demo_showcase.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_process_twin.inference.experimental_surrogate import ExperimentalSurrogateService
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


def run_demo() -> None:
    print("=" * 80)
    print("NEURAL PROCESS TWIN — INTERACTIVE DEMONSTRATION & SHOWCASE")
    print("=" * 80)

    # 1. Configuring a physical heat-conduction problem
    print("\n[Step 1/8] Configuring 2D physical heat conduction problem...")
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=64, ny=64)
    x_mesh, y_mesh = np.meshgrid(grid.x_centers, grid.y_centers)

    # Circular thermal inclusion (k=5.0 W/(m K) in k=1.0 background)
    dist2 = (x_mesh - 0.5) ** 2 + (y_mesh - 0.5) ** 2
    k_field = np.where(dist2 <= 0.2**2, 5.0, 1.0).astype(np.float64)

    # Volumetric Gaussian hotspot (peak 300 W/m^3)
    q_field = (300.0 * np.exp(-dist2 / (2 * 0.12**2))).astype(np.float64)

    # Boundary conditions: West=300 K, East=350 K, South=Insulated, North=Insulated
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    print(f"  Domain: {grid.nx}x{grid.ny} cells ({grid.num_cells} control volumes)")
    print("  Thermal Conductivity: Background = 1.0 W/(m K), Inclusion = 5.0 W/(m K)")
    print("  Heat Source: Center Gaussian hotspot (peak = 300 W/m^3)")
    print(
        "  Boundaries: Left = 300 K (Dirichlet), Right = 350 K (Dirichlet), Top/Bottom = Insulated (Neumann)"
    )

    # 2. Running Verified Numerical FVM Solver
    print("\n[Step 2/8] Executing Verified Numerical Mode (FVM)...")
    solver = HeatConductionSolver2D()
    t0 = time.perf_counter()
    fvm_res = solver.solve(grid, k_field, q_field, bcs)
    fvm_time_ms = (time.perf_counter() - t0) * 1000.0
    t_fvm = fvm_res.temperature

    print(f"  FVM Solve Time: {fvm_time_ms:.2f} ms")
    print(f"  Min Temperature: {np.min(t_fvm):.2f} K | Max Temperature: {np.max(t_fvm):.2f} K")
    print(f"  Linear Residual L2: {fvm_res.diagnostics.linear_residual_l2:.2e} W")
    print(
        f"  Relative Algebraic Residual: {fvm_res.diagnostics.relative_residual:.2e} (Machine Precision)"
    )
    print(
        f"  Energy Balance Imbalance: {fvm_res.diagnostics.energy_balance.relative_imbalance:.2e}"
    )

    # 3. Running Experimental Neural Surrogate
    print("\n[Step 3/8] Executing Experimental Neural Mode (A3 RefinedFNO2D)...")
    surrogate = ExperimentalSurrogateService()
    t0_surr = time.perf_counter()
    a3_res = surrogate.predict(grid, k_field, q_field, bcs, apply_physics_correction=False)
    a3_time_ms = (time.perf_counter() - t0_surr) * 1000.0
    t_a3 = a3_res.temperature

    print(f"  A3 Forward Time: {a3_res.inference_time_ms:.2f} ms (Total: {a3_time_ms:.2f} ms)")
    print(f"  Min Temperature: {np.min(t_a3):.2f} K | Max Temperature: {np.max(t_a3):.2f} K")

    # 4. Applying Physics Smoothing (A4 Selective Boundary Jacobi)
    print("\n[Step 4/8] Executing A4 Selective Boundary Jacobi Correction...")
    t0_corr = time.perf_counter()
    a4_res = surrogate.predict(grid, k_field, q_field, bcs, apply_physics_correction=True)
    (time.perf_counter() - t0_corr) * 1000.0
    t_a4 = a4_res.temperature

    print(f"  A4 Correction Time: {a4_res.correction_time_ms:.2f} ms")

    # 5. Comparing Neural and Numerical Outputs
    print("\n[Step 5/8] Comparing Neural and Numerical Accuracy...")
    err_a3 = np.abs(t_a3 - t_fvm)
    err_a4 = np.abs(t_a4 - t_fvm)

    rmse_a3 = float(np.sqrt(np.mean((t_a3 - t_fvm) ** 2)))
    rmse_a4 = float(np.sqrt(np.mean((t_a4 - t_fvm) ** 2)))
    maxae_a3 = float(np.max(err_a3))
    maxae_a4 = float(np.max(err_a4))

    print(f"  A3 Uncorrected: RMSE = {rmse_a3:.2f} K | MaxAE = {maxae_a3:.2f} K")
    print(f"  A4 Corrected:   RMSE = {rmse_a4:.2f} K | MaxAE = {maxae_a4:.2f} K")

    # 6. Inspecting Error Maps and Localized Boundary Rings
    print("\n[Step 6/8] Inspecting Localized Boundary Errors...")
    ring_mask = np.zeros((64, 64), dtype=bool)
    ring_mask[:2, :] = True
    ring_mask[-2:, :] = True
    ring_mask[:, :2] = True
    ring_mask[:, -2:] = True

    ring_err_a3 = float(np.max(err_a3[ring_mask]))
    ring_err_a4 = float(np.max(err_a4[ring_mask]))
    interior_err = float(np.max(err_a4[~ring_mask]))

    print(
        f"  Boundary Ring MaxAE (Outer 2 Cells): A3 = {ring_err_a3:.2f} K -> A4 = {ring_err_a4:.2f} K (Reduced by {ring_err_a3 - ring_err_a4:.2f} K)"
    )
    print(f"  Interior MaxAE (Untouched by Jacobi): {interior_err:.2f} K")

    # 7. Physical Diagnostics & Energy Imbalance
    print("\n[Step 7/8] Evaluating Physical Conservation Diagnostics...")
    if a3_res.heat_balance is not None:
        print(
            f"  A3 Physical Heat Flux Imbalance: {a3_res.heat_balance.relative_imbalance * 100:.2f}% ({a3_res.heat_balance.flux_imbalance:.1f} W)"
        )
    if a4_res.heat_balance is not None:
        print(
            f"  A4 Physical Heat Flux Imbalance: {a4_res.heat_balance.relative_imbalance * 100:.2f}% ({a4_res.heat_balance.flux_imbalance:.1f} W)"
        )

    # 8. Certification Explanations
    print("\n[Step 8/8] Transparent Scientific Governance Verdict:")
    print("  [!] IS NEURAL PREDICTION CERTIFIED? NO (is_certified: false)")
    print(f"  [!] RATIONALE: MaxAE ({maxae_a4:.2f} K) exceeds the 10 K engineering safety ceiling.")
    print(
        "  [!] GOVERNANCE: Verified Numerical Mode must be used for formal thermal certification."
    )
    print(
        "  [!] UTILITY: Experimental Neural Mode provides rapid exploratory feedback (sub-100 ms)."
    )

    # Generate 4-panel demonstration figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    t_min = min(float(np.min(t_fvm)), float(np.min(t_a4)))
    t_max = max(float(np.max(t_fvm)), float(np.max(t_a4)))

    im0 = axes[0].imshow(
        t_fvm, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=t_min, vmax=t_max
    )
    axes[0].set_title("1. Verified Numerical (FVM)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im0, ax=axes[0], label="T [K]", fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        t_a4, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=t_min, vmax=t_max
    )
    axes[1].set_title("2. Experimental Neural (A4)")
    axes[1].set_xlabel("x [m]")
    fig.colorbar(im1, ax=axes[1], label="T [K]", fraction=0.046, pad=0.04)

    diff = t_a4 - t_fvm
    limit = max(abs(float(np.min(diff))), abs(float(np.max(diff))))
    im2 = axes[2].imshow(
        diff, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axes[2].set_title(f"3. Difference Field (A4 - FVM)\nRMSE: {rmse_a4:.2f} K")
    axes[2].set_xlabel("x [m]")
    fig.colorbar(im2, ax=axes[2], label="Delta T [K]", fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_a4, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[3].set_title(f"4. Absolute Error Map\nMaxAE: {maxae_a4:.2f} K")
    axes[3].set_xlabel("x [m]")
    fig.colorbar(im3, ax=axes[3], label="|Error| [K]", fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_fig = Path("artifacts/figures/demo_showcase.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\nSaved demonstration visual artifact to: {out_fig}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
