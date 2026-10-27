"""Diagnostic script for loss scaling, gradient norms, and cosine similarities."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.models.cnn import CNNBaseline2D
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.loss import RelativeL2Loss
from neural_process_twin.training.physics_loss import (
    BoundaryConditionLoss,
    DimensionlessPDELoss,
)


def compute_gradient_vector(loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Compute gradients with respect to model parameters and flatten into a real 1D vector."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            if g.is_complex():
                grads.append(g.real.flatten().double())
                grads.append(g.imag.flatten().double())
            else:
                grads.append(g.flatten().double())
        else:
            if p.is_complex():
                grads.append(torch.zeros_like(p.real).flatten().double())
                grads.append(torch.zeros_like(p.imag).flatten().double())
            else:
                grads.append(torch.zeros_like(p).flatten().double())
    model.zero_grad()
    return torch.cat(grads)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D real gradient vectors."""
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_val = torch.dot(v1, v2) / (n1 * n2)
    return float(torch.clamp(cos_val, -1.0, 1.0).item())


def run_diagnostics(dataset_path: str = "artifacts/data/scaled_dataset.h5") -> dict:
    dataset = HeatFieldDataset(
        file_path=dataset_path,
        split="train",
        include_coords=True,
        include_bcs=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(loader))

    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=dataset.nx, ny=dataset.ny)
    normalizer = dataset.normalizer
    assert normalizer is not None and normalizer.is_fitted

    t_mean = float(normalizer.t_stats.mean)
    t_std = float(normalizer.t_stats.std)

    data_loss_fn = RelativeL2Loss()
    pde_loss_full = DimensionlessPDELoss(grid=grid, residual_type="full", q_floor=1.0)
    pde_loss_interior = DimensionlessPDELoss(grid=grid, residual_type="interior", q_floor=1.0)
    bc_loss_fn = BoundaryConditionLoss(grid=grid, delta_t_char=t_std)

    x = batch["x"]
    y = batch["y"]

    results = {}

    architectures = [
        ("cnn2d", CNNBaseline2D(in_channels=12, out_channels=1, width=32, depth=4)),
        (
            "fno2d",
            FourierNeuralOperator2D(
                in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
            ),
        ),
    ]

    for arch_name, model in architectures:
        model.eval()  # deterministic forward pass
        torch.manual_seed(42)

        y_pred = model(x)
        t_pred_k = y_pred * t_std + t_mean

        # Extract physical fields from canonical 12-channel input tensor
        k_raw = x[:, 0:1] * normalizer.k_stats.std + normalizer.k_stats.mean
        q_raw = x[:, 1:2] * normalizer.q_stats.std + normalizer.q_stats.mean
        bc_channels = x[:, 4:12]

        # Unweighted loss evaluations
        l_data = data_loss_fn(y_pred, y)
        l_pde_full = pde_loss_full(t_pred_k, k_raw, q_raw, bc_channels, t_mean=t_mean, t_std=t_std)
        l_pde_interior = pde_loss_interior(
            t_pred_k, k_raw, q_raw, bc_channels, t_mean=t_mean, t_std=t_std
        )
        l_bc = bc_loss_fn(t_pred_k, k_raw, bc_channels, t_mean=t_mean, t_std=t_std)

        # Gradient vectors
        g_data = compute_gradient_vector(l_data, model)
        g_pde_full = compute_gradient_vector(l_pde_full, model)
        g_pde_interior = compute_gradient_vector(l_pde_interior, model)
        g_bc = compute_gradient_vector(l_bc, model)

        norm_data = float(torch.linalg.norm(g_data))
        norm_pde_full = float(torch.linalg.norm(g_pde_full))
        norm_pde_int = float(torch.linalg.norm(g_pde_interior))
        norm_bc = float(torch.linalg.norm(g_bc))

        cos_data_pde_full = cosine_similarity(g_data, g_pde_full)
        cos_data_pde_int = cosine_similarity(g_data, g_pde_interior)
        cos_data_bc = cosine_similarity(g_data, g_bc)
        cos_pde_int_bc = cosine_similarity(g_pde_interior, g_bc)

        # Candidate weight calibrations
        candidate_weights = [
            {"pde_w": 0.001, "bc_w": 0.0, "type": "pde_only"},
            {"pde_w": 0.01, "bc_w": 0.0, "type": "pde_only"},
            {"pde_w": 0.05, "bc_w": 0.0, "type": "pde_only"},
            {"pde_w": 0.1, "bc_w": 0.0, "type": "pde_only"},
            {"pde_w": 0.01, "bc_w": 0.05, "type": "pde_bc"},
            {"pde_w": 0.05, "bc_w": 0.05, "type": "pde_bc"},
            {"pde_w": 0.1, "bc_w": 0.1, "type": "pde_bc"},
        ]

        calibration_grid = []
        for cand in candidate_weights:
            pw = cand["pde_w"]
            bw = cand["bc_w"]
            g_pde_vec = g_pde_interior if bw > 0 else g_pde_full
            g_tot = g_data + pw * g_pde_vec + bw * g_bc

            norm_tot_preclip = float(torch.linalg.norm(g_tot))
            norm_pde_weighted = float(pw * torch.linalg.norm(g_pde_vec))
            norm_bc_weighted = float(bw * torch.linalg.norm(g_bc))

            # Clipping simulation with max_norm=1.0
            clip_scale = min(1.0, 1.0 / (norm_tot_preclip + 1e-6))
            norm_tot_postclip = norm_tot_preclip * clip_scale

            calibration_grid.append(
                {
                    "pde_weight": pw,
                    "bc_weight": bw,
                    "loss_total": float(
                        l_data.item()
                        + pw * (l_pde_interior.item() if bw > 0 else l_pde_full.item())
                        + bw * l_bc.item()
                    ),
                    "grad_data_norm": norm_data,
                    "grad_pde_weighted_norm": norm_pde_weighted,
                    "grad_bc_weighted_norm": norm_bc_weighted,
                    "pde_to_data_ratio": norm_pde_weighted / (norm_data + 1e-12),
                    "bc_to_data_ratio": norm_bc_weighted / (norm_data + 1e-12),
                    "grad_total_preclip": norm_tot_preclip,
                    "grad_total_postclip": norm_tot_postclip,
                }
            )

        results[arch_name] = {
            "unweighted_losses": {
                "loss_data": float(l_data.item()),
                "loss_pde_full": float(l_pde_full.item()),
                "loss_pde_interior": float(l_pde_interior.item()),
                "loss_bc": float(l_bc.item()),
            },
            "unweighted_gradient_norms": {
                "norm_data": norm_data,
                "norm_pde_full": norm_pde_full,
                "norm_pde_interior": norm_pde_int,
                "norm_bc": norm_bc,
            },
            "pairwise_cosine_similarities": {
                "cos_data_pde_full": cos_data_pde_full,
                "cos_data_pde_interior": cos_data_pde_int,
                "cos_data_bc": cos_data_bc,
                "cos_pde_interior_bc": cos_pde_int_bc,
            },
            "calibration_grid": calibration_grid,
        }

    return results


if __name__ == "__main__":
    rep = run_diagnostics()
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    out_file = Path("artifacts/reports/loss_scaling_calibration.json")
    with open(out_file, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Diagnostics written to {out_file}")
    print(json.dumps(rep, indent=2))
