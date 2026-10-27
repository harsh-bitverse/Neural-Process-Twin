"""Uncertainty-quantified Fourier Neural Operator architectures for Milestone 08D.

Implements controlled efficient uncertainty variants:
- U1: HeteroscedasticFNO2D (Single-pass heteroscedastic aleatoric uncertainty)
- U2: EvidentialFNO2D (Deep Evidential Regression with Normal-Inverse-Gamma prior)
- U3: MCDropoutFNO2D (Spatial Monte Carlo Dropout for stochastic epistemic uncertainty)
- U4: DistilledStudentFNO2D (Single-pass student distilled from ensemble epistemic uncertainty)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.fno2d import SpectralConv2D


class HeteroscedasticFNO2D(SurrogateModel):
    """Single-pass FNO predicting mean temperature and local heteroscedastic scale (U1).

    Computes:
        y_norm_mean, log_var = model(x)
        sigma_norm = exp(0.5 * log_var)
    Trained via Gaussian Negative Log-Likelihood (NLL).
    """

    def __init__(
        self,
        in_channels: int = 12,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=2)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth

        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        nn.init.kaiming_normal_(self.p.weight, nonlinearity="relu")
        if self.p.bias is not None:
            nn.init.zeros_(self.p.bias)

        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(width, width, modes_x=modes_x, modes_y=modes_y)
            )
            bypass = nn.Conv2d(width, width, kernel_size=1)
            nn.init.kaiming_normal_(bypass.weight, nonlinearity="relu")
            if bypass.bias is not None:
                nn.init.zeros_(bypass.bias)
            self.bypass_layers.append(bypass)

        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)

        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q_mean = nn.Conv2d(128, 1, kernel_size=1)
        self.q_logvar = nn.Conv2d(128, 1, kernel_size=1)

        nn.init.kaiming_normal_(self.q0.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.q_mean.weight)
        nn.init.xavier_uniform_(self.q_logvar.weight)
        if self.q0.bias is not None:
            nn.init.zeros_(self.q0.bias)
        if self.q_mean.bias is not None:
            nn.init.zeros_(self.q_mean.bias)
        if self.q_logvar.bias is not None:
            nn.init.constant_(self.q_logvar.bias, -2.0)  # initialize to small initial variance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (mean, log_variance) tuple of shape (batch, 1, ny, nx).
        """
        h = self.p(x)
        for spec, bypass in zip(self.spectral_layers, self.bypass_layers, strict=True):
            h = self.act(spec(h) + bypass(h))

        h = self.act(self.q0(h))
        mean = self.q_mean(h)
        logvar = torch.clamp(self.q_logvar(h), min=-14.0, max=10.0)
        return mean, logvar

    @staticmethod
    def nll_loss(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian negative log-likelihood loss with numerical clamping."""
        precision = torch.exp(-logvar)
        diff_sq = (target - mean) ** 2
        loss = 0.5 * (precision * diff_sq + logvar)
        return torch.mean(loss)


class EvidentialFNO2D(SurrogateModel):
    """Deep Evidential Regression FNO predicting Normal-Inverse-Gamma parameters (U2).

    Outputs 4 physical fields per grid cell:
        gamma: predictive mean
        v > 0: virtual observation evidence (degrees of freedom for mean variance)
        alpha > 1: shape parameter of Inverse-Gamma
        beta > 0: scale parameter of Inverse-Gamma

    Derived uncertainties:
        aleatoric = sqrt(beta / (alpha - 1))
        epistemic = sqrt(beta / (v * (alpha - 1)))
        total = sqrt(aleatoric^2 + epistemic^2)
    """

    def __init__(
        self,
        in_channels: int = 12,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        activation: str = "gelu",
        eps: float = 1e-4,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=4)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth
        self.eps = eps

        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        nn.init.kaiming_normal_(self.p.weight, nonlinearity="relu")
        if self.p.bias is not None:
            nn.init.zeros_(self.p.bias)

        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(width, width, modes_x=modes_x, modes_y=modes_y)
            )
            bypass = nn.Conv2d(width, width, kernel_size=1)
            nn.init.kaiming_normal_(bypass.weight, nonlinearity="relu")
            if bypass.bias is not None:
                nn.init.zeros_(bypass.bias)
            self.bypass_layers.append(bypass)

        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)

        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q_out = nn.Conv2d(128, 4, kernel_size=1)

        nn.init.kaiming_normal_(self.q0.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.q_out.weight)
        if self.q0.bias is not None:
            nn.init.zeros_(self.q0.bias)
        if self.q_out.bias is not None:
            # Initialize: gamma=0, v=1, alpha=2, beta=1
            self.q_out.bias.data = torch.tensor([0.0, 0.54, 0.54, 0.54])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (gamma, v, alpha, beta) of shape (batch, 1, ny, nx) with strict NIG support:
            v > 0, alpha > 1, beta > 0.
        """
        h = self.p(x)
        for spec, bypass in zip(self.spectral_layers, self.bypass_layers, strict=True):
            h = self.act(spec(h) + bypass(h))

        h = self.act(self.q0(h))
        raw = self.q_out(h)

        gamma = raw[:, 0:1, :, :]
        # Enforce positive virtual observation evidence v > 0
        v = F.softplus(raw[:, 1:2, :, :]) + self.eps
        # Enforce alpha > 1 for finite mean of Inverse-Gamma distribution
        alpha = F.softplus(raw[:, 2:3, :, :]) + 1.0 + self.eps
        # Enforce beta > 0 scale parameter
        beta = F.softplus(raw[:, 3:4, :, :]) + self.eps

        return gamma, v, alpha, beta

    @staticmethod
    def get_uncertainties(
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aleatoric, epistemic, and total standard deviation fields."""
        var_aleatoric = beta / (alpha - 1.0)
        var_epistemic = beta / (v * (alpha - 1.0))
        var_total = var_aleatoric + var_epistemic
        return (
            torch.sqrt(torch.clamp(var_aleatoric, min=1e-8)),
            torch.sqrt(torch.clamp(var_epistemic, min=1e-8)),
            torch.sqrt(torch.clamp(var_total, min=1e-8)),
        )

    @staticmethod
    def evidential_loss(
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
        lambda_reg: float = 0.01,
    ) -> torch.Tensor:
        """Compute Deep Evidential Student-t NLL loss with evidence regularizer."""
        two_beta_one_plus_v = 2.0 * beta * (1.0 + v)
        error_sq = (target - gamma) ** 2

        # 1. Negative Log-Likelihood of Student-t marginal:
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(two_beta_one_plus_v)
            + (alpha + 0.5) * torch.log(error_sq * v + two_beta_one_plus_v)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # 2. Evidence regularizer: penalize placing evidence (2v + alpha) on high errors |y - gamma|
        abs_error = torch.abs(target - gamma)
        total_evidence = 2.0 * v + alpha
        reg = abs_error * total_evidence

        loss = torch.mean(nll) + lambda_reg * torch.mean(reg)
        return loss


class MCDropoutFNO2D(SurrogateModel):
    """FNO with Spatial Dropout for Monte Carlo epistemic sampling (U3).

    Applies 2D spatial channel dropout between Fourier layers.
    At inference time, multiple forward passes with active dropout estimate variance.
    """

    def __init__(
        self,
        in_channels: int = 12,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        dropout_p: float = 0.10,
        activation: str = "gelu",
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=1)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth
        self.dropout_p = dropout_p

        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(width, width, modes_x=modes_x, modes_y=modes_y)
            )
            self.bypass_layers.append(nn.Conv2d(width, width, kernel_size=1))
            self.dropout_layers.append(nn.Dropout2d(p=dropout_p))

        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q1 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, force_dropout: bool = False) -> torch.Tensor:
        """Forward pass with optional test-time dropout activation."""
        h = self.p(x)
        for spec, bypass, drop in zip(
            self.spectral_layers, self.bypass_layers, self.dropout_layers, strict=True
        ):
            h_latent = self.act(spec(h) + bypass(h))
            h = drop(h_latent) if (self.training or force_dropout) else h_latent

        h = self.act(self.q0(h))
        out = self.q1(h)
        return out

    def sample_mc(self, x: torch.Tensor, num_passes: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo dropout sampling.

        Returns:
            (mean, std) of shape (batch, 1, ny, nx).
        """
        preds = []
        for _ in range(num_passes):
            preds.append(self.forward(x, force_dropout=True))
        stacked = torch.stack(preds, dim=0)
        mean = torch.mean(stacked, dim=0)
        std = torch.std(stacked, dim=0, unbiased=True)
        return mean, std


class DistilledStudentFNO2D(SurrogateModel):
    """Single-pass student FNO predicting mean and ensemble-distilled epistemic scale (U4).

    Trained on teacher ensemble predictions to output both mean temperature
    and estimated epistemic uncertainty in a single evaluation.
    """

    def __init__(
        self,
        in_channels: int = 12,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=2)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth

        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(width, width, modes_x=modes_x, modes_y=modes_y)
            )
            self.bypass_layers.append(nn.Conv2d(width, width, kernel_size=1))

        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q_mean = nn.Conv2d(128, 1, kernel_size=1)
        self.q_std = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (mean, std_pred) of shape (batch, 1, ny, nx).
        """
        h = self.p(x)
        for spec, bypass in zip(self.spectral_layers, self.bypass_layers, strict=True):
            h = self.act(spec(h) + bypass(h))

        h = self.act(self.q0(h))
        mean = self.q_mean(h)
        # Enforce positive uncertainty scale
        std_pred = F.softplus(self.q_std(h)) + 1e-4
        return mean, std_pred

    @staticmethod
    def distillation_loss(
        student_mean: torch.Tensor,
        student_std: torch.Tensor,
        target: torch.Tensor,
        teacher_mean: torch.Tensor,
        teacher_std: torch.Tensor,
        lambda_teacher: float = 0.5,
        lambda_std: float = 0.5,
    ) -> torch.Tensor:
        """Combined distillation loss matching target ground truth and teacher distribution."""
        loss_target = F.mse_loss(student_mean, target)
        loss_teacher = F.mse_loss(student_mean, teacher_mean)
        loss_std = F.mse_loss(student_std, teacher_std)
        return loss_target + lambda_teacher * loss_teacher + lambda_std * loss_std
