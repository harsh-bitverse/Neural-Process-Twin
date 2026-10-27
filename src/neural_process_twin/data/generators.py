"""Physical field generators for 2D steady-state heat conduction.

This module provides reproducible, parameterized generators for:
1. Smooth Gaussian random fields (GRF) with Gaussian or Matérn covariance and positive conductivity.
2. Piecewise-constant composite materials with circular or rectangular inclusions.
3. Volumetric heat source distributions (zero, uniform, single/multi-peak Gaussian blobs).
4. Boundary condition configurations (Dirichlet and mixed Dirichlet-Neumann).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np

from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


class CovarianceKernel(StrEnum):
    """Supported covariance kernels for Gaussian random field generation."""

    GAUSSIAN = "gaussian"  # Squared exponential / RBF: C(r) = sigma^2 * exp(-r^2 / (2 * ell^2))
    MATERN = "matern"  # Matérn kernel with smoothness parameter nu > 0


class PositivityTransform(StrEnum):
    """Transformation method ensuring strictly positive thermal conductivity k(x, y) > 0."""

    SIGMOID = "sigmoid"  # k = k_min + (k_max - k_min) * (1 / (1 + exp(-xi)))
    LOG_NORMAL = "log_normal"  # k = exp(mu + sigma * xi), optionally scaled


class GaussianRandomFieldGenerator:
    """Fourier-based Gaussian Random Field (GRF) generator on a uniform 2D Grid2D.

    Generates statistically isotropic, stationary 2D Gaussian random fields using
    the spectral FFT method in O(N log N) computational complexity. Avoids assembling
    or decomposing dense O(N^2) covariance matrices.

    Physical conductivity positivity k(x, y) > 0 is enforced through either:
    1. Sigmoidal mapping: k(x, y) = k_min + (k_max - k_min) / (1 + exp(-xi(x, y))),
       guaranteeing values stay strictly inside (k_min, k_max).
    2. Log-normal transformation: k(x, y) = exp(mu + sigma * xi(x, y)),
       where xi is a standard normal GRF.

    Note on covariance preservation:
    Applying a nonlinear monotonic transformation (such as sigmoid or exponential) to a Gaussian
    field alters the exact marginal covariance structure from C(r) to a transformed correlation
    function, but strictly preserves spatial continuity, smoothness, and the characteristic
    correlation length scale ell.
    """

    def __init__(
        self,
        grid: Grid2D,
        kernel: CovarianceKernel | str = CovarianceKernel.GAUSSIAN,
        correlation_length: float = 0.2,
        nu: float = 2.5,
        variance: float = 1.0,
        positivity_transform: PositivityTransform | str = PositivityTransform.SIGMOID,
        k_min: float = 0.1,
        k_max: float = 10.0,
        log_normal_mu: float = 0.0,
        log_normal_sigma: float = 0.5,
    ) -> None:
        """Initialize GRF generator.

        Args:
            grid: 2D computational grid defining spatial bounds and resolution.
            kernel: Covariance kernel type (gaussian or matern).
            correlation_length: Spatial correlation length ell [m]. Must be > 0.
            nu: Matérn smoothness parameter nu. Must be > 0. Only used if kernel is matern.
            variance: Prior marginal variance sigma^2 before positivity transform.
            positivity_transform: Mapping used to guarantee k > 0 (sigmoid or log_normal).
            k_min: Strictly positive lower bound for sigmoid transform [W/(m K)].
            k_max: Upper bound for sigmoid transform [W/(m K)].
            log_normal_mu: Mean log-conductivity parameter for log_normal transform.
            log_normal_sigma: Scale parameter for log_normal transform.
        """
        self.grid = grid
        self.kernel = CovarianceKernel(kernel)
        self.correlation_length = float(correlation_length)
        self.nu = float(nu)
        self.variance = float(variance)
        self.positivity_transform = PositivityTransform(positivity_transform)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.log_normal_mu = float(log_normal_mu)
        self.log_normal_sigma = float(log_normal_sigma)

        self._validate_parameters()
        self._precompute_spectral_filter()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.correlation_length <= 0.0:
            raise ValueError(
                f"correlation_length must be strictly positive, got {self.correlation_length}."
            )
        if self.nu <= 0.0:
            raise ValueError(f"Matérn parameter nu must be strictly positive, got {self.nu}.")
        if self.variance <= 0.0:
            raise ValueError(f"variance must be strictly positive, got {self.variance}.")
        if self.k_min <= 0.0:
            raise ValueError(f"k_min must be strictly positive, got {self.k_min}.")
        if self.k_max <= self.k_min:
            raise ValueError(
                f"k_max ({self.k_max}) must be strictly greater than k_min ({self.k_min})."
            )

    def _precompute_spectral_filter(self) -> None:
        """Precompute the 2D power spectral density filter on the grid wavevectors."""
        # Wavevectors in 2D
        # Fourier frequencies corresponding to uniform spatial grid
        kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
        k_sq = kx_grid**2 + ky_grid**2

        ell = self.correlation_length
        sigma_sq = self.variance

        if self.kernel == CovarianceKernel.GAUSSIAN:
            # 2D Fourier transform of Gaussian kernel C(r) = sigma^2 * exp(-r^2 / (2 * ell^2)):
            # S(k) = 2 * pi * sigma^2 * ell^2 * exp(-ell^2 * |k|^2 / 2)
            psd = 2.0 * np.pi * sigma_sq * (ell**2) * np.exp(-0.5 * (ell**2) * k_sq)
        elif self.kernel == CovarianceKernel.MATERN:
            # 2D Fourier transform of Matérn kernel with smoothness nu:
            # S(k) = sigma^2 * 4 * pi * nu * (2 * nu / ell^2 + |k|^2)^(-(nu + 1)) * (2 * nu / ell^2)^nu
            # Constant factor from continuous Fourier transform in d=2:
            # C(r) = sigma^2 * 2^(1-nu) / gamma(nu) * (sqrt(2*nu)*r/ell)^nu * K_nu(sqrt(2*nu)*r/ell)
            c_factor = sigma_sq * 4.0 * np.pi * self.nu * (2.0 * self.nu / (ell**2)) ** self.nu
            psd = c_factor * (2.0 * self.nu / (ell**2) + k_sq) ** (-(self.nu + 1.0))
        else:
            raise NotImplementedError(f"Unsupported kernel {self.kernel}")

        # Discrete scaling factor for 2D inverse FFT
        # The energy of the continuous PSD mapped to discrete grid:
        # filter = sqrt(psd / (dx * dy))
        self._spectral_filter = np.sqrt(psd / (self.grid.dx * self.grid.dy))

    def generate(self, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Generate a realization of the conductivity field k(x, y).

        Args:
            rng: NumPy random generator instance or integer seed for reproducibility.

        Returns:
            2D NumPy array of shape (ny, nx) containing strictly positive conductivities [W/(m K)].
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # 1. Generate complex standard Gaussian white noise in frequency space
        # shape is (ny, nx)
        noise_real = rng.standard_normal((self.grid.ny, self.grid.nx))
        noise_imag = rng.standard_normal((self.grid.ny, self.grid.nx))
        noise = (noise_real + 1j * noise_imag) / np.sqrt(2.0)

        # 2. Filter noise in frequency domain and perform inverse 2D FFT
        field_fft = self._spectral_filter * noise
        raw_field = np.real(np.fft.ifft2(field_fft)) * np.sqrt(self.grid.nx * self.grid.ny)

        # Normalize field to zero empirical mean and unit variance for consistent transform scaling
        field_std = float(np.std(raw_field))
        xi = (raw_field - float(np.mean(raw_field))) / field_std if field_std > 1e-12 else raw_field

        # 3. Apply positivity-preserving transformation
        if self.positivity_transform == PositivityTransform.SIGMOID:
            # Standard sigmoid: 1 / (1 + exp(-xi)) in (0, 1)
            # Clip xi for numerical safety in exponent
            xi_clipped = np.clip(xi, -30.0, 30.0)
            sigmoid_val = 1.0 / (1.0 + np.exp(-xi_clipped))
            k = self.k_min + (self.k_max - self.k_min) * sigmoid_val
        elif self.positivity_transform == PositivityTransform.LOG_NORMAL:
            log_k = self.log_normal_mu + self.log_normal_sigma * xi
            log_k = np.clip(log_k, -20.0, 20.0)
            k = np.exp(log_k)
        else:
            raise NotImplementedError(f"Unsupported transform {self.positivity_transform}")

        # Final verification
        if not np.all(np.isfinite(k)):
            raise FloatingPointError("Generated conductivity field contains non-finite values.")
        if np.any(k <= 0.0):
            raise ValueError(
                f"Generated conductivity contains non-positive values (min={np.min(k)})."
            )

        return k.astype(np.float64)


@dataclass(frozen=True)
class InclusionGeometry:
    """Specification of a single geometric inclusion within a composite material."""

    shape_type: Literal["circle", "rectangle"]
    conductivity: float
    center_x: float
    center_y: float
    radius: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        if self.conductivity <= 0.0:
            raise ValueError(
                f"Inclusion conductivity must be strictly positive, got {self.conductivity}."
            )
        if self.shape_type == "circle" and self.radius <= 0.0:
            raise ValueError(f"Circular inclusion radius must be > 0, got {self.radius}.")
        if self.shape_type == "rectangle" and (self.width <= 0.0 or self.height <= 0.0):
            raise ValueError(
                f"Rectangular inclusion dimensions must be > 0, got ({self.width}, {self.height})."
            )

    def mask(self, x_mesh: np.ndarray, y_mesh: np.ndarray) -> np.ndarray:
        """Compute boolean indicator mask for control volume centers inside this inclusion."""
        if self.shape_type == "circle":
            dist_sq = (x_mesh - self.center_x) ** 2 + (y_mesh - self.center_y) ** 2
            return dist_sq <= self.radius**2
        elif self.shape_type == "rectangle":
            x_min = self.center_x - 0.5 * self.width
            x_max = self.center_x + 0.5 * self.width
            y_min = self.center_y - 0.5 * self.height
            y_max = self.center_y + 0.5 * self.height
            return (x_mesh >= x_min) & (x_mesh <= x_max) & (y_mesh >= y_min) & (y_mesh <= y_max)
        else:
            raise ValueError(f"Unknown shape_type {self.shape_type}")


class CompositeMaterialGenerator:
    """Piecewise-constant composite material conductivity field generator.

    Generates 2D thermal conductivity fields consisting of a continuous matrix
    material and embedded inclusions (e.g. circular fibers or rectangular blocks)
    with sharp conductivity contrast.
    """

    def __init__(
        self,
        grid: Grid2D,
        matrix_conductivity: float = 1.0,
        inclusions: list[InclusionGeometry] | None = None,
    ) -> None:
        """Initialize composite material generator.

        Args:
            grid: 2D computational grid.
            matrix_conductivity: Background continuous matrix conductivity [W/(m K)].
            inclusions: Explicit list of geometric inclusions. If None, random inclusions can be generated.
        """
        self.grid = grid
        if matrix_conductivity <= 0.0:
            raise ValueError(
                f"matrix_conductivity must be strictly positive, got {matrix_conductivity}."
            )
        self.matrix_conductivity = float(matrix_conductivity)
        self.inclusions = list(inclusions) if inclusions is not None else []

    def generate(
        self,
        rng: np.random.Generator | int | None = None,
        num_inclusions: int | None = None,
        contrast_range: tuple[float, float] = (10.0, 50.0),
        radius_range: tuple[float, float] = (0.05, 0.15),
        boundary_margin: float = 0.05,
    ) -> np.ndarray:
        """Generate conductivity field.

        If explicit inclusions were supplied at initialization, they are painted onto the matrix.
        Otherwise, `num_inclusions` random non-clipped inclusions are drawn.

        Args:
            rng: NumPy random generator or seed.
            num_inclusions: Number of random circular inclusions to generate if none preconfigured.
            contrast_range: Range of inclusion conductivity values [W/(m K)] or contrast ratios.
            radius_range: Range of inclusion radii [m].
            boundary_margin: Clearance margin from domain boundaries [m].

        Returns:
            2D array of shape (ny, nx) with piecewise-constant conductivities [W/(m K)].
        """
        k = np.full(self.grid.shape, self.matrix_conductivity, dtype=np.float64)
        x_mesh = self.grid.x_mesh
        y_mesh = self.grid.y_mesh

        if self.inclusions:
            for inc in self.inclusions:
                mask = inc.mask(x_mesh, y_mesh)
                k[mask] = inc.conductivity
            return k

        # Otherwise synthesize random circular inclusions
        if num_inclusions is None:
            num_inclusions = 3

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        lx = self.grid.lx
        ly = self.grid.ly
        margin = boundary_margin * min(lx, ly)

        for _ in range(num_inclusions):
            radius = float(rng.uniform(radius_range[0], radius_range[1]))
            # Safe bounding box for inclusion center
            x_low = self.grid.x_min + margin + radius
            x_high = self.grid.x_max - margin - radius
            y_low = self.grid.y_min + margin + radius
            y_high = self.grid.y_max - margin - radius

            if x_low >= x_high or y_low >= y_high:
                # Fallback if domain is too small for requested margin
                cx = float(rng.uniform(self.grid.x_min + radius, self.grid.x_max - radius))
                cy = float(rng.uniform(self.grid.y_min + radius, self.grid.y_max - radius))
            else:
                cx = float(rng.uniform(x_low, x_high))
                cy = float(rng.uniform(y_low, y_high))

            k_inc = float(rng.uniform(contrast_range[0], contrast_range[1]))
            inc = InclusionGeometry(
                shape_type="circle",
                conductivity=k_inc,
                center_x=cx,
                center_y=cy,
                radius=radius,
            )
            mask = inc.mask(x_mesh, y_mesh)
            k[mask] = k_inc

        return k


class HeatSourceGenerator:
    """Volumetric heat generation field generator q(x, y) [W/m^3].

    Supports:
    - Zero source: q(x, y) = 0.
    - Uniform source: q(x, y) = q_0.
    - Gaussian blobs: q(x, y) = sum_p A_p * exp(-((x - x_p)^2 + (y - y_p)^2) / (2 * sigma_p^2))
      where A_p is peak volumetric generation rate [W/m^3] and sigma_p is peak spread [m].
    """

    def __init__(
        self,
        grid: Grid2D,
        distribution: Literal["zero", "uniform", "gaussian_blobs"] = "gaussian_blobs",
        q_min: float = 0.0,
        q_max: float = 1000.0,
        num_peaks: int = 2,
        spread_range: tuple[float, float] = (0.05, 0.15),
    ) -> None:
        """Initialize heat source generator.

        Args:
            grid: 2D computational grid.
            distribution: Type of heat source distribution ('zero', 'uniform', 'gaussian_blobs').
            q_min: Minimum source value or peak amplitude [W/m^3].
            q_max: Maximum source value or peak amplitude [W/m^3].
            num_peaks: Number of localized thermal peaks for gaussian_blobs.
            spread_range: Range of Gaussian standard deviations (spreads) [m].
        """
        self.grid = grid
        self.distribution = distribution
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.num_peaks = int(num_peaks)
        self.spread_range = spread_range

        if self.q_min < 0.0:
            raise ValueError(f"q_min must be non-negative, got {self.q_min}.")
        if self.q_max < self.q_min:
            raise ValueError(f"q_max ({self.q_max}) must be >= q_min ({self.q_min}).")

    def generate(self, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Generate volumetric heat source field q(x, y) of shape (ny, nx) [W/m^3]."""
        if self.distribution == "zero":
            return np.zeros(self.grid.shape, dtype=np.float64)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        if self.distribution == "uniform":
            val = float(rng.uniform(self.q_min, self.q_max))
            return np.full(self.grid.shape, val, dtype=np.float64)

        if self.distribution == "gaussian_blobs":
            q = np.zeros(self.grid.shape, dtype=np.float64)
            x_mesh = self.grid.x_mesh
            y_mesh = self.grid.y_mesh

            for _ in range(self.num_peaks):
                amp = float(rng.uniform(self.q_min, self.q_max))
                spread = float(rng.uniform(self.spread_range[0], self.spread_range[1]))
                # Keep centers within domain interior (margin = 2 * spread)
                margin = min(0.1 * self.grid.lx, spread)
                cx = float(rng.uniform(self.grid.x_min + margin, self.grid.x_max - margin))
                cy = float(rng.uniform(self.grid.y_min + margin, self.grid.y_max - margin))

                r_sq = (x_mesh - cx) ** 2 + (y_mesh - cy) ** 2
                peak = amp * np.exp(-r_sq / (2.0 * spread**2))
                q += peak

            return q

        raise ValueError(f"Unsupported source distribution '{self.distribution}'.")


class BoundaryConditionSampler:
    """Sampler for well-posed boundary conditions across rectangular domain boundaries.

    Ensures mathematical well-posedness by guaranteeing at least one Dirichlet boundary
    is present, precluding ill-conditioned or singular pure-Neumann linear systems.
    """

    def __init__(
        self,
        temperature_range: tuple[float, float] = (290.0, 360.0),
        flux_range: tuple[float, float] = (-500.0, 500.0),
        mode: Literal[
            "dirichlet_all", "mixed_dirichlet_neumann", "random"
        ] = "mixed_dirichlet_neumann",
    ) -> None:
        """Initialize BC sampler.

        Args:
            temperature_range: Range for prescribed Dirichlet boundary temperatures [K].
            flux_range: Range for prescribed outward Neumann heat fluxes [W/m^2].
            mode: BC sampling mode ('dirichlet_all', 'mixed_dirichlet_neumann', or 'random').
        """
        self.temperature_range = temperature_range
        self.flux_range = flux_range
        self.mode = mode

        if self.temperature_range[0] <= 0.0:
            raise ValueError("Boundary temperatures must be strictly positive [K].")

    def sample(self, rng: np.random.Generator | int | None = None) -> BoundaryConditions:
        """Sample a complete, well-posed BoundaryConditions set for all 4 sides."""
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        def rand_t() -> float:
            return float(rng.uniform(self.temperature_range[0], self.temperature_range[1]))

        def rand_q() -> float:
            return float(rng.uniform(self.flux_range[0], self.flux_range[1]))

        if self.mode == "dirichlet_all":
            return BoundaryConditions(
                left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, rand_t()),
                right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, rand_t()),
                bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, rand_t()),
                top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, rand_t()),
            )

        if self.mode == "mixed_dirichlet_neumann":
            # Left & Right Dirichlet, Bottom & Top Neumann (e.g. insulated or flux)
            # Default bottom & top adiabatic (flux = 0) with 70% probability or random flux
            b_val = 0.0 if rng.random() < 0.7 else rand_q()
            t_val = 0.0 if rng.random() < 0.7 else rand_q()
            return BoundaryConditions(
                left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, rand_t()),
                right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, rand_t()),
                bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, b_val),
                top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, t_val),
            )

        if self.mode == "random":
            # Randomly pick types, but guarantee at least one side is Dirichlet
            sides = [BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.BOTTOM, BoundarySide.TOP]
            # Pick at least one guaranteed Dirichlet side
            dirichlet_side = rng.choice(sides)

            bcs: dict[BoundarySide, BoundaryCondition] = {}
            for side in sides:
                if side == dirichlet_side or rng.random() < 0.5:
                    bcs[side] = BoundaryCondition(side, BoundaryType.DIRICHLET, rand_t())
                else:
                    flux = 0.0 if rng.random() < 0.5 else rand_q()
                    bcs[side] = BoundaryCondition(side, BoundaryType.NEUMANN, flux)

            return BoundaryConditions(
                left=bcs[BoundarySide.LEFT],
                right=bcs[BoundarySide.RIGHT],
                bottom=bcs[BoundarySide.BOTTOM],
                top=bcs[BoundarySide.TOP],
            )

        raise ValueError(f"Unsupported BC sampler mode '{self.mode}'.")
