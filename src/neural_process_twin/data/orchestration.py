"""Solver orchestration, problem sampling, and quality control pipeline.

Orchestrates parameter field generation, numerical reference solver execution,
algebraic residual verification, and physical energy conservation diagnostics.
Implements a configurable transparent failure policy (fail-fast vs skip-and-audit log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_process_twin.data.generators import (
    BoundaryConditionSampler,
    CompositeMaterialGenerator,
    CovarianceKernel,
    GaussianRandomFieldGenerator,
    HeatSourceGenerator,
    PositivityTransform,
)
from neural_process_twin.data.splits import (
    IID_PARTITION_RULE,
    OOD_PARTITION_RULE,
    DatasetPartitionPlan,
    SplitName,
)
from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D, SolverResult


class DatasetGenerationError(Exception):
    """Raised when problem generation or reference solver verification fails under fail-fast policy."""


@dataclass(frozen=True)
class ProblemSample:
    """A fully solved and verified physical problem sample.

    Attributes:
        sample_id: Unique string identifier for the sample (e.g. 'train_00042').
        seed: Random seed used to synthesize the field parameters.
        split: The dataset partition this sample belongs to.
        conductivity: 2D array of thermal conductivity k(x, y) [W/(m K)].
        source: 2D array of volumetric heat source q(x, y) [W/m^3].
        boundary_conditions: Applied boundary conditions on the 4 domain boundaries.
        solver_result: Complete structured output from the numerical reference solver.
    """

    sample_id: str
    seed: int
    split: SplitName
    conductivity: np.ndarray
    source: np.ndarray
    boundary_conditions: BoundaryConditions
    solver_result: SolverResult


@dataclass(frozen=True)
class FailureAuditRecord:
    """Structured audit record for a rejected or failed problem generation attempt."""

    sample_id: str
    seed: int
    split: str
    reason: str
    diagnostic_details: dict[str, Any]


@dataclass
class GenerationAudit:
    """Audit summary tracking generation attempts, successes, and failure diagnostics."""

    attempted_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    failure_records: list[FailureAuditRecord] = field(default_factory=list)

    def record_failure(
        self,
        sample_id: str,
        seed: int,
        split: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a failure event to the audit manifest."""
        self.failed_samples += 1
        self.failure_records.append(
            FailureAuditRecord(
                sample_id=sample_id,
                seed=seed,
                split=split,
                reason=reason,
                diagnostic_details=details or {},
            )
        )

    def record_success(self) -> None:
        """Increment count of verified successful samples."""
        self.successful_samples += 1


class DatasetOrchestrator:
    """Orchestrates problem sampling, reference solver invocation, and physical verification.

    Applies conservative quality control checks:
    1. Algebraic sparse matrix residual: ||A*T - b||_2 / ||b||_2 < residual_tol
    2. Physical heat conservation balance: |Q_imbalance| / max(|Q_source|, |Q_boundary|, 1) < balance_tol
    3. Temperature positivity: min(T) > 0 K
    """

    def __init__(
        self,
        grid: Grid2D,
        solver: HeatConductionSolver2D | None = None,
        fail_fast: bool = True,
        residual_tolerance: float = 1e-6,
        heat_balance_tolerance: float = 1e-3,
        max_retries_per_sample: int = 10,
    ) -> None:
        """Initialize dataset orchestrator.

        Args:
            grid: 2D computational domain grid.
            solver: Instance of HeatConductionSolver2D. If None, default solver is used.
            fail_fast: If True, any solver or diagnostic failure raises DatasetGenerationError.
                If False, the failed sample is logged to the audit manifest and skipped.
            residual_tolerance: Upper bound for normalized linear system residual.
            heat_balance_tolerance: Upper bound for relative physical heat imbalance.
            max_retries_per_sample: Maximum consecutive retry attempts when fail_fast=False.
        """
        self.grid = grid
        self.solver = solver or HeatConductionSolver2D(tolerance=1e-10)
        self.fail_fast = fail_fast
        self.residual_tolerance = float(residual_tolerance)
        self.heat_balance_tolerance = float(heat_balance_tolerance)
        self.max_retries_per_sample = int(max_retries_per_sample)

    def _sample_fields(
        self,
        split: SplitName,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, BoundaryConditions]:
        """Synthesize conductivity, heat source, and boundary conditions for a given split and seed."""
        rng = np.random.default_rng(seed)

        if split == SplitName.TEST_OOD:
            # OOD: High-contrast composite inclusions or high-contrast GRF
            # With 70% probability: Piecewise-constant composite with contrast ratio 20:1 to 50:1
            # With 30% probability: Fine correlation length GRF with short ell (0.05 m)
            if rng.random() < 0.7:
                comp_gen = CompositeMaterialGenerator(
                    grid=self.grid,
                    matrix_conductivity=1.0,
                )
                k_field = comp_gen.generate(
                    rng=rng,
                    num_inclusions=int(rng.integers(2, 5)),
                    contrast_range=(20.0, 50.0),
                    radius_range=(0.06, 0.16),
                )
            else:
                grf_gen = GaussianRandomFieldGenerator(
                    grid=self.grid,
                    kernel=CovarianceKernel.MATERN,
                    correlation_length=float(rng.uniform(0.04, 0.08)),
                    nu=1.5,
                    positivity_transform=PositivityTransform.SIGMOID,
                    k_min=0.5,
                    k_max=25.0,
                )
                k_field = grf_gen.generate(rng=rng)

            # OOD heat source: elevated peak amplitude
            source_gen = HeatSourceGenerator(
                grid=self.grid,
                distribution="gaussian_blobs",
                q_min=OOD_PARTITION_RULE.source_amplitude_range[0],
                q_max=OOD_PARTITION_RULE.source_amplitude_range[1],
                num_peaks=int(rng.integers(2, 5)),
                spread_range=(0.04, 0.12),
            )
            q_field = source_gen.generate(rng=rng)

            # OOD boundary conditions
            bc_sampler = BoundaryConditionSampler(
                temperature_range=OOD_PARTITION_RULE.temperature_bc_range,
                flux_range=(-1000.0, 1000.0),
                mode="mixed_dirichlet_neumann",
            )
            bcs = bc_sampler.sample(rng=rng)

        else:
            # IID (train, validation, test_iid): Smooth GRF and moderate thermal parameters
            rule = IID_PARTITION_RULE
            corr_len = float(
                rng.uniform(rule.correlation_length_range[0], rule.correlation_length_range[1])
            )
            grf_gen = GaussianRandomFieldGenerator(
                grid=self.grid,
                kernel=CovarianceKernel.GAUSSIAN,
                correlation_length=corr_len,
                positivity_transform=PositivityTransform.SIGMOID,
                k_min=rule.conductivity_range[0],
                k_max=rule.conductivity_range[1],
            )
            k_field = grf_gen.generate(rng=rng)

            # IID heat source: moderate Gaussian thermal peaks
            source_gen = HeatSourceGenerator(
                grid=self.grid,
                distribution="gaussian_blobs",
                q_min=rule.source_amplitude_range[0],
                q_max=rule.source_amplitude_range[1],
                num_peaks=int(rng.integers(1, 4)),
                spread_range=(0.06, 0.18),
            )
            q_field = source_gen.generate(rng=rng)

            # IID boundary conditions
            bc_sampler = BoundaryConditionSampler(
                temperature_range=rule.temperature_bc_range,
                flux_range=(-300.0, 300.0),
                mode="mixed_dirichlet_neumann",
            )
            bcs = bc_sampler.sample(rng=rng)

        return k_field, q_field, bcs

    def generate_sample(
        self,
        sample_id: str,
        seed: int,
        split: SplitName,
    ) -> ProblemSample:
        """Generate, solve, and physically verify a single problem sample.

        Raises:
            DatasetGenerationError: If solver fails or physical quality check is violated.
        """
        # 1. Synthesize parameter fields
        k_field, q_field, bcs = self._sample_fields(split=split, seed=seed)

        # 2. Invoke numerical reference solver
        try:
            result = self.solver.solve(
                grid=self.grid,
                conductivity=k_field,
                source=q_field,
                boundary_conditions=bcs,
            )
        except Exception as exc:
            raise DatasetGenerationError(
                f"Solver invocation failed for {sample_id}: {exc}"
            ) from exc

        # 3. Quality control: verify temperature positivity
        t_min = float(np.min(result.temperature))
        if not np.all(np.isfinite(result.temperature)):
            raise DatasetGenerationError(
                f"Non-finite temperature detected in solution for {sample_id}."
            )
        if t_min <= 0.0:
            raise DatasetGenerationError(
                f"Negative or zero physical temperature ({t_min:.2f} K) detected in {sample_id}."
            )

        # 4. Quality control: verify algebraic sparse residual
        rel_res = result.diagnostics.relative_residual
        if rel_res > self.residual_tolerance:
            raise DatasetGenerationError(
                f"Algebraic relative residual ({rel_res:.2e}) exceeded tolerance "
                f"({self.residual_tolerance:.2e}) in {sample_id}."
            )

        # 5. Quality control: verify physical heat balance conservation
        rel_imbalance = result.diagnostics.energy_balance.relative_imbalance
        if rel_imbalance > self.heat_balance_tolerance:
            raise DatasetGenerationError(
                f"Physical heat balance relative error ({rel_imbalance:.2e}) exceeded "
                f"tolerance ({self.heat_balance_tolerance:.2e}) in {sample_id}."
            )

        return ProblemSample(
            sample_id=sample_id,
            seed=seed,
            split=split,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
            solver_result=result,
        )

    def generate_split(
        self,
        split: SplitName,
        seeds: list[int],
        audit: GenerationAudit,
    ) -> list[ProblemSample]:
        """Generate an entire partition of verified samples.

        Args:
            split: Dataset partition (train, validation, test_iid, test_ood).
            seeds: List of unique seeds for each requested sample in the partition.
            audit: GenerationAudit tracker to update.

        Returns:
            List of successfully solved and verified ProblemSample objects.
        """
        samples: list[ProblemSample] = []

        for idx, base_seed in enumerate(seeds):
            sample_id = f"{split.value}_{idx:05d}"
            current_seed = base_seed
            retry_count = 0
            success = False

            while not success and retry_count < self.max_retries_per_sample:
                audit.attempted_samples += 1
                try:
                    sample = self.generate_sample(
                        sample_id=sample_id,
                        seed=current_seed,
                        split=split,
                    )
                    samples.append(sample)
                    audit.record_success()
                    success = True
                except DatasetGenerationError as err:
                    if self.fail_fast:
                        audit.record_failure(
                            sample_id=sample_id,
                            seed=current_seed,
                            split=split.value,
                            reason=str(err),
                        )
                        raise

                    # Skip-and-audit policy: record diagnostic and increment retry seed
                    retry_count += 1
                    audit.record_failure(
                        sample_id=sample_id,
                        seed=current_seed,
                        split=split.value,
                        reason=str(err),
                        details={"retry_attempt": retry_count},
                    )
                    current_seed += 100_000  # Shift seed deterministically for retry

            if not success:
                raise DatasetGenerationError(
                    f"Exceeded max retries ({self.max_retries_per_sample}) generating "
                    f"valid sample for {sample_id}."
                )

        return samples

    def generate_dataset(
        self,
        plan: DatasetPartitionPlan,
    ) -> tuple[dict[SplitName, list[ProblemSample]], GenerationAudit]:
        """Generate complete dataset across all 4 partitions.

        Args:
            plan: DatasetPartitionPlan specifying sample counts and base seed.

        Returns:
            Tuple of (samples_by_split, generation_audit).
        """
        audit = GenerationAudit()
        dataset: dict[SplitName, list[ProblemSample]] = {}

        for split in (
            SplitName.TRAIN,
            SplitName.VALIDATION,
            SplitName.TEST_IID,
            SplitName.TEST_OOD,
        ):
            seeds = plan.get_split_seeds(split)
            if seeds:
                dataset[split] = self.generate_split(split=split, seeds=seeds, audit=audit)
            else:
                dataset[split] = []

        if plan.num_cal > 0:
            cal_seeds = plan.get_split_seeds(SplitName.CALIBRATION)
            dataset[SplitName.CALIBRATION] = self.generate_split(
                split=SplitName.CALIBRATION, seeds=cal_seeds, audit=audit
            )

        return dataset, audit
