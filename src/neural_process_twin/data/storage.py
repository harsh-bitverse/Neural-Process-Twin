"""Archival storage and inspection for Neural Process Twin datasets using HDF5.

Provides a structured, versioned, compressed HDF5 dataset archive supporting:
1. Complete grid geometries and coordinates.
2. Parameter and solution fields (k, q, T) with chunked gzip compression.
3. Boundary conditions and solver diagnostic metrics.
4. Fitted normalization statistics (fitted exclusively on train partition).
5. Generation audit trail and quality control metadata.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.data.orchestration import GenerationAudit, ProblemSample
from neural_process_twin.data.splits import SplitName
from neural_process_twin.solvers.grid import Grid2D

DATASET_FORMAT_VERSION = "1.0.0"


class StorageSchemaError(Exception):
    """Raised when an HDF5 dataset file violates the expected schema or version."""


@dataclass(frozen=True)
class SplitSummary:
    """Summary statistics for a single dataset partition."""

    name: str
    sample_count: int
    k_min: float
    k_max: float
    k_mean: float
    q_min: float
    q_max: float
    q_mean: float
    temperature_min: float
    temperature_max: float
    temperature_mean: float
    max_relative_residual: float
    max_heat_imbalance: float


@dataclass(frozen=True)
class DatasetInspectionSummary:
    """Complete summary of an archived HDF5 dataset."""

    format_version: str
    timestamp: str
    generator: str
    grid_shape: tuple[int, int]
    domain_bounds: tuple[float, float, float, float]
    total_samples: int
    splits: dict[str, SplitSummary]
    normalizer_method: str | None
    failed_samples_count: int

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            f"Neural Process Twin - HDF5 Dataset Archive (Version {self.format_version})",
            "=" * 70,
            f"Created At: {self.timestamp}",
            f"Generator:  {self.generator}",
            f"Domain:     x in [{self.domain_bounds[0]:.2f}, {self.domain_bounds[1]:.2f}], "
            f"y in [{self.domain_bounds[2]:.2f}, {self.domain_bounds[3]:.2f}] m",
            f"Grid:       {self.grid_shape[0]} x {self.grid_shape[1]} (ny x nx)",
            f"Total Samples: {self.total_samples} (Failed generation attempts: {self.failed_samples_count})",
            f"Normalizer: {self.normalizer_method or 'None'}",
            "-" * 70,
            f"{'Partition':<15} | {'Count':<7} | {'k range [W/(m K)]':<18} | {'T range [K]':<16} | {'Max Res':<8}",
            "-" * 70,
        ]
        for name, s in self.splits.items():
            k_str = f"[{s.k_min:.2f}, {s.k_max:.2f}]"
            t_str = f"[{s.temperature_min:.1f}, {s.temperature_max:.1f}]"
            res_str = f"{s.max_relative_residual:.1e}"
            lines.append(
                f"{name:<15} | {s.sample_count:<7} | {k_str:<18} | {t_str:<16} | {res_str:<8}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


class HDF5DatasetWriter:
    """Writes synthesized problem ensembles and metadata to an HDF5 archive."""

    def __init__(
        self,
        compression: str = "gzip",
        compression_opts: int = 4,
        dtype: str = "float32",
    ) -> None:
        """Initialize writer.

        Args:
            compression: HDF5 compression filter ('gzip' or 'lzf').
            compression_opts: Gzip compression level (1-9).
            dtype: Floating-point storage precision for field arrays ('float32' or 'float64').
        """
        self.compression = compression
        self.compression_opts = compression_opts if compression == "gzip" else None
        self.dtype = np.float32 if dtype == "float32" else np.float64

    def write(
        self,
        output_path: str | Path,
        grid: Grid2D,
        dataset: dict[SplitName, list[ProblemSample]],
        normalizer: FieldNormalizer | None = None,
        audit: GenerationAudit | None = None,
    ) -> None:
        """Write complete dataset to HDF5 file.

        Args:
            output_path: Destination path on disk.
            grid: 2D computational domain grid.
            dataset: Mapping from SplitName to list of ProblemSample objects.
            normalizer: Optional fitted FieldNormalizer. If None and train split exists,
                a FieldNormalizer will be fitted automatically on train and stored.
            audit: Generation audit summary.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Fit normalizer on train split if not already supplied
        train_samples = dataset.get(SplitName.TRAIN, [])
        if normalizer is None and train_samples:
            k_train = np.stack([s.conductivity for s in train_samples])
            q_train = np.stack([s.source for s in train_samples])
            t_train = np.stack([s.solver_result.temperature for s in train_samples])
            normalizer = FieldNormalizer().fit(k_train, q_train, t_train)

        with h5py.File(output_path, "w") as f:
            # Root attributes
            f.attrs["format_version"] = DATASET_FORMAT_VERSION
            f.attrs["generator"] = "Neural Process Twin - Milestone 02"
            f.attrs["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
            f.attrs["grid_nx"] = grid.nx
            f.attrs["grid_ny"] = grid.ny
            f.attrs["x_min"] = grid.x_min
            f.attrs["x_max"] = grid.x_max
            f.attrs["y_min"] = grid.y_min
            f.attrs["y_max"] = grid.y_max
            f.attrs["thickness"] = grid.thickness

            # /grid group
            grid_grp = f.create_group("grid")
            grid_grp.create_dataset("x_centers", data=grid.x_centers)
            grid_grp.create_dataset("y_centers", data=grid.y_centers)
            grid_grp.create_dataset("x_mesh", data=grid.x_mesh)
            grid_grp.create_dataset("y_mesh", data=grid.y_mesh)

            # Store each split
            str_dtype = h5py.string_dtype(encoding="utf-8")
            for split_name, samples in dataset.items():
                split_key = (
                    split_name.value if isinstance(split_name, SplitName) else str(split_name)
                )
                split_grp = f.create_group(split_key)
                n_samples = len(samples)
                split_grp.attrs["sample_count"] = n_samples

                if n_samples == 0:
                    continue

                # Prepare arrays
                k_stack = np.stack([s.conductivity for s in samples]).astype(self.dtype)
                q_stack = np.stack([s.source for s in samples]).astype(self.dtype)
                t_stack = np.stack([s.solver_result.temperature for s in samples]).astype(
                    self.dtype
                )
                sample_ids = [s.sample_id for s in samples]
                seeds = np.array([s.seed for s in samples], dtype=np.int64)

                # Chunking: one slice per sample (1, ny, nx)
                chunks_3d = (1, grid.ny, grid.nx)

                # Create datasets
                split_grp.create_dataset(
                    "k",
                    data=k_stack,
                    chunks=chunks_3d,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                split_grp.create_dataset(
                    "q",
                    data=q_stack,
                    chunks=chunks_3d,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                split_grp.create_dataset(
                    "temperature",
                    data=t_stack,
                    chunks=chunks_3d,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                split_grp.create_dataset("sample_id", data=sample_ids, dtype=str_dtype)
                split_grp.create_dataset("seed", data=seeds)

                # /<split>/boundary_conditions group
                bc_grp = split_grp.create_group("boundary_conditions")
                bc_evals = [s.boundary_conditions.evaluate_all(grid) for s in samples]
                bc_grp.create_dataset(
                    "left_type",
                    data=[s.boundary_conditions.left.bc_type.value for s in samples],
                    dtype=str_dtype,
                )
                bc_grp.create_dataset(
                    "right_type",
                    data=[s.boundary_conditions.right.bc_type.value for s in samples],
                    dtype=str_dtype,
                )
                bc_grp.create_dataset(
                    "bottom_type",
                    data=[s.boundary_conditions.bottom.bc_type.value for s in samples],
                    dtype=str_dtype,
                )
                bc_grp.create_dataset(
                    "top_type",
                    data=[s.boundary_conditions.top.bc_type.value for s in samples],
                    dtype=str_dtype,
                )

                bc_grp.create_dataset(
                    "left_value",
                    data=np.stack([ev["left"] for ev in bc_evals]).astype(self.dtype),
                )
                bc_grp.create_dataset(
                    "right_value",
                    data=np.stack([ev["right"] for ev in bc_evals]).astype(self.dtype),
                )
                bc_grp.create_dataset(
                    "bottom_value",
                    data=np.stack([ev["bottom"] for ev in bc_evals]).astype(self.dtype),
                )
                bc_grp.create_dataset(
                    "top_value",
                    data=np.stack([ev["top"] for ev in bc_evals]).astype(self.dtype),
                )

                # /<split>/diagnostics group
                diag_grp = split_grp.create_group("diagnostics")
                residuals = np.array(
                    [s.solver_result.diagnostics.relative_residual for s in samples],
                    dtype=np.float64,
                )
                imbalances = np.array(
                    [
                        s.solver_result.diagnostics.energy_balance.relative_imbalance
                        for s in samples
                    ],
                    dtype=np.float64,
                )
                solve_times = np.array(
                    [s.solver_result.solve_time_seconds for s in samples],
                    dtype=np.float64,
                )
                diag_grp.create_dataset("relative_residual", data=residuals)
                diag_grp.create_dataset("heat_imbalance", data=imbalances)
                diag_grp.create_dataset("solve_time_seconds", data=solve_times)

            # /normalization group
            if normalizer and normalizer.is_fitted:
                norm_grp = f.create_group("normalization")
                norm_dict = normalizer.to_dict()
                norm_grp.attrs["method"] = norm_dict["method"]
                norm_grp.attrs["eps"] = norm_dict["eps"]
                for field_name in ("k_stats", "q_stats", "t_stats"):
                    stats = norm_dict.get(field_name)
                    if stats:
                        sub_grp = norm_grp.create_group(field_name)
                        for k, v in stats.items():
                            sub_grp.attrs[k] = v

            # /audit group
            audit_grp = f.create_group("audit")
            if audit:
                audit_grp.attrs["attempted_samples"] = audit.attempted_samples
                audit_grp.attrs["successful_samples"] = audit.successful_samples
                audit_grp.attrs["failed_samples"] = audit.failed_samples
                if audit.failure_records:
                    rec_strs = [
                        json.dumps(
                            {
                                "sample_id": r.sample_id,
                                "seed": r.seed,
                                "split": r.split,
                                "reason": r.reason,
                                "details": r.diagnostic_details,
                            }
                        )
                        for r in audit.failure_records
                    ]
                    audit_grp.create_dataset("failure_records", data=rec_strs, dtype=str_dtype)
            else:
                audit_grp.attrs["attempted_samples"] = sum(len(s) for s in dataset.values())
                audit_grp.attrs["successful_samples"] = sum(len(s) for s in dataset.values())
                audit_grp.attrs["failed_samples"] = 0


def inspect_hdf5_dataset(file_path: str | Path) -> DatasetInspectionSummary:
    """Inspect and summarize an HDF5 dataset archive.

    Args:
        file_path: Path to the HDF5 dataset file.

    Returns:
        Structured DatasetInspectionSummary dataclass.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        StorageSchemaError: If format_version or mandatory groups are missing.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        version = f.attrs.get("format_version")
        if not version:
            raise StorageSchemaError(f"Missing mandatory format_version attribute in {file_path}.")

        generator = str(f.attrs.get("generator", "Unknown"))
        timestamp = str(f.attrs.get("timestamp", "Unknown"))
        nx = int(f.attrs.get("grid_nx", 0))
        ny = int(f.attrs.get("grid_ny", 0))
        x_min = float(f.attrs.get("x_min", 0.0))
        x_max = float(f.attrs.get("x_max", 0.0))
        y_min = float(f.attrs.get("y_min", 0.0))
        y_max = float(f.attrs.get("y_max", 0.0))

        splits: dict[str, SplitSummary] = {}
        total_samples = 0

        for split_key in ("train", "validation", "test_iid", "test_ood", "calibration"):
            if split_key not in f:
                continue
            split_grp = f[split_key]
            count = int(split_grp.attrs.get("sample_count", 0))
            if count == 0 or "k" not in split_grp:
                splits[split_key] = SplitSummary(
                    name=split_key,
                    sample_count=count,
                    k_min=0.0,
                    k_max=0.0,
                    k_mean=0.0,
                    q_min=0.0,
                    q_max=0.0,
                    q_mean=0.0,
                    temperature_min=0.0,
                    temperature_max=0.0,
                    temperature_mean=0.0,
                    max_relative_residual=0.0,
                    max_heat_imbalance=0.0,
                )
                continue

            k_data = split_grp["k"][:]
            q_data = split_grp["q"][:]
            t_data = split_grp["temperature"][:]
            residuals = split_grp["diagnostics"]["relative_residual"][:]
            imbalances = split_grp["diagnostics"]["heat_imbalance"][:]

            total_samples += count
            splits[split_key] = SplitSummary(
                name=split_key,
                sample_count=count,
                k_min=float(np.min(k_data)),
                k_max=float(np.max(k_data)),
                k_mean=float(np.mean(k_data)),
                q_min=float(np.min(q_data)),
                q_max=float(np.max(q_data)),
                q_mean=float(np.mean(q_data)),
                temperature_min=float(np.min(t_data)),
                temperature_max=float(np.max(t_data)),
                temperature_mean=float(np.mean(t_data)),
                max_relative_residual=float(np.max(residuals)),
                max_heat_imbalance=float(np.max(np.abs(imbalances))),
            )

        norm_method = None
        if "normalization" in f:
            norm_method = str(f["normalization"].attrs.get("method", "standard"))

        failed_count = 0
        if "audit" in f:
            failed_count = int(f["audit"].attrs.get("failed_samples", 0))

        return DatasetInspectionSummary(
            format_version=str(version),
            timestamp=timestamp,
            generator=generator,
            grid_shape=(ny, nx),
            domain_bounds=(x_min, x_max, y_min, y_max),
            total_samples=total_samples,
            splits=splits,
            normalizer_method=norm_method,
            failed_samples_count=failed_count,
        )
