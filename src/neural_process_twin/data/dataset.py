"""PyTorch Dataset and DataLoader ingestion pipeline for physical field predictions.

Provides:
1. Multi-worker safe, lazy HDF5 dataset streaming without deadlocks.
2. Physics-aware input tensor construction (conductivity, heat source, spatial coordinates).
3. Supervised target temperature field tensors with forward and inverse normalization.
4. Deterministic DataLoader construction for training, validation, and benchmarking.
"""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

# Guard against OpenBLAS memory allocation failures in Windows DataLoader worker processes
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neural_process_twin.data.boundary_encoding import encode_boundary_conditions
from neural_process_twin.data.normalization import FieldNormalizer, FieldStatistics
from neural_process_twin.data.splits import SplitName
from neural_process_twin.solvers.grid import Grid2D


class HeatFieldDataset(Dataset):
    """PyTorch Dataset for steady-state heat conduction surrogate modeling.

    Designed for safe multi-worker execution (num_workers > 0) by lazily initializing
    the HDF5 file handle per worker process upon first read.

    Tensors produced:
        - Input 'x': Shape (in_channels, ny, nx) containing:
            - Channel 0: Normalized thermal conductivity k(x, y)
            - Channel 1: Normalized volumetric heat source q(x, y)
            - Optional Channel 2: Normalized x-coordinate in [0, 1]
            - Optional Channel 3: Normalized y-coordinate in [0, 1]
        - Target 'y': Shape (1, ny, nx) containing normalized temperature T(x, y)
        - Raw physical fields and sample metadata.
    """

    def __init__(
        self,
        file_path: str | Path,
        split: SplitName | str = SplitName.TRAIN,
        normalizer: FieldNormalizer | None = None,
        include_coords: bool = True,
        include_bcs: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize dataset.

        Args:
            file_path: Path to the HDF5 archive on disk.
            split: Dataset partition ('train', 'validation', 'test_iid', 'test_ood').
            normalizer: Pre-fitted FieldNormalizer instance. If None, looks for stored
                normalization statistics inside the HDF5 archive.
            include_coords: If True, adds normalized (x, y) spatial coordinate channels.
            dtype: Target PyTorch tensor floating-point datatype (default torch.float32).
        """
        self.file_path = str(Path(file_path).resolve())
        split_str = split.value if isinstance(split, SplitName) else str(split)
        if split_str.lower() in {"val", "valid"}:
            split_str = "validation"
        self.split = split_str
        self.include_coords = include_coords
        self.include_bcs = include_bcs
        self.dtype = dtype

        # Worker-local file handle, initialized lazily in __getitem__
        self._file: h5py.File | None = None

        # Inspect basic metadata without keeping file handle open
        with h5py.File(self.file_path, "r") as f:
            if self.split not in f:
                raise KeyError(
                    f"Partition '{self.split}' not found in dataset archive {self.file_path}."
                )

            split_grp = f[self.split]
            self.length = int(split_grp.attrs.get("sample_count", len(split_grp["temperature"])))

            # Read grid coordinates
            grid_grp = f["grid"]
            self.x_mesh = grid_grp["x_mesh"][:].astype(np.float32)
            self.y_mesh = grid_grp["y_mesh"][:].astype(np.float32)
            self.x_min = float(f.attrs.get("x_min", np.min(self.x_mesh)))
            self.x_max = float(f.attrs.get("x_max", np.max(self.x_mesh)))
            self.y_min = float(f.attrs.get("y_min", np.min(self.y_mesh)))
            self.y_max = float(f.attrs.get("y_max", np.max(self.y_mesh)))

            self.grid = Grid2D(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                nx=self.x_mesh.shape[1],
                ny=self.x_mesh.shape[0],
                thickness=float(f.attrs.get("thickness", 1.0)),
            )

            # If normalizer not provided, attempt to reconstruct from archive
            if normalizer is None and "normalization" in f:
                norm_grp = f["normalization"]
                method = str(norm_grp.attrs.get("method", "standard"))
                eps = float(norm_grp.attrs.get("eps", 1e-8))
                norm = FieldNormalizer(method=method, eps=eps)

                def read_stats(grp_name: str) -> FieldStatistics | None:
                    if grp_name in norm_grp:
                        sg = norm_grp[grp_name]
                        return FieldStatistics(
                            mean=float(sg.attrs["mean"]),
                            std=float(sg.attrs["std"]),
                            min=float(sg.attrs["min"]),
                            max=float(sg.attrs["max"]),
                        )
                    return None

                norm.k_stats = read_stats("k_stats")
                norm.q_stats = read_stats("q_stats")
                norm.t_stats = read_stats("t_stats")
                self.normalizer = norm if norm.is_fitted else None
            else:
                self.normalizer = normalizer

        # Precompute normalized spatial coordinate grids if requested
        if self.include_coords:
            lx = max(self.x_max - self.x_min, 1e-12)
            ly = max(self.y_max - self.y_min, 1e-12)
            self.norm_x_mesh = ((self.x_mesh - self.x_min) / lx).astype(np.float32)
            self.norm_y_mesh = ((self.y_mesh - self.y_min) / ly).astype(np.float32)
        else:
            self.norm_x_mesh = None
            self.norm_y_mesh = None

    def __len__(self) -> int:
        return self.length

    @property
    def nx(self) -> int:
        """Number of grid cells along the x-dimension."""
        return self.grid.nx

    @property
    def ny(self) -> int:
        """Number of grid cells along the y-dimension."""
        return self.grid.ny

    def _get_file_handle(self) -> h5py.File:
        """Retrieve or lazily open worker-local HDF5 file handle."""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and process a single physical problem sample by index.

        Args:
            idx: Integer index in [0, len(self) - 1].

        Returns:
            Dictionary containing:
                - 'x': Input tensor of shape (in_channels, ny, nx)
                - 'y': Target temperature tensor of shape (1, ny, nx)
                - 'sample_id': Unique string identifier
                - 'seed': Random seed used for generation
                - 'k_raw': Unnormalized conductivity tensor (1, ny, nx)
                - 'q_raw': Unnormalized heat source tensor (1, ny, nx)
                - 'temperature_raw': Unnormalized temperature tensor (1, ny, nx)
        """
        if not (0 <= idx < self.length):
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}.")

        f = self._get_file_handle()
        split_grp = f[self.split]

        k_raw = split_grp["k"][idx].astype(np.float32)
        q_raw = split_grp["q"][idx].astype(np.float32)
        t_raw = split_grp["temperature"][idx].astype(np.float32)

        sample_id_raw = split_grp["sample_id"][idx]
        sample_id = (
            sample_id_raw.decode("utf-8")
            if isinstance(sample_id_raw, bytes)
            else str(sample_id_raw)
        )
        seed = int(split_grp["seed"][idx])

        # Normalize fields if normalizer is active
        if self.normalizer is not None:
            k_feat = self.normalizer.transform_k(k_raw)
            q_feat = self.normalizer.transform_q(q_raw)
            t_feat = self.normalizer.transform_temperature(t_raw)
        else:
            k_feat = k_raw
            q_feat = q_raw
            t_feat = t_raw

        # Construct input channel stack: (C, ny, nx)
        channels_list = [k_feat, q_feat]
        if self.include_coords and self.norm_x_mesh is not None and self.norm_y_mesh is not None:
            channels_list.extend([self.norm_x_mesh, self.norm_y_mesh])

        bc_dict: dict[str, Any] | None = None
        if "boundary_conditions" in split_grp:
            bc_grp = split_grp["boundary_conditions"]
            bc_dict = {
                "left_type": bc_grp["left_type"][idx],
                "right_type": bc_grp["right_type"][idx],
                "bottom_type": bc_grp["bottom_type"][idx],
                "top_type": bc_grp["top_type"][idx],
                "left_value": bc_grp["left_value"][idx],
                "right_value": bc_grp["right_value"][idx],
                "bottom_value": bc_grp["bottom_value"][idx],
                "top_value": bc_grp["top_value"][idx],
            }

        if self.include_bcs:
            if bc_dict is None:
                raise RuntimeError(
                    f"boundary_conditions group missing in partition '{self.split}' "
                    f"of {self.file_path} but include_bcs=True was requested."
                )
            bc_channels = encode_boundary_conditions(
                bc_dict, grid=self.grid, normalizer=self.normalizer
            )
            for c in range(8):
                channels_list.append(bc_channels[c])

        input_channels = np.stack(channels_list, axis=0)

        # Target temperature shape: (1, ny, nx)
        target_tensor = np.expand_dims(t_feat, axis=0)

        item = {
            "x": torch.from_numpy(input_channels).to(self.dtype),
            "y": torch.from_numpy(target_tensor).to(self.dtype),
            "sample_id": sample_id,
            "seed": seed,
            "k_raw": torch.from_numpy(np.expand_dims(k_raw, axis=0)).to(self.dtype),
            "q_raw": torch.from_numpy(np.expand_dims(q_raw, axis=0)).to(self.dtype),
            "temperature_raw": torch.from_numpy(np.expand_dims(t_raw, axis=0)).to(self.dtype),
        }
        if bc_dict is not None:
            item["bc_raw"] = bc_dict
        return item

    def __getstate__(self) -> dict[str, Any]:
        """Custom serialization to prevent pickling unpicklable open h5py file handles."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom deserialization ensuring worker starts with closed file handle."""
        self.__dict__.update(state)
        self._file = None

    def close(self) -> None:
        """Close the open HDF5 file handle if present."""
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for deterministic multi-worker data loading."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: HeatFieldDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Create a configured, reproducible PyTorch DataLoader for HeatFieldDataset.

    Args:
        dataset: Instance of HeatFieldDataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of subprocess workers for parallel loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory before returning.
        seed: Random seed for deterministic sample shuffling.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
