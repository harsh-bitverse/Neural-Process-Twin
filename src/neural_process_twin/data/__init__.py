"""Data generation, storage, normalization, and PyTorch ingestion for Neural Process Twin.

This package manages:
1. Reusable physical field generators (Gaussian random fields, composite materials, heat sources, boundary conditions).
2. Numerical solver orchestration, convergence verification, and physical energy balance diagnostics.
3. Failure auditing policy (fail-fast vs. transparent skip-and-audit log).
4. Disjoint dataset partitioning ('train', 'validation', 'test_iid', 'test_ood') with explicit physical OOD criteria.
5. Versioned, compressed HDF5 archival storage and dataset inspection utilities.
6. Safe multi-worker PyTorch Dataset and DataLoader ingestion with feature normalization.
"""

from __future__ import annotations

from neural_process_twin.data.boundary_encoding import (
    CANONICAL_CHANNEL_NAMES,
    NUM_CANONICAL_CHANNELS,
    decode_boundary_conditions,
    encode_boundary_conditions,
)
from neural_process_twin.data.dataset import (
    HeatFieldDataset,
    create_dataloader,
    seed_worker,
)
from neural_process_twin.data.generators import (
    BoundaryConditionSampler,
    CompositeMaterialGenerator,
    CovarianceKernel,
    GaussianRandomFieldGenerator,
    HeatSourceGenerator,
    InclusionGeometry,
    PositivityTransform,
)
from neural_process_twin.data.normalization import (
    FieldNormalizer,
    FieldStatistics,
    NormalizationMethod,
)
from neural_process_twin.data.orchestration import (
    DatasetGenerationError,
    DatasetOrchestrator,
    FailureAuditRecord,
    GenerationAudit,
    ProblemSample,
)
from neural_process_twin.data.splits import (
    IID_PARTITION_RULE,
    OOD_PARTITION_RULE,
    DatasetPartitionPlan,
    PartitionRule,
    SplitName,
    verify_ood_criteria,
    verify_partitions_disjoint,
)
from neural_process_twin.data.storage import (
    DATASET_FORMAT_VERSION,
    DatasetInspectionSummary,
    HDF5DatasetWriter,
    SplitSummary,
    StorageSchemaError,
    inspect_hdf5_dataset,
)

__all__ = [
    "CANONICAL_CHANNEL_NAMES",
    "DATASET_FORMAT_VERSION",
    "IID_PARTITION_RULE",
    "NUM_CANONICAL_CHANNELS",
    "OOD_PARTITION_RULE",
    "BoundaryConditionSampler",
    "CompositeMaterialGenerator",
    "CovarianceKernel",
    "DatasetGenerationError",
    "DatasetInspectionSummary",
    "DatasetOrchestrator",
    "DatasetPartitionPlan",
    "FailureAuditRecord",
    "FieldNormalizer",
    "FieldStatistics",
    "GaussianRandomFieldGenerator",
    "GenerationAudit",
    "HDF5DatasetWriter",
    "HeatFieldDataset",
    "HeatSourceGenerator",
    "InclusionGeometry",
    "NormalizationMethod",
    "PartitionRule",
    "PositivityTransform",
    "ProblemSample",
    "SplitName",
    "SplitSummary",
    "StorageSchemaError",
    "create_dataloader",
    "decode_boundary_conditions",
    "encode_boundary_conditions",
    "inspect_hdf5_dataset",
    "seed_worker",
    "verify_ood_criteria",
    "verify_partitions_disjoint",
]
