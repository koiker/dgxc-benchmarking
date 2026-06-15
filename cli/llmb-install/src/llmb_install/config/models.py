# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""Configuration data models for LLMB Install."""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import AfterValidator, BaseModel, BeforeValidator, model_validator

from llmb_install.constants import SUPPORTED_GPU_TYPES


def _check_gpu_type(v: str) -> str:
    if v not in SUPPORTED_GPU_TYPES:
        raise ValueError(f"Unsupported gpu_type '{v}'. Must be one of: {sorted(SUPPORTED_GPU_TYPES)}")
    return v


def _coerce_env_vars(v: Any) -> Any:
    if v is None:
        return {}
    if not isinstance(v, dict):
        return v  # let pydantic's type check handle it
    null_keys = [k for k, val in v.items() if val is None]
    if null_keys:
        raise ValueError(
            f"environment_vars contains null values for: {null_keys}. "
            "Use empty strings ('') instead of null/blank values, or remove the keys entirely."
        )
    return {k: str(val) for k, val in v.items()}


def _check_non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("Value must not be blank or whitespace-only")
    return v


GpuType = Annotated[str, AfterValidator(_check_gpu_type)]
NonBlankStr = Annotated[str, AfterValidator(_check_non_blank)]
EnvironmentVars = Annotated[Dict[str, str], BeforeValidator(_coerce_env_vars)]


class SlurmConfig(BaseModel):
    """SLURM cluster configuration."""

    account: NonBlankStr
    gpu_partition: NonBlankStr
    cpu_partition: NonBlankStr
    gpu_partition_gres: Optional[int] = None
    cpu_partition_gres: Optional[int] = None


class RunAIConfig(BaseModel):
    """Run:ai (NVIDIA Run:ai / DGX Cloud) cluster configuration.

    Unlike Slurm, Run:ai schedules OCI container images directly onto Kubernetes
    nodes, so there is no enroot/sqsh build step. Submission is done with the
    ``runai`` CLI (after an interactive ``runai login``), so no Application
    credentials (app_id/app_secret) are required.
    """

    # Run:ai project (Kubernetes namespace) jobs are submitted into.
    project_name: NonBlankStr
    # Existing PVC providing the shared workspace (NEMORUN_HOME, HF cache, etc.).
    pvc_claim_name: NonBlankStr
    pvc_mount_path: NonBlankStr = "/nemo-workspace"
    # OCI image Run:ai pulls for every pod (no sqsh). e.g. nvcr.io/nvidia/nemo:25.04.
    container_image: NonBlankStr
    # RoCE/GDR rails exposed as Kubernetes extended resources, one per rail,
    # e.g. ["nvidia.com/r0-p0=1", ...]. Empty => rely on default networking.
    extended_resources: List[str] = []
    # Multus network annotations (k8s.v1.cni.cncf.io/networks=...), one per entry.
    annotations: List[str] = []
    # Enlarge /dev/shm (runai --large-shm) for dataloader shared memory.
    large_shm: bool = True
    # Place the RoCE rails on the master pod too (not just workers).
    rails_on_master: bool = True
    # Optional Run:ai node pool(s) to target.
    node_pools: Optional[str] = None


class ClusterSettings(BaseModel):
    """Base cluster configuration shared across all config models.

    Contains the stable settings that describe a cluster environment:
    GPU type, architecture, venv strategy, slurm/runai configuration, etc.
    """

    venv_type: Literal['uv', 'venv', 'conda']
    gpu_type: GpuType
    node_architecture: Literal['x86_64', 'aarch64']
    # Launch platform recorded in cluster_config.yaml and used by llmb-run.
    platform: Literal['slurm', 'runai'] = 'slurm'
    install_method: Literal['local', 'slurm'] = 'slurm'
    slurm: Optional[SlurmConfig] = None
    runai: Optional[RunAIConfig] = None
    workload_selection_mode: Literal['custom', 'exemplar'] = 'custom'
    environment_vars: EnvironmentVars = {}
    image_folder: Optional[str] = None

    @model_validator(mode='after')
    def _validate_platform_install_method(self) -> 'ClusterSettings':
        # Run:ai pulls OCI images directly and runs install-time tasks (HF asset
        # prefetch, tool downloads) on the login node; there is no srun/enroot
        # path, so 'slurm' install_method is invalid for the runai platform.
        if self.platform == 'runai' and self.install_method == 'slurm':
            raise ValueError(
                "install_method='slurm' is not supported with platform='runai'. "
                "Use install_method='local' (install-time tasks run on the login node)."
            )
        return self


class PlayfileConfig(ClusterSettings):
    """Schema for headless playfile configuration.

    Defines the fields that are valid in a playfile YAML, along with
    playfile-specific validation rules (non-empty workloads, deprecated
    key rejection). The platform-specific block (slurm or runai) is
    required for the selected platform.
    """

    install_path: NonBlankStr
    install_method: Literal['local', 'slurm']  # required in playfiles (no default)
    selected_workloads: List[NonBlankStr]

    @model_validator(mode='before')
    @classmethod
    def reject_deprecated_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # TODO: Remove this deprecated-key check after next public release.
            deprecated_keys = [key for key in ('slurm_info', 'env_vars') if key in data]
            if deprecated_keys:
                raise ValueError(
                    "Playfiles must use top-level slurm and environment_vars; "
                    "slurm_info/env_vars are no longer supported."
                )
        return data

    @model_validator(mode='after')
    def validate_playfile_rules(self) -> 'PlayfileConfig':
        if not self.selected_workloads:
            raise ValueError("selected_workloads cannot be empty")
        # Require the config block matching the chosen platform.
        if self.platform == 'slurm' and self.slurm is None:
            raise ValueError("platform='slurm' requires a top-level 'slurm' block in the playfile.")
        if self.platform == 'runai' and self.runai is None:
            raise ValueError("platform='runai' requires a top-level 'runai' block in the playfile.")
        return self


class InstallConfig(ClusterSettings):
    """Central configuration object for LLMB installation."""

    # Required fields (no defaults)
    install_path: NonBlankStr

    # Optional fields (with defaults)
    selected_workloads: List[NonBlankStr] = []
    ui_mode: Literal['simple', 'rich', 'express'] = 'simple'
    cache_dirs_configured: bool = False
    dev_mode: bool = False
    llmb_repo: Optional[str] = None
    is_incremental_install: bool = False

    def to_play_dict(self) -> Dict[str, Any]:
        """Convert config to playfile-compatible dictionary."""
        data = PlayfileConfig.model_validate(self.model_dump()).model_dump()
        if data.get('image_folder') is None:
            data.pop('image_folder', None)
        return data

    def get_remaining_workloads(self, completed: List[str]) -> List[str]:
        """Get workloads that still need to be installed."""
        return [w for w in self.selected_workloads if w not in completed]

    @property
    def locked_fields_for_resume(self) -> List[str]:
        """Get list of fields that cannot be changed during resume edit."""
        return [
            'install_path',
            'gpu_type',
            'node_architecture',
            'venv_type',
            'llmb_repo',
            'dev_mode',
            'image_folder',
            'platform',
        ]

    @property
    def editable_fields_for_resume(self) -> List[str]:
        """Get list of fields that can be changed during resume edit."""
        return ['slurm', 'runai', 'install_method', 'selected_workloads']
