# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Shared scheduler/platform argument builder for Megatron-Bridge recipes.
#
# Each recipe's launch.sh sources this file and appends the resulting
# PLATFORM_ARGS array to scripts/performance/setup_experiment.py:
#
#     _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     while [[ "$_d" != "/" && ! -f "$_d/common/platform_args.sh" ]]; do _d="$(dirname "$_d")"; done
#     source "$_d/common/platform_args.sh"
#     python3 scripts/performance/setup_experiment.py ... "${PLATFORM_ARGS[@]}"
#
# Platform selection (PLATFORM env, default "slurm"):
#   slurm  Slurm + enroot/pyxis. Requires SBATCH_ACCOUNT, SBATCH_PARTITION.
#          Honors ADDITIONAL_SLURM_PARAMS (semicolon-separated sbatch flags).
#   runai  NVIDIA Run:ai on Kubernetes (Kubeflow Training Operator + RunAIPlugin).
#          Requires DGXC_PROJECT_NAME, DGXC_PVC_CLAIM_NAME. Optional:
#          DGXC_PVC_MOUNT_PATH, RUNAI_EXTENDED_RESOURCES, RUNAI_ANNOTATIONS,
#          RUNAI_LARGE_SHM, RUNAI_ENV_VARS_JSON.
#
# Run:ai environment variables:
#   DGXC_PROJECT_NAME      Run:ai project name (K8s namespace = runai-<name>)
#   DGXC_PVC_CLAIM_NAME    Shared workspace PVC claim name
#   DGXC_PVC_MOUNT_PATH    Container mount path for the PVC (default: /nemo-workspace)
#   RUNAI_EXTENDED_RESOURCES  Space-separated "key=val" pairs for SR-IOV/RoCE rails
#                             (e.g. "nvidia.com/r0-p0=1 nvidia.com/r1-p0=1 ...")
#   RUNAI_ANNOTATIONS      Space-separated "key=val" pairs for Multus annotations
#                             (e.g. "k8s.v1.cni.cncf.io/networks=default/r0-p0,...")
#   RUNAI_LARGE_SHM        Mount memory-backed /dev/shm (default: true)
#   RUNAI_ENV_VARS_JSON    JSON dict of extra env vars for the training container

PLATFORM=${PLATFORM:-slurm}
PLATFORM=${PLATFORM,,}

PLATFORM_ARGS=()

# Helper: build a JSON object from space-separated "key=value" pairs.
_kv_to_json() {
    local _first=1 _kv
    printf '{'
    for _kv in "$@"; do
        [[ $_first -eq 1 ]] && _first=0 || printf ','
        printf '"%s":"%s"' "${_kv%%=*}" "${_kv#*=}"
    done
    printf '}'
}

if [[ $PLATFORM == "runai" ]]; then
    : "${DGXC_PROJECT_NAME:?DGXC_PROJECT_NAME is required for PLATFORM=runai (e.g. nccl-benchmarking)}"
    : "${DGXC_PVC_CLAIM_NAME:?DGXC_PVC_CLAIM_NAME is required for PLATFORM=runai (e.g. nemo-workspace)}"
    DGXC_PVC_MOUNT_PATH=${DGXC_PVC_MOUNT_PATH:-/nemo-workspace}

    PLATFORM_ARGS=(
        --kubeflow_namespace "runai-${DGXC_PROJECT_NAME}"
        --csp runai
        --runai_pvc_claim_name "$DGXC_PVC_CLAIM_NAME"
        --runai_pvc_mount_path "$DGXC_PVC_MOUNT_PATH"
        --runai_large_shm "${RUNAI_LARGE_SHM:-true}"
    )

    # RoCE/GDR extended resources → JSON dict
    if [[ -n ${RUNAI_EXTENDED_RESOURCES:-} ]]; then
        # shellcheck disable=SC2086
        PLATFORM_ARGS+=(--runai_extended_resources_json "$(_kv_to_json $RUNAI_EXTENDED_RESOURCES)")
    fi

    # Pod annotations (Multus networks, etc.) → JSON dict
    if [[ -n ${RUNAI_ANNOTATIONS:-} ]]; then
        # shellcheck disable=SC2086
        PLATFORM_ARGS+=(--runai_annotations_json "$(_kv_to_json $RUNAI_ANNOTATIONS)")
    fi

    # Extra env vars for the training container (JSON passthrough)
    if [[ -n ${RUNAI_ENV_VARS_JSON:-} ]]; then
        PLATFORM_ARGS+=(--runai_env_json "$RUNAI_ENV_VARS_JSON")
    fi
else
    : "${SBATCH_ACCOUNT:?SBATCH_ACCOUNT is required for PLATFORM=slurm}"
    : "${SBATCH_PARTITION:?SBATCH_PARTITION is required for PLATFORM=slurm}"
    PLATFORM_ARGS=(
        --account "$SBATCH_ACCOUNT"
        --partition "$SBATCH_PARTITION"
        --packager none
    )
    # Extra sbatch flags (e.g. "nodelist=node001;reservation=my_res"), passed through verbatim.
    if [[ -n ${ADDITIONAL_SLURM_PARAMS:-} ]]; then
        PLATFORM_ARGS+=(--additional_slurm_params "${ADDITIONAL_SLURM_PARAMS}")
    fi
fi

# Always return success when sourced: the final optional `[[ ... ]] &&` above can
# evaluate false, which would otherwise make `source` return non-zero and abort a
# caller running under `set -e`.
:
