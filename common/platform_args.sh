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
#   runai  NVIDIA Run:ai via the `runai` CLI (SSO `runai login`, no app creds).
#          Requires DGXC_PROJECT_NAME, DGXC_PVC_CLAIM_NAME. Optional:
#          DGXC_PVC_MOUNT_PATH, RUNAI_EXTENDED_RESOURCES, RUNAI_ANNOTATIONS,
#          RUNAI_NODE_POOLS, RUNAI_LARGE_SHM, RUNAI_RAILS_ON_MASTER,
#          RUNAI_EXTRA_SUBMIT_ARGS, RUNAI_PRINT_ONLY.
#   dgxc   DGX Cloud REST API. Requires DGXC_BASE_URL, DGXC_APP_ID,
#          DGXC_APP_SECRET, DGXC_PROJECT_NAME, DGXC_PVC_CLAIM_NAME. Optional:
#          DGXC_PVC_MOUNT_PATH, DGXC_CLUSTER, DGXC_KUBE_APISERVER_URL.
#
# RoCE/GDR rails and Multus annotations are space-separated lists and are
# expanded into repeated --runai_* flags.

PLATFORM=${PLATFORM:-slurm}
PLATFORM=${PLATFORM,,}

PLATFORM_ARGS=()

if [[ $PLATFORM == "runai" ]]; then
    # Run:ai via the `runai` CLI — analogous to sbatch: requires a prior `runai login` (SSO),
    # NOT a Run:ai Application. No DGXC_BASE_URL / DGXC_APP_ID / DGXC_APP_SECRET needed.
    : "${DGXC_PROJECT_NAME:?DGXC_PROJECT_NAME is required for PLATFORM=runai (e.g. nccl-benchmarking)}"
    : "${DGXC_PVC_CLAIM_NAME:?DGXC_PVC_CLAIM_NAME is required for PLATFORM=runai (e.g. nemo-workspace)}"
    DGXC_PVC_MOUNT_PATH=${DGXC_PVC_MOUNT_PATH:-/nemo-workspace}
    PLATFORM_ARGS=(
        --platform runai
        --dgxc_project_name "$DGXC_PROJECT_NAME"
        --dgxc_pvc_claim_name "$DGXC_PVC_CLAIM_NAME"
        --dgxc_pvc_mount_path "$DGXC_PVC_MOUNT_PATH"
    )
    # RoCE/GDR rails + Multus network annotations (space-separated lists -> repeated flags).
    for _res in ${RUNAI_EXTENDED_RESOURCES:-}; do PLATFORM_ARGS+=(--runai_extended_resource "$_res"); done
    for _ann in ${RUNAI_ANNOTATIONS:-}; do PLATFORM_ARGS+=(--runai_annotation "$_ann"); done
    for _ex in ${RUNAI_EXTRA_SUBMIT_ARGS:-}; do PLATFORM_ARGS+=(--runai_extra_submit_arg "$_ex"); done
    [[ -n ${RUNAI_NODE_POOLS:-} ]] && PLATFORM_ARGS+=(--runai_node_pools "$RUNAI_NODE_POOLS")
    [[ -n ${RUNAI_LARGE_SHM:-} ]] && PLATFORM_ARGS+=(--runai_large_shm "$RUNAI_LARGE_SHM")
    [[ -n ${RUNAI_RAILS_ON_MASTER:-} ]] && PLATFORM_ARGS+=(--runai_rails_on_master "$RUNAI_RAILS_ON_MASTER")
    [[ ${RUNAI_PRINT_ONLY:-0} == "1" ]] && PLATFORM_ARGS+=(--runai_print_only)
elif [[ $PLATFORM == "dgxc" ]]; then
    : "${DGXC_BASE_URL:?DGXC_BASE_URL is required for PLATFORM=dgxc}"
    : "${DGXC_APP_ID:?DGXC_APP_ID is required for PLATFORM=dgxc}"
    : "${DGXC_APP_SECRET:?DGXC_APP_SECRET is required for PLATFORM=dgxc}"
    : "${DGXC_PROJECT_NAME:?DGXC_PROJECT_NAME is required for PLATFORM=dgxc}"
    : "${DGXC_PVC_CLAIM_NAME:?DGXC_PVC_CLAIM_NAME is required for PLATFORM=dgxc}"
    DGXC_PVC_MOUNT_PATH=${DGXC_PVC_MOUNT_PATH:-/nemo-workspace}
    PLATFORM_ARGS=(
        --platform dgxc
        --dgxc_base_url "$DGXC_BASE_URL"
        --dgxc_app_id "$DGXC_APP_ID"
        --dgxc_app_secret "$DGXC_APP_SECRET"
        --dgxc_project_name "$DGXC_PROJECT_NAME"
        --dgxc_pvc_claim_name "$DGXC_PVC_CLAIM_NAME"
        --dgxc_pvc_mount_path "$DGXC_PVC_MOUNT_PATH"
    )
    [[ -n ${DGXC_CLUSTER:-} ]] && PLATFORM_ARGS+=(--dgxc_cluster "$DGXC_CLUSTER")
    [[ -n ${DGXC_KUBE_APISERVER_URL:-} ]] && PLATFORM_ARGS+=(--dgxc_kube_apiserver_url "$DGXC_KUBE_APISERVER_URL")
else
    : "${SBATCH_ACCOUNT:?SBATCH_ACCOUNT is required for PLATFORM=slurm}"
    : "${SBATCH_PARTITION:?SBATCH_PARTITION is required for PLATFORM=slurm}"
    PLATFORM_ARGS=(
        --platform slurm
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
