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

"""Run:ai container image preparation.

Unlike Slurm (which builds an enroot ``.sqsh`` on shared storage), Run:ai pulls
OCI images directly onto each Kubernetes node on first use. Large images (the
NeMo container is ~20+ GB) can exceed the kubelet ``runtimeRequestTimeout`` when
several nodes pull simultaneously, leaving worker pods stuck in
``ImagePullBackOff``.

To avoid that on the first real benchmark, this module can submit a short
"image puller" Run:ai distributed job: one pod per node, each requesting a full
node's GPUs (so the scheduler spreads them across nodes), running ``sleep`` long
enough for the image to land in every node's containerd cache, then exiting.
"""

import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

# GPUs per node by GPU family. Mirrors the GPUS_PER_NODE logic in workload launch.sh.
_GPUS_PER_NODE_BY_GPU = {
    'gb300': 4,
    'gb200': 4,
    'b300': 8,
    'b200': 8,
    'h100': 8,
}


def gpus_per_node_for(gpu_type: str) -> int:
    """Return the GPUs-per-node for a GPU type (defaults to 8 if unknown)."""
    return _GPUS_PER_NODE_BY_GPU.get((gpu_type or '').lower(), 8)


def is_runai_cli_available() -> bool:
    """Check whether the ``runai`` CLI is on PATH."""
    return shutil.which('runai') is not None


def _sanitize_job_name(image_filename: str) -> str:
    """Build a DNS-1123-safe job name fragment from an image filename."""
    stem = re.sub(r'\.sqsh$', '', image_filename)
    name = re.sub(r'[^a-z0-9-]+', '-', stem.lower()).strip('-')
    return name[:40] or 'image'


def build_prepull_command(
    image_url: str,
    job_name: str,
    project_name: str,
    num_nodes: int,
    gpus_per_node: int,
    sleep_seconds: int = 120,
) -> List[str]:
    """Build a ``runai training pytorch submit`` command that warms every node.

    One master + (num_nodes - 1) workers, each requesting a full node's GPUs so
    the Run:ai scheduler places exactly one pod per node, forcing each node to
    pull ``image_url`` before the pods sleep and exit.
    """
    cmd = [
        'runai', 'training', 'pytorch', 'submit', job_name,
        '-p', project_name,
        '-i', image_url,
        '--gpu-devices-request', str(gpus_per_node),
    ]
    if num_nodes > 1:
        cmd += ['--workers', str(num_nodes - 1)]
    cmd += ['--command', '--', 'bash', '-c', f'echo image-prepull-ok && sleep {sleep_seconds}']
    return cmd


def prepull_images_runai(
    images: Dict[str, str],
    project_name: str,
    num_nodes: Optional[int],
    gpus_per_node: int,
    submit: bool = True,
    sleep_seconds: int = 120,
) -> None:
    """Warm every node's containerd cache for the required Run:ai images.

    Args:
        images: Mapping of image URL -> filename (from get_required_images()).
        project_name: Run:ai project to submit the puller job into.
        num_nodes: Cluster node count to spread the puller across. If None, the
            command is printed for the operator to run manually instead of submitted.
        gpus_per_node: GPUs requested per pod so the scheduler places one per node.
        submit: When False, only print the commands (dry run).
        sleep_seconds: How long each puller pod sleeps after the image lands.

    This never raises on failure: image pre-pull is an optimization, and the real
    benchmark will still pull on demand. Problems are logged as warnings.
    """
    if not images:
        return

    print("\nRun:ai container image preparation")
    print("----------------------------------")
    print("Run:ai pulls OCI images directly onto each node (no enroot/sqsh build).")
    for image_url, filename in sorted(images.items()):
        print(f"  - {image_url}")

    cli_ok = is_runai_cli_available()
    can_submit = submit and cli_ok and bool(num_nodes)

    if not cli_ok:
        print("\n'runai' CLI not found on PATH; skipping automatic pre-pull.")
    elif num_nodes is None:
        print(
            "\nCluster node count unknown; not submitting an automatic pre-pull job.\n"
            "To warm all nodes before the first benchmark, run one job per image:"
        )

    for image_url, filename in sorted(images.items()):
        job_name = f"prepull-{_sanitize_job_name(filename)}"
        cmd = build_prepull_command(
            image_url,
            job_name,
            project_name,
            num_nodes or 1,
            gpus_per_node,
            sleep_seconds=sleep_seconds,
        )
        printable = ' '.join(cmd)

        if not can_submit:
            print(f"  {printable}")
            continue

        print(f"\nSubmitting image-puller job for {image_url}")
        print(f"  $ {printable}")
        try:
            subprocess.run(cmd, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.warning("Run:ai image pre-pull submit failed for %s: %s", image_url, exc)
            print(f"  Warning: pre-pull submit failed ({exc}); the image will pull on first job.")
            continue
        # Give the scheduler/pull a head start before returning to the installer.
        print(f"  Submitted '{job_name}'. Allowing pods to pull (this may take several minutes)...")
        time.sleep(min(sleep_seconds, 15))

    if can_submit:
        print(
            "\nImage-puller job(s) submitted. They sleep briefly so every node caches the image, "
            "then exit. Check status with: runai training pytorch list -p " + project_name
        )
