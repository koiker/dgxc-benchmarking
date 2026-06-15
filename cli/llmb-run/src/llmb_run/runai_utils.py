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

"""Run:ai CLI utilities for llmb-run job tracking.

The job-history subsystem was built around Slurm (integer job ids, ``sacct``).
On the Run:ai platform there is no ``sacct``; status comes from the ``runai``
CLI instead. This module provides the Run:ai equivalents:

* :func:`parse_runai_job_handle` decodes the ``nemo_run``/Run:ai job handle into
  a stable integer surrogate (for the INTEGER primary key) plus the Run:ai
  workload name used for status queries.
* :func:`get_runai_job_statuses` queries ``runai training pytorch`` and returns
  records shaped like ``SlurmAccountingRecord`` so the existing history
  update/refresh code can consume them unchanged.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger('llmb_run.runai_utils')

RUNAI_TIMEOUT_SECONDS = 30

# Run:ai workload phases mapped onto the Slurm-style state vocabulary that
# job_history already understands (see TERMINAL_STATES / state styles there).
# Lower-cased keys; unknown phases pass through upper-cased so they still render.
_PHASE_TO_STATE = {
    "completed": "COMPLETED",
    "succeeded": "COMPLETED",
    "failed": "FAILED",
    "error": "FAILED",
    "stopped": "CANCELLED",
    "deleting": "PURGED",
    "deleted": "PURGED",
    "running": "RUNNING",
    "updating": "RUNNING",
    "degraded": "RUNNING",
    "terminating": "RUNNING",
    "pending": "PENDING",
    "creating": "PENDING",
    "initializing": "PENDING",
}

# State names (post-mapping) that mean the workload has stopped progressing.
_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "PURGED"}


@dataclass(frozen=True)
class RunaiAccountingRecord:
    """Mirror of slurm_utils.SlurmAccountingRecord for Run:ai workloads.

    The field names match SlurmAccountingRecord on purpose so job_history's
    ``_update_slurm_record`` can write either record type unchanged.
    """

    job_id: int
    state: str
    elapsed: str
    submit_time: str
    node_list: str
    exit_code: str


def runai_state_from_phase(phase: str | None) -> str:
    if not phase:
        return ""
    return _PHASE_TO_STATE.get(phase.strip().lower(), phase.strip().upper())


def parse_runai_job_handle(raw_job_id: object) -> tuple[int, str]:
    """Decode a ``nemo_run``/Run:ai job handle into ``(surrogate_id, workload_name)``.

    ``nemo_run`` emits a handle shaped like::

        <experiment>_<unix_ts>___<job_name>___<runai-workload-dns-name>

    The trailing integer of the first ``___`` segment is the experiment id
    (a unix timestamp, unique per submission); we use it as a stable integer
    surrogate for the history table's INTEGER primary key. The last ``___``
    segment is the Run:ai workload name used by ``runai ... describe/list``.

    Raises ``ValueError`` when no numeric id can be derived.
    """
    text = str(raw_job_id or '').strip()
    if not text:
        raise ValueError("Empty Run:ai job handle.")

    parts = text.split('___')
    workload_name = parts[-1].strip()

    match = re.search(r'(\d+)\s*$', parts[0])
    if not match:
        match = re.search(r'(\d+)', text)
    if not match:
        raise ValueError(f"Unable to derive a numeric job id from Run:ai handle '{raw_job_id}'.")

    return int(match.group(1)), (workload_name or parts[0].strip())


def _runai_env() -> dict[str, str]:
    env = dict(os.environ)
    # Keep the deprecation banner out of stdout so JSON parsing stays clean.
    env.setdefault("SUPPRESS_DEPRECATION_MESSAGE", "true")
    return env


def _run_runai(args: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["runai", *args],
            capture_output=True,
            text=True,
            timeout=RUNAI_TIMEOUT_SECONDS,
            env=_runai_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "").strip()
    return result.stdout, None


def list_runai_workloads(project: str) -> tuple[dict[str, dict] | None, str | None]:
    """Return ``{workload_name: workload_dict}`` for one project.

    Returns ``(None, error)`` when the CLI could not be queried so callers can
    distinguish "no workloads" from "could not refresh" (matching the sacct
    contract in slurm_utils).
    """
    out, err = _run_runai(["training", "pytorch", "list", "-p", project, "--json"])
    if out is None:
        return None, err

    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return None, f"unparseable `runai ... list --json` output: {e}"

    workloads: dict[str, dict] = {}
    for workload in data.get("workloads") or []:
        name = workload.get("name")
        if name:
            workloads[name] = workload
    return workloads, None


def describe_runai_workload(name: str, project: str) -> dict | None:
    out, err = _run_runai(["training", "pytorch", "describe", name, "-p", project, "-o", "json"])
    if out is None:
        logger.debug(f"runai describe failed for {name}: {err}")
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.debug(f"unparseable runai describe json for {name}: {e}")
        return None


def get_runai_job_statuses(project: str, name_to_id: dict[str, int]) -> dict[int, RunaiAccountingRecord] | None:
    """Return accounting records keyed by integer job id for the named workloads.

    ``name_to_id`` maps Run:ai workload name -> history surrogate job id. Names
    absent from ``runai list`` are simply omitted (the workload was deleted);
    the caller marks those PURGED, mirroring the sacct path. Returns ``None``
    when the CLI itself could not be queried.
    """
    if not name_to_id:
        return {}

    workloads, err = list_runai_workloads(project)
    if workloads is None:
        logger.warning(f"Unable to refresh Run:ai job status: {err}")
        return None

    records: dict[int, RunaiAccountingRecord] = {}
    for name, job_id in name_to_id.items():
        summary = workloads.get(name)
        if summary is None:
            continue  # absent from cluster -> caller decides (PURGED)

        state = runai_state_from_phase(summary.get("phase"))
        elapsed = ""
        node_list = ""
        submit_time = ""

        # describe carries timing + node placement; best-effort enrichment.
        detail = describe_runai_workload(name, project)
        if detail:
            d_state, elapsed, node_list, submit_time = _summarize_describe(detail)
            state = d_state or state

        records[job_id] = RunaiAccountingRecord(
            job_id=job_id,
            state=state,
            elapsed=elapsed,
            submit_time=submit_time,
            node_list=node_list,
            exit_code="",
        )

    return records


def _summarize_describe(detail: dict) -> tuple[str, str, str, str]:
    general = detail.get("general") or {}
    state = runai_state_from_phase(general.get("phase"))

    created = _parse_iso(general.get("createdAt"))
    updated = _parse_iso(general.get("updatedAt"))
    if created is not None:
        end = updated if (state in _TERMINAL_STATES and updated is not None) else _now_utc()
        elapsed = _format_elapsed(end - created)
    else:
        elapsed = ""

    nodes = sorted(
        {
            pod.get("nodeName")
            for pod in ((detail.get("podsInfo") or {}).get("pods") or [])
            if pod.get("nodeName")
        }
    )
    node_list = ",".join(nodes)

    submit_time = _normalize_iso(general.get("createdAt"))
    return state, elapsed, node_list, submit_time


def _parse_iso(value: object) -> datetime.datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _normalize_iso(value: object) -> str:
    dt = _parse_iso(value)
    if dt is None:
        return ""
    return dt.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _format_elapsed(delta: datetime.timedelta) -> str:
    total = int(max(delta.total_seconds(), 0))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
