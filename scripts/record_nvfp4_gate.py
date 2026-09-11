# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn an NVFP4 accuracy-gate run into a reviewed entry in ``nvfp4_gate_record.json``.

``scripts/prequant_accuracy_gate.py`` renders the 28 prompt/resolution pairs against the dense
reference and writes a ``results.json``. That file is evidence on one machine; what the auto
ladder consults is ``studio/backend/core/inference/nvfp4_gate_record.json``, which is checked in.
This script is the bridge, and its output is meant to land in a COMMIT that a human reviewed:

  python scripts/record_nvfp4_gate.py \\
      --results outputs/nvfp4_pr2/zimage_gate/results.json \\
      --checkpoint outputs/nvfp4_pr2/Z-Image-Turbo-NVFP4-P1.pt \\
      --family z-image --base-repo Tongyi-MAI/Z-Image-Turbo \\
      --policy-id zimg_f8mod_toq34_v1 \\
      --repo-id unsloth/Z-Image-Turbo-NVFP4 --filename Z-Image-Turbo-NVFP4.pt \\
      --backend flashinfer --cuda-graphs --gpu B200 \\
      --gate-script scripts/prequant_accuracy_gate.py

It fails closed on everything it can check without a GPU: the policy id must resolve in the tree
AND be the one ``resolve_policy`` gives for ``(family, base)``, the checkpoint must exist (its
sha256 is computed here, not taken on trust), a run that did not pass is refused unless
``--allow-fail`` records it explicitly as a failure for bookkeeping, and an entry whose key is
already in the file is refused rather than duplicated or silently overwritten -- two runs of one
checkpoint that disagree is a fact worth stopping on.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"
DEFAULT_RECORD = BACKEND / "core" / "inference" / "nvfp4_gate_record.json"


def sha256_file(path: Any, *, chunk: int = 1024 * 1024) -> str:
    """The sha256 of a file, read in chunks (a checkpoint is tens of gigabytes)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def record_key(record: dict) -> tuple:
    """The identity of a record: one gate run of one checkpoint at one policy version."""
    from core.inference.diffusion_nvfp4_gate import RECORD_KEY_FIELDS
    return tuple(str(record.get(field, "")).strip().lower() for field in RECORD_KEY_FIELDS)


def _resolutions(results: dict) -> list:
    """The distinct resolutions the run rendered at, in first-seen order."""
    seen: list = []
    for entry in results.get("results") or []:
        if not isinstance(entry, dict):
            continue
        value = entry.get("resolution")
        if value is None and entry.get("width") and entry.get("height"):
            value = f"{entry['width']}x{entry['height']}"
        if value is not None and value not in seen:
            seen.append(value)
    return seen


def _seeds(results: dict) -> list:
    """The distinct seeds the run used, in first-seen order."""
    seen: list = []
    for entry in results.get("results") or []:
        if isinstance(entry, dict) and entry.get("seed") is not None and entry["seed"] not in seen:
            seen.append(entry["seed"])
    return seen


def build_record(
    results: dict,
    *,
    family: str,
    base_repo: str,
    policy_id: str,
    checkpoint: Any,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    backend: Optional[str] = None,
    cuda_graphs: bool = False,
    gpu: Optional[str] = None,
    gptq: bool = False,
    results_path: Optional[str] = None,
    gate_script: Any = None,
) -> dict:
    """One gate record from a parsed ``results.json`` plus the checkpoint it was measured on.

    Raises when the policy does not resolve in the tree for this ``(family, base)``: a record
    naming a policy the runtime cannot re-resolve is unreadable evidence, and the reader would
    answer False for it anyway.
    """
    from core.inference.diffusion_nvfp4_policy import policy_by_id, resolve_policy

    policy = policy_by_id(policy_id)
    if policy is None:
        raise ValueError(f"no in-tree NVFP4 policy is called {policy_id!r}")
    resolved = resolve_policy(family, base_repo)
    if resolved is None or resolved.policy_id != policy.policy_id:
        raise ValueError(
            f"the tree resolves {(resolved.policy_id if resolved else None)!r} for "
            f"family {family!r} base {base_repo!r}, not {policy_id!r}; the gate measured a "
            "model this commit would not build"
        )
    script = Path(gate_script) if gate_script is not None else None
    return {
        "family": str(family).strip().lower(),
        "base_repo": str(base_repo).strip(),
        "policy_id": policy.policy_id,
        "policy_version": int(policy.version),
        "checkpoint_sha256": sha256_file(checkpoint),
        "repo_id": repo_id,
        "filename": filename or Path(checkpoint).name,
        "gptq": bool(gptq),
        "all_pass": bool(results.get("all_pass")),
        "num_pairs": results.get("num_pairs"),
        "num_passed": results.get("num_passed"),
        "aggregates": results.get("aggregates"),
        "gates": results.get("gates"),
        "resolutions": _resolutions(results),
        "seeds": _seeds(results),
        "cuda_graphs": bool(cuda_graphs),
        "backend": backend,
        "gpu": gpu,
        "versions": results.get("versions"),
        "gate_script_sha256": (
            sha256_file(script) if script is not None and script.is_file() else None
        ),
        "run_date": str(
            results.get("run_date") or datetime.date.today().isoformat()  # noqa: DTZ011
        ),
        "results_path": results_path,
    }


def append_record(
    path: Any,
    record: dict,
    *,
    allow_fail: bool = False,
) -> dict:
    """Append ``record`` to the gate file at ``path`` and write it back.

    Refuses a failed run unless ``allow_fail`` (a record exists to LIFT a deny; storing a failure
    by accident is how a deny gets lifted by evidence that says the opposite), and refuses a key
    the file already carries.
    """
    from core.inference.diffusion_nvfp4_gate import GATE_RECORD_VERSION

    if not record.get("all_pass") and not allow_fail:
        raise ValueError(
            "this run did not pass every pair (all_pass is false); pass --allow-fail to record "
            "it as a failure for bookkeeping, which does NOT enable nvfp4 for this base"
        )
    target = Path(path)
    document: Any = {"version": GATE_RECORD_VERSION, "records": []}
    if target.is_file():
        document = json.loads(target.read_text(encoding = "utf-8"))
    records = list(document.get("records") or [])
    key = record_key(record)
    if any(record_key(existing) == key for existing in records):
        raise ValueError(
            f"{target} already carries a record for {key}; edit or remove it rather than "
            "appending a second verdict for the same checkpoint"
        )
    records.append(record)
    document["records"] = records
    document.setdefault("version", GATE_RECORD_VERSION)
    target.write_text(json.dumps(document, indent = 2, sort_keys = False) + "\n", encoding = "utf-8")
    return document


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument("--results", required = True, help = "prequant_accuracy_gate.py results.json")
    p.add_argument("--checkpoint", required = True, help = "the artifact the gate ran on")
    p.add_argument("--family", required = True)
    p.add_argument("--base-repo", required = True, help = "upstream base the checkpoint was baked from")
    p.add_argument("--policy-id", required = True, help = "e.g. zimg_f8mod_toq34_v1")
    p.add_argument("--repo-id", default = None, help = "hosted repo the artifact publishes under")
    p.add_argument("--filename", default = None, help = "artifact filename (default: the file's)")
    p.add_argument("--backend", default = None, help = "nvfp4 backend the gate ran: torchao|flashinfer")
    p.add_argument("--cuda-graphs", action = "store_true", help = "the gate ran with graphs captured")
    p.add_argument("--gpu", default = None, help = "the GPU the gate ran on, e.g. B200")
    p.add_argument("--gptq", action = "store_true", help = "the checkpoint carries GPTQ correction")
    p.add_argument(
        "--gate-script",
        default = None,
        help = "the gate script that produced --results, hashed into the record. Defaults to a "
        "sibling prequant_accuracy_gate.py, which is absent when the gate was driven from a "
        "harness outside this repo; pass it so the record still names what ran.",
    )
    p.add_argument("--record", default = str(DEFAULT_RECORD), help = "gate record file to append to")
    p.add_argument(
        "--allow-fail",
        action = "store_true",
        help = "record a run that did NOT pass (bookkeeping only; it enables nothing)",
    )
    args = p.parse_args(argv)

    sys.path.insert(0, str(BACKEND))
    try:
        results = json.loads(Path(args.results).read_text(encoding = "utf-8"))
        record = build_record(
            results,
            family = args.family,
            base_repo = args.base_repo,
            policy_id = args.policy_id,
            checkpoint = args.checkpoint,
            repo_id = args.repo_id,
            filename = args.filename,
            backend = args.backend,
            cuda_graphs = args.cuda_graphs,
            gpu = args.gpu,
            gptq = args.gptq,
            results_path = str(args.results),
            gate_script = (
                Path(args.gate_script).resolve()
                if args.gate_script
                else Path(__file__).resolve().parent / "prequant_accuracy_gate.py"
            ),
        )
        append_record(args.record, record, allow_fail = args.allow_fail)
    except Exception as exc:  # noqa: BLE001 -- a tool reports, it does not traceback
        print(f"error: {exc}", flush = True)
        return 2
    print(
        f"recorded {record['family']} {record['policy_id']} v{record['policy_version']} "
        f"all_pass={record['all_pass']} sha256={record['checkpoint_sha256'][:12]} -> {args.record}",
        flush = True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
