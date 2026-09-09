# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Merge the per-stage adapters from a layer-split run back into one usable checkpoint.

Merging is a UNION, not an average: `build_stage_model` replaces foreign layers with
`Identity()` rather than deleting them, so layer numbering is preserved and the stages' key
sets are disjoint by construction. If that stops holding this refuses rather than guessing,
because a half-populated adapter loads fine and quietly produces a worse model.
Dependency-light on purpose: the CLI imports this on every platform.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional, Tuple

ADAPTER_BIN = "adapter_model.safetensors"
ADAPTER_CFG = "adapter_config.json"
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def stage_dirs(root: str) -> List[str]:
    if not osp.isdir(root):
        raise RuntimeError(f"not a directory: {root}")
    found = []
    for name in os.listdir(root):
        m = re.fullmatch(r"stage(\d+)", name)
        if m and osp.isdir(osp.join(root, name)):
            found.append((int(m.group(1)), osp.join(root, name)))
    if not found:
        raise RuntimeError(
            f"no stage*/ directories in {root}. A layer-split run writes stage0/, stage1/, ...; "
            f"point this at the --save directory, not at one stage inside it."
        )
    found.sort()
    ranks = [r for r, _ in found]
    if ranks != list(range(len(ranks))):
        raise RuntimeError(
            f"stage directories are not contiguous from 0: found {ranks}. A missing stage means "
            f"missing layers, and merging would silently produce a partly-untrained adapter."
        )
    return [p for _, p in found]


def layer_of(key: str) -> Optional[int]:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def inspect_stage(path: str) -> Dict[str, Any]:
    from safetensors import safe_open  # local: keep module import cheap

    blob = osp.join(path, ADAPTER_BIN)
    if not osp.isfile(blob):
        raise RuntimeError(f"{blob} missing -- was this stage saved with --save?")
    keys, layers = [], set()
    with safe_open(blob, framework = "pt") as f:
        for k in f.keys():
            keys.append(k)
            n = layer_of(k)
            if n is not None:
                layers.add(n)
    return {"path": path, "keys": keys, "layers": sorted(layers), "n_keys": len(keys)}


def plan_merge(root: str) -> Dict[str, Any]:
    """Check that the stages compose into exactly one model. Pure inspection, writes nothing."""
    stages = [inspect_stage(p) for p in stage_dirs(root)]
    problems: List[str] = []

    # Two stages claiming one layer means the split is not what we think, and a union would
    # silently keep whichever was written last.
    seen: Dict[int, str] = {}
    for st in stages:
        for n in st["layers"]:
            if n in seen:
                problems.append(
                    f"layer {n} appears in BOTH {osp.basename(seen[n])} and "
                    f"{osp.basename(st['path'])} -- stages must own disjoint layers"
                )
            seen[n] = st["path"]

    # Gaps: contiguous coverage from 0 is what stage_layers() guarantees.
    covered = sorted(seen)
    if covered and covered != list(range(covered[0], covered[-1] + 1)):
        missing = sorted(set(range(covered[0], covered[-1] + 1)) - set(covered))
        problems.append(
            f"layers {missing} are in no stage -- the merged model would be untrained there"
        )
    if covered and covered[0] != 0:
        problems.append(f"layers 0..{covered[0]-1} are in no stage")

    # Every stage holds the non-layer modules, so a trained copy is indistinguishable from an
    # untouched one: a genuine ambiguity, not a tie to break.
    non_layer: Dict[str, List[str]] = {}
    for st in stages:
        for k in st["keys"]:
            if layer_of(k) is None:
                non_layer.setdefault(k, []).append(osp.basename(st["path"]))
    ambiguous = {k: v for k, v in non_layer.items() if len(v) > 1}

    return {
        "root": root,
        "stages": [
            {"path": s["path"], "layers": s["layers"], "n_keys": s["n_keys"]} for s in stages
        ],
        "n_stages": len(stages),
        "layers_covered": covered,
        "ambiguous_keys": ambiguous,
        "problems": problems,
        "ok": not problems,
    }


def merge(
    root: str,
    out: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan = plan_merge(root)
    if not plan["ok"] and not force:
        raise RuntimeError(
            "refusing to merge:\n  "
            + "\n  ".join(plan["problems"])
            + "\nA merged adapter with wrong or missing layers loads without error and quietly "
            "produces a worse model. Pass force=True only if you understand the consequence."
        )

    tensors: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for st in plan["stages"]:
        with safe_open(osp.join(st["path"], ADAPTER_BIN), framework = "pt") as f:
            for k in f.keys():
                n = layer_of(k)
                # Keep the FIRST copy and record it, not letting the last writer win invisibly.
                if k in tensors and n is None:
                    continue
                if k in tensors and n is not None and not force:
                    raise RuntimeError(f"duplicate layer tensor {k} -- stages overlap")
                tensors[k] = f.get_tensor(k)
                provenance[k] = osp.basename(st["path"])

    os.makedirs(out, exist_ok = True)
    save_file(tensors, osp.join(out, ADAPTER_BIN))

    # Verbatim: rank/alpha/targets are identical across stages by construction, and rewriting
    # would risk inventing a config nobody trained with.
    src_cfg = osp.join(plan["stages"][0]["path"], ADAPTER_CFG)
    if osp.isfile(src_cfg):
        with open(src_cfg, encoding = "utf-8") as f:
            cfg = json.load(f)
        with open(osp.join(out, ADAPTER_CFG), "w", encoding = "utf-8") as f:
            json.dump(cfg, f, indent = 2)

    return {
        "out": out,
        "n_tensors": len(tensors),
        "n_stages": plan["n_stages"],
        "layers": plan["layers_covered"],
        "from_stage": provenance,
        "ok": True,
    }


def _cmd_merge(
    root: str,
    out: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    plan = plan_merge(root)
    print(f"  stages   {plan['n_stages']}")
    for st in plan["stages"]:
        ls = st["layers"]
        span = f"{ls[0]}..{ls[-1]}" if ls else "(none)"
        print(f"    {osp.basename(st['path']):10s} layers {span:12s} {st['n_keys']} tensors")
    print(
        f"  layers   {len(plan['layers_covered'])} covered, contiguous from 0"
        if plan["ok"]
        else "  layers   INCOMPLETE"
    )
    if plan["ambiguous_keys"]:
        print(
            f"  note     {len(plan['ambiguous_keys'])} non-layer tensors exist in several "
            f"stages (e.g. embeddings); keeping stage0's copy"
        )
    for p in plan["problems"]:
        print(f"  PROBLEM  {p}")
    if not plan["ok"]:
        print("  refusing to merge -- a partly-populated adapter loads fine and trains worse")
        return 1
    if dry_run or not out:
        print(f"  would write {out or '<--out DIR>'}")
        return 0
    res = merge(root, out)
    print(f"  merged   {res['n_tensors']} tensors -> {res['out']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser("spark_merge", description = __doc__.split("\n")[0])
    ap.add_argument("root", help = "the --save directory containing stage0/, stage1/, ...")
    ap.add_argument("--out", default = None, help = "where to write the merged adapter")
    ap.add_argument("--dry-run", action = "store_true", help = "inspect and report, write nothing")
    a = ap.parse_args(argv)
    try:
        return _cmd_merge(a.root, a.out, a.dry_run)
    except RuntimeError as e:
        print(f"  {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
