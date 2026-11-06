# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Each wired leg's checkpoint must be on the prefetch lane, or the leg downloads it on its own card."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from kaggle_t4_ci import legs  # noqa: E402


def _payload_default(entry: str) -> str | None:
    """The `--model` default the payload itself carries."""
    source = (PAYLOAD_DIR / entry).read_text(encoding = "utf-8")
    if 'ap.add_argument("--model", default = DEFAULT_MODEL)' not in source:
        return None
    match = re.search(r'^DEFAULT_MODEL = "([^"]+)"', source, re.MULTILINE)
    return match.group(1) if match else None


def models_for(leg) -> set[str]:
    args = list(leg.args)
    if "--model" in args:
        named = args[args.index("--model") + 1]
    else:
        named = _payload_default(leg.entry)
    if not named:
        return set()
    return {named, legs.LOAD_REDIRECTS.get(named, named)}


def test_every_wired_leg_loads_a_prefetched_checkpoint():
    wired = [name for kernel in legs.KERNELS for name in kernel]
    missing = {}
    for name in wired:
        wanted = models_for(legs.LEGS[name])
        if wanted and not (wanted & set(legs.PREFETCH_REPOS)):
            missing[name] = sorted(wanted)
    assert not missing, (
        "these wired legs download their checkpoint on an allocated card "
        f"instead of on the free prefetch lane: {missing}. Add them to "
        "PREFETCH_REPOS."
    )


def test_the_model_walk_reads_the_payload_default_and_not_only_the_args():
    """Three of the five wired legs carry no --model at all. A rule that read
    the args alone would report them as having no checkpoint and pass by
    finding nothing, which is the shape of a guard that guards nothing."""
    assert models_for(legs.LEGS["canary"]) == {"unsloth/Qwen2.5-0.5B-Instruct"}
    assert "unsloth/gpt-oss-20b-unsloth-bnb-4bit" in models_for(legs.LEGS["gptoss"]), (
        "the gpt-oss redirect is not being applied, so the prefetch would warm "
        "the 16-bit repo that no sm_75 run ever loads"
    )


def test_the_critical_path_leg_is_fetched_before_the_one_with_slack():
    """The lane fetches in list order, so the leg that sets the makespan goes first."""
    order = list(legs.PREFETCH_REPOS)
    assert order.index("unsloth/Qwen3.5-2B") < order.index(
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    ), "the leg that sets the makespan is queued behind the leg with 500s of slack"


def test_nothing_is_prefetched_that_no_leg_reads():
    """A repo no leg loads is wasted bandwidth, so every leg is checked, not only the wired ones."""
    loaded = set()
    for leg in legs.LEGS.values():
        loaded |= models_for(leg)
    # Studio's models are fetched by the Studio builder under its own HF_HOME.
    stray = [repo for repo in legs.PREFETCH_REPOS if repo not in loaded]
    assert not stray, f"prefetched but never loaded by any leg: {stray}"


def test_every_redirect_target_is_prefetched_under_its_EXACT_name():
    """The HF cache keys on the literal repo string, so the prefetch needs the exact casing."""
    prefetch = set(legs.PREFETCH_REPOS)
    wired = {name for kernel in legs.KERNELS for name in kernel}
    for leg_name in wired:
        leg = legs.LEGS[leg_name]
        args = list(leg.args)
        named = (
            args[args.index("--model") + 1] if "--model" in args else _payload_default(leg.entry)
        )
        if named is None or named not in legs.LOAD_REDIRECTS:
            continue
        target = legs.LOAD_REDIRECTS[named]
        assert target in prefetch, (
            f"{leg_name} loads {target!r}, which is not prefetched under that "
            f"exact string; a case-different entry does not warm the same cache"
        )


def test_the_qwen3_redirect_matches_the_measured_case():
    """Repo casing must match the loader's current canonical spelling as shown in a kernel report."""
    assert legs.LOAD_REDIRECTS["unsloth/Qwen3-0.6B"] == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"


def test_a_blob_is_counted_once_not_once_per_symlink(tmp_path, monkeypatch):
    """Snapshots symlink into blobs/, so a walk must count each blob once or reported throughput doubles."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "kaggle_prefetch_under_test", ROOT / ".github" / "scripts" / "kaggle_prefetch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    folder = tmp_path / "hub" / "models--org--model"
    blobs = folder / "blobs"
    snapshot = folder / "snapshots" / "rev"
    blobs.mkdir(parents = True)
    snapshot.mkdir(parents = True)
    for index in range(3):
        blob = blobs / f"sha{index}"
        blob.write_bytes(b"x" * 1000)
        (snapshot / f"shard{index}.safetensors").symlink_to(blob)
    # Without symlinks a config is a real file in both places, another double count.
    (blobs / "cfg").write_bytes(b"y" * 10)
    os.link(blobs / "cfg", snapshot / "config.json")

    source = module.prefetch_cell(repos = [("org/model", None)], hf_home = str(tmp_path))
    match = re.search(r"def _repo_bytes\(repo\):.*?\n\ndef ", source, re.S)
    assert match, "the generated cell no longer defines _repo_bytes"
    namespace = {"os": os}
    exec(match.group(0)[: -len("\n\ndef ")], namespace)  # noqa: S102

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert namespace["_repo_bytes"]("org/model") == 3010, (
        "the walk counts a blob once per link to it, so every reported size and "
        "MB/s in the prefetch evidence is inflated"
    )
