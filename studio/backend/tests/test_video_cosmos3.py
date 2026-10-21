# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano and Cosmos3-Edge on diffusers' ``Cosmos3OmniPipeline``: the ``core.inference.video_cosmos3`` glue (the
repo layout the download planner checks, NVIDIA's Cosmos guardrail precheck and its opt-out, the ``from_pretrained``
kwargs, the output adapter, the untiled encode of the conditioning clip), the model cards' schedules on the real
scheduler, and VideoBackend's Cosmos3 paths: the load, the render, the download plan, local folders and the validation
refusals. The family rows themselves are pinned in test_video_cosmos3_families.py.

Hermetic: no network, no weights, no GPU. VideoBackend runs for real over test_video_backend's fake torch/diffusers
runtime with ``diffusers.Cosmos3OmniPipeline`` replaced by ``_FakeCosmos3Pipeline``, whose ``__call__`` takes exactly
the keywords of the real one. Hub listings and configs come from ``fixtures/cosmos3_checkpoints.json``: public Hub
metadata (file listings with sizes, model_index.json, the root, transformer, scheduler and VAE configs) at pinned
revisions: nvidia/Cosmos3-Nano@e59a53c2 and nvidia/Cosmos3-Edge@344d602b for the two families, whose model cards the
"README.md:<line>" citations refer to, plus the Policy-DROID and Super builds. Diffusers lines are at d42ebf9, the
Studio pin. Checks against the real UniPCMultistepScheduler and Cosmos3OmniPipeline need diffusers and skip without it,
as on Backend CI."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import importlib.machinery
import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusion_batched
from core.inference import video as video_mod
from core.inference import video_cosmos3 as vc3
from core.inference.diffusion_families import (
    _installed_at_least,
    local_pipeline_components_are_complete,
    pipeline_class_requirement,
)
from core.inference.diffusion_flow_shift import apply_scheduler_overrides
from core.inference.video import VideoBackend
from core.inference.video_families import VIDEO_CANCELLED_MSG, detect_video_family
from utils import hf_cache_settings

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# The fake runtime and the autouse pins every VideoBackend test runs under.
from test_video_backend import (  # noqa: E402,F401
    _FakeWanDiT,
    _FakeWanVae,
    _assume_the_restricted_load_is_available,
    _data_url,
    _pin_video_fetch_to_the_named_repo,
    _roomy_host_ram,
    fake_runtime,
)

NANO = "nvidia/Cosmos3-Nano"
EDGE = "nvidia/Cosmos3-Edge"
_FAMILY = {NANO: "cosmos3-nano", EDGE: "cosmos3-edge"}
# Not a base repo; its name token makes it a cosmos3-nano pick.
_UNKNOWN_REPO = "someone/cosmos3-nano-repack"
# Spelled out rather than read from video_cosmos3: the refusal prints it for the user to type.
_OPT_OUT_ENV = "UNSLOTH_COSMOS3_SAFETY_CHECKER"
# Public Hub metadata at the pinned revisions, keyed "<repo>@<sha>": the file listing with sizes, model_index.json,
# the root config's model_type, and the transformer / scheduler / VAE configs.
_CHECKPOINTS = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "cosmos3_checkpoints.json").read_text(
        encoding = "utf-8"
    )
)


def _pinned(repo: str) -> tuple[str, dict]:
    """(revision, fixture entry) for ``repo``; the fixture pins exactly one revision per repo."""
    (key,) = [k for k in _CHECKPOINTS if k.partition("@")[0] == repo]
    return key.partition("@")[2], _CHECKPOINTS[key]


def _diffusers_with_cosmos3():
    """The installed diffusers, for the checks against the real Cosmos3 classes. Skips where they cannot exist: Backend
    CI has no diffusers, and Python 3.9 installs pin diffusers 0.36.0 (requirements/diffusers-pin.txt), which predates
    Cosmos3OmniPipeline. The version is read on its release, as Studio's listing gate reads it, so a diffusers new
    enough to have the classes fails these checks if it lost them instead of skipping."""
    diffusers = pytest.importorskip(
        "diffusers"
    )  # Backend CI has no diffusers; this path runs on it
    minimum, _ = pipeline_class_requirement("Cosmos3OmniPipeline")
    if minimum and not _installed_at_least(diffusers.__version__, minimum):
        pytest.skip(
            f"diffusers {diffusers.__version__} predates Cosmos3OmniPipeline (>= {minimum})"
        )
    return diffusers


def _checkpoint(repo: str) -> dict:
    return _pinned(repo)[1]


# The two base repos at the revisions the families were written against.
_NANO = _checkpoint(NANO)
_EDGE = _checkpoint(EDGE)

# Other families' Diffusers layouts, names only: a Wan2.2 Diffusers repo, and trimmed copies of the listings the
# suite plans downloads with (tests/test_video_ltx25.py _PACK, tests/test_video_backend.py _H3_SIBLINGS). Each has
# text_encoder/ and tokenizer/; the LTX-2.5 pack and MiniMax-H3 also ship the root modular_model_index.json that
# Cosmos3 repos carry.
_WAN_LISTING = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_LTX25_LISTING = (
    "model_index.json",
    "modular_model_index.json",
    "connectors/config.json",
    "processor/tokenizer.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "vae/config.json",
    "vocoder/config.json",
)
_H3_LISTING = (
    "model_index.json",
    "modular_model_index.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "processor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model-00001-of-00001.safetensors",
    "tokenizer/chat_template.jinja",
    "transformer/config.json",
    "transformer_ref/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


def _opt_out(monkeypatch, value):
    """Set the guardrail opt-out, or clear it (None) so a developer shell that exports it does not leak in."""
    if value is None:
        monkeypatch.delenv(vc3.SAFETY_CHECKER_ENV, raising = False)
    else:
        monkeypatch.setenv(vc3.SAFETY_CHECKER_ENV, value)


def _guardrail_package(
    monkeypatch,
    present,
    error = None,
):
    """Answer ``find_spec("cosmos_guardrail")``: present, absent, or raising ``error``. Every other name resolves as
    usual. Returns the names asked about."""
    real_find_spec = importlib.util.find_spec
    asked = []

    def find_spec(name, package = None):
        if name != vc3.GUARDRAIL_PACKAGE:
            return real_find_spec(name, package)
        asked.append(name)
        if error is not None:
            raise error
        return importlib.machinery.ModuleSpec(name, None) if present else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return asked


def _set_guardrail(monkeypatch, state: str) -> None:
    """For the VideoBackend tests, the state an install leaves: ``state`` is "opted out"
    (UNSLOTH_COSMOS3_SAFETY_CHECKER=0, no package), "installed" (cosmos_guardrail importable, no opt-out) or "absent"
    (neither). The package is set through sys.modules, which importlib.util.find_spec reads first. Studio's own
    environment ships without it."""
    _opt_out(monkeypatch, "0" if state == "opted out" else None)
    if state == "installed":
        module = types.ModuleType("cosmos_guardrail")
        module.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
        monkeypatch.setitem(sys.modules, "cosmos_guardrail", module)
    else:
        # A None entry makes importlib.util.find_spec answer None: not importable, whatever this host has installed.
        monkeypatch.setitem(sys.modules, "cosmos_guardrail", None)


def _hub_cache(monkeypatch, answer):
    """Stand in for ``huggingface_hub.try_to_load_from_cache``: return ``answer``, or raise it when it is an
    exception. Returns the (repo_id, filename) lookups."""
    lookups = []

    def try_to_load_from_cache(repo_id, filename, *args, **kwargs):
        lookups.append((repo_id, filename))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", try_to_load_from_cache)
    return lookups


def _write_index(folder, index):
    folder.mkdir(parents = True, exist_ok = True)
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    return folder


class _Log:
    """Records ``logger.<level>(fmt, *args)``; the backend logger is structlog, which caplog does not see."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args, **kwargs):
        self.lines.append(("info", fmt % args if args else fmt))

    def warning(self, fmt, *args, **kwargs):
        self.lines.append(("warning", fmt % args if args else fmt))


class _FakeCosmos3Vae(_FakeWanVae):
    """``AutoencoderKLWan`` as the Cosmos3 load and render touch it: diffusers' ``use_tiling`` flag, which
    ``enable_tiling`` sets, and an ``encode`` and ``decode`` that log whether tiling was on at each call (``calls``).
    ``Cosmos3OmniPipeline._encode_video`` calls ``vae.encode`` once per render, on the conditioning clip
    (pipeline_cosmos3_omni.py:507-515, :866-884). ``encode_raises`` holds errors for the next encodes to raise."""

    def __init__(self) -> None:
        super().__init__()
        self.use_tiling = False
        self.calls: list = []
        self.encode_raises: list = []

    def enable_tiling(self) -> None:
        super().enable_tiling()
        self.use_tiling = True

    def disable_tiling(self) -> None:
        self.tiled = False
        self.use_tiling = False

    def encode(self, *args, **kwargs):
        self.calls.append("encode tiled" if self.use_tiling else "encode untiled")
        if self.encode_raises:
            raise self.encode_raises.pop(0)
        return ("latents", args, kwargs)

    def decode(self, *args, **kwargs):
        self.calls.append("decode tiled" if self.use_tiling else "decode untiled")
        return super().decode(*args, **kwargs)


# torch.OutOfMemoryError (torch.cuda.OutOfMemoryError) by its name, without importing torch: is_oom_error matches the
# class name as well as the message.
_TorchOutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})


def _record_cuda_cache_flushes(monkeypatch, calls: list) -> None:
    """``import torch`` inside the out-of-memory retry gets a stand-in whose ``cuda.empty_cache`` appends
    "empty_cache" to ``calls``, so the order of flush and retry shows on any host and no real torch is touched."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache = lambda: calls.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", torch)


# ── module glue / guardrail ──────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(_CHECKPOINTS))
def test_every_cosmos3_fixture_listing_has_the_cosmos3_layout(key):
    # Each Cosmos3 Diffusers repo ships a root modular_model_index.json and a text_tokenizer/ (the transformer is its
    # own text model). Without this check the download planner read them as MiniMax-H3's modular layout and dropped
    # text_tokenizer/, which Cosmos3OmniPipeline.__init__ requires. The planner passes a one-shot generator.
    listing = _CHECKPOINTS[key]["listing"]
    assert any(name == "modular_model_index.json" for name, _size in listing)
    assert vc3.is_cosmos3_listing(name for name, _size in listing) is True


@pytest.mark.parametrize(
    "listing",
    [_WAN_LISTING, _LTX25_LISTING, _H3_LISTING],
    ids = ["wan2.2", "ltx-2.5-pack", "minimax-h3"],
)
def test_wan_ltx25_and_h3_listings_are_not_read_as_cosmos3(listing):
    # Each has text_encoder/ and tokenizer/, never text_tokenizer/. For H3, which also ships modular_model_index.json,
    # this keeps the planner's H3 filter engaged.
    assert vc3.is_cosmos3_listing(iter(listing)) is False


@pytest.mark.parametrize(
    "listing, expected",
    [
        ((), False),
        (("model_index.json", "text_tokenizer/tokenizer.json"), True),
        # A Cosmos3 listing that gained a text encoder (a repack with an external one) is not this layout.
        (
            tuple(name for name, _size in _EDGE["listing"]) + ("text_encoder/model.safetensors",),
            False,
        ),
        # Folders only: a root file named like the folder is not the folder.
        (("model_index.json", "text_tokenizer.json", "tokenizer/tokenizer.json"), False),
    ],
    ids = ["empty", "tokenizer-only", "edge-plus-text-encoder", "root-file-only"],
)
def test_the_cosmos3_layout_needs_a_text_tokenizer_folder_and_no_text_encoder(listing, expected):
    # A one-shot iterator, as the planner passes it, in sorted order: text_encoder/ comes before text_tokenizer/,
    # so a check that walked the iterator twice would no longer see the encoder.
    assert vc3.is_cosmos3_listing(iter(sorted(listing))) is expected


@pytest.mark.parametrize(
    "value, opted_out",
    [
        ("0", True),
        ("false", True),
        ("FALSE", True),
        ("no", True),
        ("off", True),
        (" Off\n", True),
        (None, False),
        ("", False),
        ("1", False),
        ("true", False),
        ("yes", False),
        ("on", False),
        ("disabled", False),
    ],
)
def test_the_cosmos3_safety_checker_opt_out_takes_only_explicit_off_values(
    monkeypatch, value, opted_out
):
    # Turning a safety filter off has to be explicit: any other value, or none, keeps the checkpoint's choice.
    assert vc3.SAFETY_CHECKER_ENV == "UNSLOTH_COSMOS3_SAFETY_CHECKER"
    _opt_out(monkeypatch, value)
    assert vc3.safety_checker_opted_out() is opted_out


@pytest.mark.parametrize("present", [True, False])
def test_cosmos3_guardrail_presence_is_find_spec_on_cosmos_guardrail(monkeypatch, present):
    # cosmos_guardrail is the module diffusers imports CosmosSafetyChecker from (pipeline_cosmos3_omni.py:50-51 at
    # d42ebf9); find_spec answers without importing it.
    asked = _guardrail_package(monkeypatch, present)
    assert vc3.GUARDRAIL_PACKAGE == "cosmos_guardrail"
    assert vc3.guardrail_installed() is present
    assert asked == ["cosmos_guardrail"]


@pytest.mark.parametrize(
    "error",
    [
        ImportError("No module named 'cosmos_guardrail'"),
        ValueError("cosmos_guardrail.__spec__ is None"),
    ],
    ids = ["ImportError", "ValueError"],
)
def test_a_find_spec_error_reads_as_cosmos3_guardrail_missing(monkeypatch, error):
    # find_spec raises ValueError for a module left in sys.modules without a spec (a half-done import) and
    # ImportError when a finder fails: a broken install counts as no install.
    _guardrail_package(monkeypatch, True, error = error)
    assert vc3.guardrail_installed() is False


def test_the_cosmos3_guardrail_package_is_the_one_the_real_pipeline_imports():
    _diffusers_with_cosmos3()
    from diffusers.pipelines.cosmos import pipeline_cosmos3_omni

    # Studio checks for this package before the download; diffusers imports the checker from it and raises
    # ImportError in __init__ without it (pipeline_cosmos3_omni.py:50-58 at d42ebf9), after the download.
    tree = ast.parse(inspect.getsource(pipeline_cosmos3_omni))
    imported = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (vc3.GUARDRAIL_PACKAGE, "CosmosSafetyChecker") in imported


@pytest.mark.parametrize(
    "repo, index, enabled",
    [
        # Nano's model_index.json leaves the key out, so Cosmos3OmniPipeline.__init__'s default (True) applies.
        (NANO, _NANO["model_index"], True),
        # Edge's ships "enable_safety_checker": false.
        (EDGE, _EDGE["model_index"], False),
        # The index in hand decides, whatever the base-repo table says.
        (EDGE, {**_EDGE["model_index"], "enable_safety_checker": True}, True),
        (NANO, {**_NANO["model_index"], "enable_safety_checker": False}, False),
        (_UNKNOWN_REPO, {}, True),
        # Only a JSON false turns it off. A string is truthy to the pipeline's __init__ too, which then builds the
        # checker (pipeline_cosmos3_omni.py:437-439 at d42ebf9).
        (EDGE, {**_EDGE["model_index"], "enable_safety_checker": "false"}, True),
    ],
    ids = ["nano", "edge", "edge-index-on", "nano-index-off", "empty-index", "string-false"],
)
def test_an_index_dict_decides_the_cosmos3_safety_checker(monkeypatch, repo, index, enabled):
    lookups = _hub_cache(monkeypatch, None)
    assert vc3.checkpoint_enables_safety_checker(repo, index) is enabled
    assert lookups == []  # an index in hand is never looked up


@pytest.mark.parametrize(
    "folder, index, enabled",
    [
        ("Cosmos3-Nano", _NANO["model_index"], True),
        ("Cosmos3-Edge", _EDGE["model_index"], False),
        # The folder's own index decides, not its name.
        ("Cosmos3-Edge", _NANO["model_index"], True),
        ("Cosmos3-Nano", _EDGE["model_index"], False),
    ],
    ids = ["nano", "edge", "nano-index-in-edge-folder", "edge-index-in-nano-folder"],
)
def test_a_local_folder_index_decides_the_cosmos3_safety_checker_not_its_name(
    monkeypatch, tmp_path, folder, index, enabled
):
    lookups = _hub_cache(monkeypatch, None)
    path = _write_index(tmp_path / folder, index)
    assert vc3.checkpoint_enables_safety_checker(str(path)) is enabled
    assert lookups == []  # a local folder never asks the Hub cache


@pytest.mark.parametrize(
    "content",
    [None, "{not json", "[]", '"enable_safety_checker"', b"\xff\xfe\x00"],
    ids = ["missing", "malformed", "a-list", "a-string", "not-utf8"],
)
def test_an_unreadable_local_index_keeps_the_cosmos3_safety_checker_on(
    monkeypatch, tmp_path, content
):
    # Named like Edge, whose shipped index turns the checker off: a folder's name never turns it off.
    folder = tmp_path / "Cosmos3-Edge"
    folder.mkdir()
    if isinstance(content, bytes):
        (folder / "model_index.json").write_bytes(content)
    elif content is not None:
        (folder / "model_index.json").write_text(content, encoding = "utf-8")
    lookups = _hub_cache(monkeypatch, None)
    assert vc3._cached_model_index(str(folder)) is None
    assert vc3.checkpoint_enables_safety_checker(str(folder)) is True
    assert lookups == []


@pytest.mark.parametrize(
    "repo, cached, enabled",
    [
        (NANO, _NANO["model_index"], True),
        (EDGE, _EDGE["model_index"], False),
        # A later revision that flips the setting is honoured from the cache, over the base-repo table.
        (EDGE, {**_EDGE["model_index"], "enable_safety_checker": True}, True),
        (NANO, {**_NANO["model_index"], "enable_safety_checker": False}, False),
    ],
    ids = ["nano", "edge", "edge-flipped-on", "nano-flipped-off"],
)
def test_the_hf_cache_index_decides_the_cosmos3_safety_checker_over_the_table(
    monkeypatch, tmp_path, repo, cached, enabled
):
    monkeypatch.chdir(tmp_path)  # a Hub id is also a relative path; no folder of that name here
    blob = _write_index(tmp_path / "blob", cached) / "model_index.json"
    lookups = _hub_cache(monkeypatch, str(blob))
    assert vc3.checkpoint_enables_safety_checker(repo) is enabled
    assert lookups == [(repo, "model_index.json")]


@pytest.mark.parametrize(
    "answer",
    [None, object(), ValueError("Repo id must use alphanumeric chars or '-', '_', '.'")],
    ids = ["not-cached", "known-missing-sentinel", "lookup-raises"],
)
def test_a_cosmos3_hf_cache_miss_falls_back_to_the_base_repo_table(monkeypatch, tmp_path, answer):
    # Load validation is network-free, so before the first download only the shipped values are known.
    monkeypatch.chdir(tmp_path)
    _hub_cache(monkeypatch, answer)
    expected = {
        NANO: True,
        EDGE: False,
        EDGE.upper(): False,  # Hub ids match case-insensitively
        _UNKNOWN_REPO: True,  # the pipeline default
    }
    assert {repo: vc3.checkpoint_enables_safety_checker(repo) for repo in expected} == expected


def test_a_hub_cache_snapshot_decides_the_cosmos3_safety_checker(monkeypatch, tmp_path):
    # The real lookup over a real cache layout (refs/main -> snapshots/<sha>/model_index.json), with both hub-cache
    # roots pointed at it as the suite's conftest pins them. The cached Edge index is flipped on, so a pass means the
    # snapshot was read, not the base-repo table.
    import huggingface_hub.constants as hub_constants

    revision, _entry = _pinned(EDGE)
    hub = tmp_path / "hub"
    repo_dir = hub / "models--nvidia--Cosmos3-Edge"
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(revision, encoding = "utf-8")
    _write_index(
        repo_dir / "snapshots" / revision, {**_EDGE["model_index"], "enable_safety_checker": True}
    )
    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.chdir(tmp_path)
    assert vc3.checkpoint_enables_safety_checker(EDGE) is True


@pytest.mark.parametrize("opted_out", [False, True])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("repo, checker_on", [(NANO, True), (EDGE, False)])
def test_cosmos3_guardrail_refusal_needs_checker_on_package_missing_and_no_opt_out(
    monkeypatch, repo, checker_on, installed, opted_out
):
    # Refuse before eviction and a 35 GB download in the case where from_pretrained would raise ImportError: the
    # checkpoint enables the checker (Nano's index leaves the pipeline default on, Edge's ships it off), the package
    # is missing, and the user has not opted out.
    _hub_cache(monkeypatch, None)
    _guardrail_package(monkeypatch, installed)
    _opt_out(monkeypatch, "0" if opted_out else None)
    refusal = vc3.guardrail_refusal(repo)
    assert (refusal is not None) is (checker_on and not installed and not opted_out)


def test_the_cosmos3_guardrail_refusal_names_the_package_the_install_and_the_opt_out(monkeypatch):
    _hub_cache(monkeypatch, None)
    _guardrail_package(monkeypatch, False)
    _opt_out(monkeypatch, None)
    message = vc3.guardrail_refusal(NANO)
    # Actionable either way: install the package (PyPI cosmos-guardrail), or opt out by name.
    for part in (
        f"'{NANO}'",
        "enable_safety_checker",
        "cosmos_guardrail",
        "pip install cosmos-guardrail",
        f"{vc3.SAFETY_CHECKER_ENV}=0",
    ):
        assert part in message, part
    # An index in hand decides without a lookup: Edge's ships the checker off.
    assert vc3.guardrail_refusal(NANO, index = _EDGE["model_index"]) is None


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_cosmos3_pipeline_kwargs_drop_sound_and_keep_the_checkpoint_guardrail(
    monkeypatch, repo, installed
):
    # Studio renders video only, so Nano's 1.99 GB sound tokenizer never loads (None skips an optional component).
    # Without the opt-out there is no enable_safety_checker kwarg: the checkpoint's model_index.json decides.
    _hub_cache(monkeypatch, None)
    _guardrail_package(monkeypatch, installed)
    _opt_out(monkeypatch, None)
    log = _Log()
    monkeypatch.setattr(vc3, "logger", log)
    assert vc3.pipeline_kwargs(repo) == {"sound_tokenizer": None}
    assert log.lines == []


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("repo, warns", [(NANO, True), (EDGE, False)])
def test_the_cosmos3_opt_out_turns_the_checker_off_and_warns_when_the_checkpoint_enables_it(
    monkeypatch, repo, warns, installed
):
    # The opt-out is the user's explicit choice, so it holds whether or not cosmos_guardrail is installed.
    _hub_cache(monkeypatch, None)
    _guardrail_package(monkeypatch, installed)
    _opt_out(monkeypatch, "0")
    log = _Log()
    monkeypatch.setattr(vc3, "logger", log)
    assert vc3.pipeline_kwargs(repo) == {"sound_tokenizer": None, "enable_safety_checker": False}
    # The log records that a checker the checkpoint asks for was skipped; Edge ships it off, so nothing is skipped.
    warnings = [line for level, line in log.lines if level == "warning"]
    assert len(warnings) == (1 if warns else 0)
    assert all(vc3.SAFETY_CHECKER_ENV in line and repo in line for line in warnings)


def test_cosmos3_untiled_encode_turns_tiling_off_for_the_encode_and_restores_it():
    # Cosmos3's image-to-video conditions every denoising step on the VAE-encoded first-frame clip, so a tiled encode's
    # small shift grows across the clip: Nano i2v at 1280x720x189 against plain diffusers on the same stack measured
    # 24.7 dB mean PSNR with the tiled encode and 34.5 dB untiled. The decode keeps the load's tiling.
    vae = _FakeCosmos3Vae()
    vae.enable_tiling()
    log = _Log()
    assert vc3.install_untiled_encode(types.SimpleNamespace(vae = vae), logger = log) is True
    assert getattr(vae.encode, "_unsloth_untiled_encode", False) is True
    clip = object()
    assert vae.encode(clip, return_dict = False) == ("latents", (clip,), {"return_dict": False})
    assert vae.calls == ["encode untiled"]
    assert vae.use_tiling is True
    assert [level for level, _line in log.lines] == ["info"]


def test_cosmos3_untiled_encode_is_a_plain_call_when_tiling_is_off(monkeypatch):
    # With tiling off the stock encode already runs untiled: the wrapper changes nothing, and an out-of-memory there
    # reaches the caller as it would without it (no tiled retry the load never chose).
    vae = _FakeCosmos3Vae()
    _record_cuda_cache_flushes(monkeypatch, vae.calls)
    log = _Log()
    assert vc3.install_untiled_encode(types.SimpleNamespace(vae = vae), logger = log) is True
    clip = object()
    assert vae.encode(clip) == ("latents", (clip,), {})
    vae.encode_raises = [_TorchOutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")]
    with pytest.raises(_TorchOutOfMemoryError):
        vae.encode(clip)
    assert vae.calls == ["encode untiled", "encode untiled"]
    assert vae.use_tiling is False
    assert [level for level, _line in log.lines] == ["info"]  # the install line only


@pytest.mark.parametrize(
    "error_type, message",
    [
        # torch's class, matched by name whatever the message says.
        (_TorchOutOfMemoryError, "Tried to allocate 2.00 GiB"),
        # A plain RuntimeError, matched by its message.
        (RuntimeError, "CUDA out of memory. Tried to allocate 2.00 GiB"),
    ],
    ids = ["oom-class", "oom-message"],
)
def test_cosmos3_untiled_encode_retries_tiled_once_on_out_of_memory(
    monkeypatch, error_type, message
):
    # The untiled encode of a long clip can need more memory than the tiled one the load planned for: fall back to
    # tiled rather than fail the render, once the failed attempt's tensors are gone and the allocator cache is emptied.
    assert diffusion_batched.is_oom_error(error_type(message))
    vae = _FakeCosmos3Vae()
    vae.enable_tiling()
    _record_cuda_cache_flushes(monkeypatch, vae.calls)
    log = _Log()
    assert vc3.install_untiled_encode(types.SimpleNamespace(vae = vae), logger = log) is True
    clip = object()
    vae.encode_raises = [error_type(message)]
    assert vae.encode(clip) == ("latents", (clip,), {})
    assert vae.calls == ["encode untiled", "empty_cache", "encode tiled"]
    assert vae.use_tiling is True
    warnings = [line for level, line in log.lines if level == "warning"]
    assert len(warnings) == 1 and "out of memory" in warnings[0] and "tiled" in warnings[0]
    # Once: when the tiled retry runs out of memory too, the encode fails as it would without the wrapper.
    vae.calls.clear()
    vae.encode_raises = [error_type(message), error_type(message)]
    with pytest.raises(error_type):
        vae.encode(clip)
    assert vae.calls == ["encode untiled", "empty_cache", "encode tiled"]
    assert vae.use_tiling is True


def test_cosmos3_untiled_encode_reraises_other_errors_with_tiling_restored(monkeypatch):
    # Only memory earns the tiled retry; any other failure is the encode's own and must surface unchanged, with the
    # flag back on for the decode and the next render.
    vae = _FakeCosmos3Vae()
    vae.enable_tiling()
    _record_cuda_cache_flushes(monkeypatch, vae.calls)
    log = _Log()
    assert vc3.install_untiled_encode(types.SimpleNamespace(vae = vae), logger = log) is True
    error = ValueError("expected a 5-D [B, C, T, H, W] clip")
    assert not diffusion_batched.is_oom_error(error)
    vae.encode_raises = [error]
    with pytest.raises(ValueError) as raised:
        vae.encode(object())
    assert raised.value is error
    assert vae.calls == ["encode untiled"]
    assert vae.use_tiling is True
    assert [level for level, _line in log.lines] == ["info"]


def test_cosmos3_untiled_encode_installs_once():
    # The load installs it on every Cosmos3 load with tiling; a second install keeps the first wrapper rather than
    # nesting another around it.
    vae = _FakeCosmos3Vae()
    vae.enable_tiling()
    pipe = types.SimpleNamespace(vae = vae)
    log = _Log()
    assert vc3.install_untiled_encode(pipe, logger = log) is True
    wrapped = vae.encode
    assert vc3.install_untiled_encode(pipe, logger = log) is True
    assert vae.encode is wrapped
    vae.encode(object())
    assert vae.calls == ["encode untiled"]
    assert [level for level, _line in log.lines] == ["info"]


@pytest.mark.parametrize(
    "pipe",
    [
        types.SimpleNamespace(),
        types.SimpleNamespace(vae = None),
        types.SimpleNamespace(vae = types.SimpleNamespace(use_tiling = True)),
        types.SimpleNamespace(vae = types.SimpleNamespace(use_tiling = True, encode = None)),
        types.SimpleNamespace(vae = types.SimpleNamespace(use_tiling = True, encode = "encode")),
    ],
    ids = ["no-vae", "vae-none", "no-encode", "encode-none", "encode-not-callable"],
)
def test_cosmos3_untiled_encode_needs_a_vae_with_a_callable_encode(pipe):
    # Nothing to wrap: the load goes on with the pipeline as built.
    vae = getattr(pipe, "vae", None)
    before = dict(vars(vae)) if vae is not None else None
    log = _Log()
    assert vc3.install_untiled_encode(pipe, logger = log) is False
    assert log.lines == []
    assert (dict(vars(vae)) if vae is not None else None) == before


# ── scheduler and output ─────────────────────────────────────────────────────


def _held_overrides(scheduler, overrides):
    """The rebuilt scheduler's value for each override key; a KeyError means its config dropped the key."""
    return {key: scheduler.config[key] for key, _value in overrides}


def _pipeline_schedule(scheduler, steps, native):
    """The (sigmas, timesteps) Cosmos3OmniPipeline sets for ``steps`` (pipeline_cosmos3_omni.py:1693-1701 at diffusers
    d42ebf9): under ``use_native_flow_schedule`` it hands the scheduler its own linspace sigmas."""
    import numpy as np

    if native:
        sigmas = np.linspace(1.0 - 1.0 / scheduler.config.num_train_timesteps, 0.0, steps + 1)[:-1]
        scheduler.set_timesteps(steps, device = "cpu", sigmas = sigmas)
    else:
        scheduler.set_timesteps(steps, device = "cpu")
    return scheduler.sigmas.clone(), scheduler.timesteps.clone()


def test_cosmos3_edge_overrides_sample_the_card_schedule_not_the_shipped_one():
    pytest.importorskip("diffusers")  # Backend CI has no diffusers; this path runs on it
    import numpy as np
    import torch
    from diffusers import UniPCMultistepScheduler

    fam = detect_video_family(EDGE)
    shipped = UniPCMultistepScheduler.from_config(_EDGE["scheduler_config"])
    pipe = types.SimpleNamespace(scheduler = shipped)
    assert apply_scheduler_overrides(pipe, fam.scheduler_overrides) is True
    # Edge README.md:759-761, the card's diffusers image-to-video example.
    card = UniPCMultistepScheduler.from_config(
        shipped.config, flow_shift = 12.0, use_karras_sigmas = False
    )
    assert dict(pipe.scheduler.config) == dict(card.config)
    # A key UniPC does not define would be dropped from both configs above without a word.
    assert _held_overrides(pipe.scheduler, fam.scheduler_overrides) == dict(fam.scheduler_overrides)
    # Edge's model_index.json sets use_native_flow_schedule, so the pipeline passes its own sigmas; with Karras off
    # the shift is applied to them.
    native = _EDGE["model_index"]["use_native_flow_schedule"]
    ours = _pipeline_schedule(pipe.scheduler, fam.default_steps, native)
    cards = _pipeline_schedule(card, fam.default_steps, native)
    before = _pipeline_schedule(
        UniPCMultistepScheduler.from_config(_EDGE["scheduler_config"]), fam.default_steps, native
    )
    assert torch.equal(ours[0], cards[0]) and torch.equal(ours[1], cards[1])
    # The card's schedule, independently of UniPC: the native linspace at shift 12, s*t / (1 + (s - 1)*t), then 0.
    t = np.linspace(1.0 - 1.0 / shipped.config.num_train_timesteps, 0.0, fam.default_steps + 1)[:-1]
    assert torch.allclose(
        ours[0][:-1].double(), torch.from_numpy(12.0 * t / (1.0 + 11.0 * t)), atol = 1e-6
    )
    assert float(ours[0][-1]) == 0.0
    # Last nonzero sigma: 0.128 on the shipped Karras schedule, 0.387 on the card's.
    assert not torch.equal(ours[0], before[0])


def test_cosmos3_nano_override_matches_the_card_and_keeps_the_shipped_karras_schedule():
    pytest.importorskip("diffusers")  # Backend CI has no diffusers; this path runs on it
    import torch
    from diffusers import UniPCMultistepScheduler

    fam = detect_video_family(NANO)
    shipped = UniPCMultistepScheduler.from_config(_NANO["scheduler_config"])
    pipe = types.SimpleNamespace(scheduler = shipped)
    assert apply_scheduler_overrides(pipe, fam.scheduler_overrides) is True
    # Nano README.md:887, the card's diffusers text-to-video example: flow_shift only.
    card = UniPCMultistepScheduler.from_config(shipped.config, flow_shift = 10.0)
    assert dict(pipe.scheduler.config) == dict(card.config)
    assert _held_overrides(pipe.scheduler, fam.scheduler_overrides) == dict(fam.scheduler_overrides)
    native = bool(_NANO["model_index"].get("use_native_flow_schedule", False))  # absent: off
    ours = _pipeline_schedule(pipe.scheduler, fam.default_steps, native)
    cards = _pipeline_schedule(card, fam.default_steps, native)
    before = _pipeline_schedule(
        UniPCMultistepScheduler.from_config(_NANO["scheduler_config"]), fam.default_steps, native
    )
    assert torch.equal(ours[0], cards[0]) and torch.equal(ours[1], cards[1])
    # The shipped use_karras_sigmas pre-empts the only branch that reads flow_shift (scheduling_unipc_multistep.py:375
    # vs 428-434 at d42ebf9), so the card's shift is inert and Nano samples the shipped Karras schedule. Pinned so a
    # change on either side (the row turning Karras off as the diffusers docs do, or UniPC reading the shift under
    # Karras) is noticed and decided.
    assert torch.equal(ours[0], before[0]) and torch.equal(ours[1], before[1])


def test_cosmos3_output_frames_are_the_unbatched_video_as_returned():
    # Cosmos3OmniPipelineOutput.video is one clip, already unbatched (postprocess_video(...)[0],
    # pipeline_cosmos3_omni.py:1915 at d42ebf9), and both cards export result.video as is (Nano README.md:900, Edge
    # README.md:781). The generic path's output.frames[0] would raise on it; video[0] would be a single frame.
    clip = [object(), object(), object()]
    output = types.SimpleNamespace(video = clip, frames = [["not the clip"]])
    assert vc3.output_frames(output) is clip


def test_the_real_cosmos3_pipeline_output_carries_the_video_studio_reads():
    _diffusers_with_cosmos3()
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipelineOutput

    assert "video" in {field.name for field in dataclasses.fields(Cosmos3OmniPipelineOutput)}
    clip = [object()]
    assert vc3.output_frames(Cosmos3OmniPipelineOutput(video = clip)) is clip


# ── the fake Cosmos3OmniPipeline ─────────────────────────────────────────────


class _FakeCosmos3Scheduler:
    """UniPCMultistepScheduler as far as Studio touches it: ``config``, and ``from_config``, which
    ``apply_scheduler_overrides`` rebuilds through (UniPC has no ``set_shift``)."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls({**dict(config), **kwargs})


class _FakeCosmos3Output:
    """``Cosmos3OmniPipelineOutput(video, sound = None, action = None)``: one clip, already unbatched, and no
    ``frames`` (diffusers pipeline_cosmos3_omni.py:282-298), so code that reads ``output.frames[0]`` fails here."""

    __slots__ = ("video", "sound", "action")

    def __init__(
        self,
        video,
        sound = None,
        action = None,
    ) -> None:
        self.video = video
        self.sound = sound
        self.action = action


def _records_passed_keywords(call):
    """Keep ``call``'s spelled-out signature for ``inspect.signature`` (Studio gates fps, frame_rate and
    guidance_scale_2 on it) and record exactly the keywords a render passed: Nano's card values (35 steps, CFG 6,
    24 fps) equal the pipeline's own defaults, so a defaulted parameter would read the same as a passed one."""

    @functools.wraps(call)
    def wrapper(self, *args, **kwargs):
        self.passed = dict(kwargs)
        return call(self, *args, **kwargs)

    return wrapper


def _shipped_scheduler_config(source) -> dict:
    """The scheduler_config.json ``source`` ships; both repos carry UniPC with Karras sigmas and flow_shift 1.0."""
    for repo in (NANO, EDGE):
        if str(source).strip().lower() == repo.lower():
            return dict(_checkpoint(repo)["scheduler_config"])
    return dict(_checkpoint(NANO)["scheduler_config"])


class _FakeCosmos3Pipeline:
    """``diffusers.Cosmos3OmniPipeline`` as VideoBackend drives it.

    ``from_pretrained`` records its kwargs and returns an instance of this class, as the real classmethod returns a
    ``Cosmos3OmniPipeline``; the family names that class as its own image-to-video class, so a first frame needs no
    pipeline swap. ``__call__`` takes the keywords of the real ``__call__`` (diffusers 0.41.0.dev0 at d42ebf9, the
    Studio pin) and nothing else, so a keyword the real pipeline lacks (``frame_rate``, ``guidance_scale_2``,
    ``last_image``, ``sigmas``) raises TypeError here as it would there. The call follows the real one
    (pipeline_cosmos3_omni.py:884, :1509, :1719-1892): the conditioning clip is VAE-encoded before the loop,
    ``_interrupt`` is reset per call, the loop runs inside ``self.progress_bar`` and skips its remaining steps once
    ``_interrupt`` is set, ``callback_on_step_end(self, i, t, {"latents": ...})`` runs at the end of each step and the
    loop pops ``latents`` from the dict it returns, and the decode runs after the loop, interrupted or not."""

    loads: list = []

    def __init__(self, scheduler_config: dict) -> None:
        self.transformer = _FakeWanDiT()
        self.vae = _FakeCosmos3Vae()
        self.scheduler = _FakeCosmos3Scheduler(scheduler_config)
        self.passed = None
        self.outputs: list = []
        self.steps_computed = 0
        # Test hook: fired inside each computed step, before Studio's callback runs at its end.
        self.on_step = None
        self._interrupt = False

    @property
    def components(self) -> dict:
        return {"transformer": self.transformer, "vae": self.vae, "scheduler": self.scheduler}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cls.loads.append({"source": pretrained_model_name_or_path, **kwargs})
        return cls(_shipped_scheduler_config(pretrained_model_name_or_path))

    def to(self, device):
        return self

    @contextlib.contextmanager
    def progress_bar(self, total = None):
        # DiffusionPipeline.progress_bar: Studio's denoise-phase hook wraps it.
        yield types.SimpleNamespace(update = lambda n = 1: None)

    def maybe_free_model_hooks(self) -> None:
        pass

    @_records_passed_keywords
    def __call__(
        self,
        prompt,
        negative_prompt = None,
        image = None,
        video = None,
        condition_frame_indexes_vision = (0, 1),
        condition_video_keep = "first",
        num_frames = None,
        height = None,
        width = None,
        fps = 24.0,
        num_inference_steps = 35,
        guidance_scale = 6.0,
        enable_sound = False,
        generator = None,
        latents = None,
        sound_latents = None,
        action_latents = None,
        action = None,
        output_type = "pil",
        return_dict = True,
        use_system_prompt = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ("latents",),
        add_resolution_template = True,
        add_duration_template = True,
        enable_safety_check = True,
        mixed_precision_format = None,
        mixed_precision_first_steps = None,
        mixed_precision_last_steps = None,
        mixed_precision_reasoner_policy = None,
    ):
        self._interrupt = False
        steps = int(num_inference_steps)
        # The conditioning clip: the first frame repeated to the clip length, or zeros on text-to-video.
        self.vae.encode(("clip", image))
        latents = object()
        with self.progress_bar(total = steps) as bar:
            for i in range(steps):
                if self._interrupt:
                    continue
                self.steps_computed += 1
                if self.on_step is not None:
                    self.on_step(i)
                if callback_on_step_end is not None:
                    outputs = callback_on_step_end(
                        self, i, 1000.0 * (1 - i / steps), {"latents": latents}
                    )
                    latents = outputs.pop("latents", latents)
                bar.update()
        self.vae.decode(latents)
        # The Wan2.2 VAE decodes 1 + 4 * ((T - 1) // 4) frames, whatever T was asked for.
        count = 1 + 4 * ((int(num_frames or 189) - 1) // 4)
        output = _FakeCosmos3Output(video = [object() for _ in range(count)])
        self.maybe_free_model_hooks()
        self.outputs.append(output)
        return output


@pytest.fixture
def cosmos3_runtime(fake_runtime, monkeypatch):
    """``fake_runtime`` with ``diffusers.Cosmos3OmniPipeline`` as ``_FakeCosmos3Pipeline``, the guardrail opted out
    (tests about it set their own state), and the MP4 encoder recording what it is handed."""
    monkeypatch.setattr(
        sys.modules["diffusers"], "Cosmos3OmniPipeline", _FakeCosmos3Pipeline, raising = False
    )
    _FakeCosmos3Pipeline.loads = []
    _set_guardrail(monkeypatch, "opted out")
    encoded: list = []

    def _encode(frames, fps, audio, pipe):
        encoded.append({"frames": frames, "fps": fps, "audio": audio, "pipe": pipe})
        return b"MP4"

    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(_encode))
    yield encoded


def _loaded(repo: str) -> tuple[VideoBackend, dict]:
    backend = VideoBackend()
    status = backend.load_pipeline(repo, model_kind = "pipeline")
    return backend, status


# ── load path ────────────────────────────────────────────────────────────────


def test_the_real_cosmos3_pipeline_init_takes_the_kwargs_studio_loads_with(monkeypatch):
    _diffusers_with_cosmos3()
    from diffusers import Cosmos3OmniPipeline

    _hub_cache(monkeypatch, None)
    _opt_out(monkeypatch, "0")
    monkeypatch.setattr(vc3, "logger", _Log())
    passed = vc3.pipeline_kwargs(NANO)
    assert set(passed) == {"sound_tokenizer", "enable_safety_checker"}
    # from_pretrained drops a kwarg __init__ does not take with only a warning (pipeline_utils.py:981 at d42ebf9):
    # a rename would silently load the sound tokenizer again and build the guardrail despite the opt-out.
    assert set(passed) <= set(inspect.signature(Cosmos3OmniPipeline.__init__).parameters)
    assert "sound_tokenizer" in Cosmos3OmniPipeline._optional_components  # so None is a valid value


@pytest.mark.parametrize(
    "repo,guardrail,safety",
    [
        # The explicit opt-out is the one case where Studio overrides the checkpoint's choice.
        (NANO, "opted out", {"enable_safety_checker": False}),
        (EDGE, "opted out", {"enable_safety_checker": False}),
        # Otherwise the checkpoint decides: Nano builds the guardrail from the installed package, and Edge's index
        # turns it off.
        (NANO, "installed", {}),
        (EDGE, "absent", {}),
    ],
)
def test_cosmos3_load_skips_the_sound_tokenizer_and_overrides_the_guardrail_only_on_opt_out(
    cosmos3_runtime, monkeypatch, repo, guardrail, safety
):
    # Studio renders video only: sound_tokenizer=None skips building Nano's 2 GB audio decoder (Edge has none).
    _set_guardrail(monkeypatch, guardrail)
    _loaded(repo)
    (load,) = _FakeCosmos3Pipeline.loads
    assert load["source"] == repo
    assert "sound_tokenizer" in load and load["sound_tokenizer"] is None
    assert {k: v for k, v in load.items() if k == "enable_safety_checker"} == safety


@pytest.mark.parametrize(
    "repo,overrides",
    [
        # Nano's card rebuilds UniPC with flow_shift=10.0 only (Nano README.md:887).
        (NANO, {"flow_shift": 10.0}),
        # Edge's card also turns Karras sigmas off, which is what makes the shift act (Edge README.md:759-761).
        (EDGE, {"flow_shift": 12.0, "use_karras_sigmas": False}),
    ],
)
def test_cosmos3_load_rebuilds_the_scheduler_with_the_model_card_overrides(
    cosmos3_runtime, repo, overrides
):
    shipped = _checkpoint(repo)["scheduler_config"]
    # Both repos ship flow_shift 1.0 with Karras sigmas, which neither card samples with as shipped.
    assert (shipped["flow_shift"], shipped["use_karras_sigmas"]) == (1.0, True)
    backend, _status = _loaded(repo)
    scheduler = backend._state.pipe.scheduler
    assert type(scheduler) is _FakeCosmos3Scheduler
    assert scheduler.config == {**shipped, **overrides}


@pytest.mark.parametrize(
    "repo,steps,frames,canvas",
    [
        # Nano card: 1280x720, 189 frames, 35 steps, CFG 6, 24 fps (Nano README.md:889-898, :900).
        (NANO, 35, 189, [1280, 720]),
        # Edge card (image-to-video): 832x480, 121 frames, 20 steps, CFG 6, 24 fps (Edge README.md:763-776).
        (EDGE, 20, 121, [832, 480]),
    ],
)
def test_cosmos3_status_reports_the_card_defaults_and_a_first_frame_only(
    cosmos3_runtime, repo, steps, frames, canvas
):
    _backend, status = _loaded(repo)
    assert status["loaded"] is True and status["family"] == _FAMILY[repo]
    assert status["model_kind"] == "pipeline"
    # Video only: the sound tokenizer is not loaded, so no clip carries audio.
    assert status["has_audio"] is False
    # A first frame through the pipeline's own image=; no last frame, no references.
    assert status["supports_first_frame"] is True
    assert status["supports_last_frame"] is False
    assert status["supports_keyframes"] is False and status["supports_references"] is False
    defaults = status["defaults"]
    assert (defaults["steps"], defaults["guidance"]) == (steps, 6.0)
    assert (defaults["num_frames"], defaults["fps"]) == (frames, 24)
    assert (defaults["frame_step"], defaults["frame_offset"]) == (4, 1)
    assert defaults["resolution_presets"][0] == canvas
    assert defaults["canvas_from_presets"] is True


@pytest.mark.parametrize(
    "repo,wrapped",
    [(NANO, True), (EDGE, True), ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", False)],
    ids = ["nano", "edge", "wan2.2-ti2v-5b"],
)
def test_cosmos3_load_with_vae_tiling_wraps_the_encode_and_a_wan_load_does_not(
    cosmos3_runtime, monkeypatch, repo, wrapped
):
    # The untiled encode is Cosmos3's (measured on its image-to-video; video.py installs it for the Cosmos3 rows only),
    # so a Wan load keeps the stock encode. Wan2.2-TI2V-5B has the same VAE (AutoencoderKLWan, z_dim 48: both Cosmos3
    # repos ship its vae_config); its fake gets an encode here, so a hook that wrapped every family would show.
    monkeypatch.setattr(
        _FakeWanVae, "encode", lambda self, *args, **kwargs: ("latents", args), raising = False
    )
    backend, status = _loaded(repo)
    assert status["vae_tiling"] is True
    encode = backend._state.pipe.vae.encode
    assert callable(encode)
    assert getattr(encode, "_unsloth_untiled_encode", False) is wrapped


# ── generate path ────────────────────────────────────────────────────────────

# What VideoBackend.generate hands Cosmos3OmniPipeline.__call__. The unconditional ones (prompt, size, frames,
# steps, generator, the rows' cfg_kwarg guidance_scale, and image= with a first frame) would raise TypeError if
# renamed; the signature-gated ones (negative_prompt, fps, callback_on_step_end, and the rows' call_kwargs) would
# silently stop reaching the model. add_*_template are what the Edge card's call turns off (Edge README.md:774-775).
_STUDIO_CALL_KWARGS = (
    "prompt",
    "negative_prompt",
    "image",
    "num_frames",
    "height",
    "width",
    "fps",
    "num_inference_steps",
    "guidance_scale",
    "generator",
    "callback_on_step_end",
    "add_resolution_template",
    "add_duration_template",
)


@pytest.mark.parametrize("name", _STUDIO_CALL_KWARGS)
def test_the_real_cosmos3_pipeline_call_takes_every_kwarg_studio_relies_on(name):
    _diffusers_with_cosmos3()
    from diffusers import Cosmos3OmniPipeline
    assert name in inspect.signature(Cosmos3OmniPipeline.__call__).parameters


def test_the_real_cosmos3_pipeline_call_renders_no_sound_unless_asked():
    _diffusers_with_cosmos3()
    from diffusers import Cosmos3OmniPipeline

    # Studio loads without the sound tokenizer (pipeline_kwargs) and never passes enable_sound, so it relies on this
    # default: with sound on and no sound tokenizer the call raises ValueError (pipeline_cosmos3_omni.py:1028-1030 at
    # d42ebf9).
    assert (
        inspect.signature(Cosmos3OmniPipeline.__call__).parameters["enable_sound"].default is False
    )


def test_the_fake_cosmos3_pipeline_takes_exactly_the_real_call_keywords():
    _diffusers_with_cosmos3()
    from diffusers import Cosmos3OmniPipeline

    # The render tests below lean on the fake refusing what the real __call__ refuses (frame_rate, guidance_scale_2,
    # last_image) and taking what it takes, so a diffusers change to the signature has to fail here first.
    real = inspect.signature(Cosmos3OmniPipeline.__call__).parameters
    fake = inspect.signature(_FakeCosmos3Pipeline.__call__).parameters
    assert [(name, p.kind) for name, p in fake.items()] == [
        (name, p.kind) for name, p in real.items()
    ]


# Text-to-video on Nano, and image-to-video on both. Edge's card documents image-to-video only (Edge README.md:712),
# so no test here depends on Studio offering Edge for text-to-video.
_RENDERS = [
    pytest.param(NANO, None, id = "nano-t2v"),
    pytest.param(NANO, (1920, 1080), id = "nano-i2v"),
    pytest.param(EDGE, (1920, 1080), id = "edge-i2v"),
]


def _first_frame(size) -> dict:
    """generate() keywords for a first frame of ``size`` (width, height), or none for a text-only render."""
    return {"first_frame": _data_url(*size)} if size else {}


@pytest.mark.parametrize("repo,first_frame", _RENDERS)
def test_cosmos3_generate_passes_fps_as_a_float_and_only_keywords_the_pipeline_takes(
    cosmos3_runtime, repo, first_frame
):
    """Both cards set the frame rate per request (Nano README.md:112, Edge README.md:101-102: 12-30 fps); the
    pipeline's fps fills the duration template and modulates the temporal RoPE, so the requested rate has to reach it
    rather than only the export. Its __call__ has no frame_rate (LTX-2's spelling) and no guidance_scale_2."""
    backend, _status = _loaded(repo)
    result = backend.generate(
        prompt = "a robot arm stacks blocks",
        negative_prompt = "blurry",
        steps = 2,
        fps = 16,
        guidance_2 = 3.5,
        **_first_frame(first_frame),
    )
    passed = backend._state.pipe.passed
    # The prompt-template flags are left out here: each row's are pinned on their own below.
    assert set(passed) - {"add_resolution_template", "add_duration_template"} == {
        "prompt",
        "negative_prompt",
        "num_inference_steps",
        "guidance_scale",
        "width",
        "height",
        "num_frames",
        "fps",
        "generator",
        "callback_on_step_end",
    } | ({"image"} if first_frame else set())
    assert passed["fps"] == 16.0 and type(passed["fps"]) is float
    assert passed["negative_prompt"] == "blurry"
    assert result["fps"] == 16 and cosmos3_runtime[-1]["fps"] == 16


@pytest.mark.parametrize(
    "repo,first_frame,steps,frames,size,conditioning",
    [
        pytest.param(NANO, None, 35, 189, (1280, 720), "t2va", id = "nano-t2v"),
        # A 16:9 first frame lands on the default canvas, the nearest-aspect preset.
        pytest.param(NANO, (1920, 1080), 35, 189, (1280, 720), "i2va", id = "nano-i2v"),
        pytest.param(EDGE, (1920, 1080), 20, 121, (832, 480), "i2va", id = "edge-i2v"),
    ],
)
def test_cosmos3_generate_fills_the_family_defaults_the_request_leaves_out(
    cosmos3_runtime, repo, first_frame, steps, frames, size, conditioning
):
    backend, _status = _loaded(repo)
    seen: list = []
    pipe = backend._state.pipe
    pipe.on_step = lambda i: seen.append(backend.generate_progress())
    result = backend.generate(prompt = "a robot arm stacks blocks", **_first_frame(first_frame))
    passed = pipe.passed
    assert (passed["num_inference_steps"], passed["guidance_scale"]) == (steps, 6.0)
    assert (passed["width"], passed["height"], passed["num_frames"]) == (*size, frames)
    assert passed["fps"] == 24.0
    assert (result["steps"], result["guidance"], result["num_frames"]) == (steps, 6.0, frames)
    assert (result["width"], result["height"], result["fps"]) == (*size, 24)
    assert result["conditioning"] == conditioning and result["has_audio"] is False
    # The bar counts the family's steps, in the denoise phase from the moment the loop opens progress_bar.
    assert [(gen["phase"], gen["total_steps"]) for gen in seen] == [("denoise", steps)] * steps


@pytest.mark.parametrize(
    "repo,first_frame,requested,frames",
    [
        # 4k+1 lattice (the Wan2.2 VAE compresses time by 4), floored.
        (NANO, None, 120, 117),
        # Nano's window ends at 397, the last lattice count inside the card's 5-400 (Nano README.md:113).
        (NANO, None, 400, 397),
        # Edge's window is 53-149, the lattice counts inside the card's 50-150 (Edge README.md:102).
        (EDGE, (1920, 1080), 30, 53),
        (EDGE, (1920, 1080), 100, 97),
        (EDGE, (1920, 1080), 200, 149),
    ],
)
def test_cosmos3_generate_snaps_frames_and_size_onto_the_family_lattice(
    cosmos3_runtime, repo, first_frame, requested, frames
):
    backend, _status = _loaded(repo)
    # An explicit size wins over the first frame's canvas, so it snaps the same way with or without one.
    result = backend.generate(
        prompt = "p",
        steps = 1,
        num_frames = requested,
        width = 1000,
        height = 700,
        **_first_frame(first_frame),
    )
    passed = backend._state.pipe.passed
    assert passed["num_frames"] == frames and result["num_frames"] == frames
    # Sizes floor to multiples of 16, the VAE's spatial factor (vae/config.json scale_factor_spatial).
    assert _checkpoint(repo)["vae_config"]["scale_factor_spatial"] == 16
    assert (passed["width"], passed["height"]) == (992, 688)


@pytest.mark.parametrize("repo,first_frame", _RENDERS)
def test_cosmos3_generate_exports_the_clip_from_output_video(cosmos3_runtime, repo, first_frame):
    """Cosmos3OmniPipelineOutput carries ``video`` (already unbatched), ``sound`` and ``action``, never ``frames``; the
    fake has no ``frames`` attribute, so the generic ``output.frames[0]`` would raise here."""
    backend, _status = _loaded(repo)
    result = backend.generate(prompt = "p", steps = 2, num_frames = 61, **_first_frame(first_frame))
    output = backend._state.pipe.outputs[-1]
    assert not hasattr(output, "frames")
    (export,) = cosmos3_runtime
    assert export["frames"] is output.video
    assert export["audio"] is None and export["pipe"] is None
    assert result["num_frames"] == len(output.video) == 61
    assert result["mp4_bytes"] == b"MP4"


@pytest.mark.parametrize(
    "repo,image,canvas",
    [
        (NANO, (1920, 1080), (1280, 720)),
        (NANO, (1080, 1920), (720, 1280)),
        (EDGE, (1920, 1080), (832, 480)),
        (EDGE, (1080, 1920), (480, 832)),
    ],
)
def test_cosmos3_first_frame_reaches_image_unfitted_on_a_preset_canvas(
    cosmos3_runtime, monkeypatch, repo, image, canvas
):
    """The pipeline resizes the image to cover the canvas and centre-crops it itself (pipeline_cosmos3_omni.py:85-99),
    so Studio hands over the decoded image as given; pre-fitting would resample it twice. The canvas is the
    nearest-aspect preset."""
    from core.inference import diffusion as diffusion_mod

    decoded: list = []
    real_decode = diffusion_mod.decode_b64_image

    def _decode(*args, **kwargs):
        decoded.append(real_decode(*args, **kwargs))
        return decoded[-1]

    monkeypatch.setattr(diffusion_mod, "decode_b64_image", _decode)
    backend, _status = _loaded(repo)
    pipe = backend._state.pipe
    result = backend.generate(prompt = "p", steps = 2, first_frame = _data_url(*image))
    passed = pipe.passed
    (first,) = decoded
    assert passed["image"] is first and first.size == image
    assert (passed["width"], passed["height"]) == canvas
    assert (result["width"], result["height"]) == canvas
    assert result["conditioning"] == "i2va"


def test_cosmos3_nano_text_only_render_after_a_first_frame_render_passes_no_image(cosmos3_runtime):
    # One pipeline object serves both modes (no image-to-video swap), so nothing of an image render may carry over.
    backend, _status = _loaded(NANO)
    pipe = backend._state.pipe
    backend.generate(prompt = "p", steps = 2, first_frame = _data_url(1920, 1080))
    assert "image" in pipe.passed
    result = backend.generate(prompt = "p", steps = 2)
    assert "image" not in pipe.passed and result["conditioning"] == "t2va"


@pytest.mark.parametrize("repo,first_frame", _RENDERS)
def test_cosmos3_render_encodes_the_conditioning_clip_untiled_and_restores_tiling_for_the_decode(
    cosmos3_runtime, repo, first_frame
):
    # Before the denoise the pipeline VAE-encodes its conditioning clip (the first frame repeated to the clip length,
    # zeros on text-to-video; pipeline_cosmos3_omni.py:866-884) and every step attends to it, so the load's untiled
    # encode runs; the decode keeps the load's tiling (on this CPU fake, tiled; on a resident CUDA load the shared
    # install_untiled_decode decides).
    backend, status = _loaded(repo)
    vae = backend._state.pipe.vae
    assert status["vae_tiling"] is True and vae.calls == []
    backend.generate(prompt = "p", steps = 2, **_first_frame(first_frame))
    assert vae.calls == ["encode untiled", "decode tiled"]
    assert vae.use_tiling is True


def test_cosmos3_edge_generate_turns_the_prompt_templates_off_like_its_card(cosmos3_runtime):
    """Edge's diffusers example passes add_resolution_template=False and add_duration_template=False (Edge
    README.md:774-775), as its vLLM-Omni example sets use_resolution_template / use_duration_template False
    (README.md:503-504). Left at the pipeline default (True), the prompt and negative prompt gain resolution and
    duration sentences the card does not use."""
    backend, _status = _loaded(EDGE)
    backend.generate(prompt = "p", steps = 2, first_frame = _data_url(1920, 1080))
    passed = backend._state.pipe.passed
    assert (passed.get("add_resolution_template"), passed.get("add_duration_template")) == (
        False,
        False,
    )


@pytest.mark.parametrize("first_frame", [None, (1920, 1080)], ids = ["t2v", "i2v"])
def test_cosmos3_nano_generate_leaves_the_prompt_templates_to_the_pipeline_like_its_card(
    cosmos3_runtime, first_frame
):
    # Nano's diffusers example passes neither flag (Nano README.md:889-898), so the pipeline's defaults (True) add the
    # resolution and duration sentences; Studio runs that pipeline and follows that example. (Its vLLM-Omni and SGLang
    # examples turn both off, README.md:368-369 and :952.)
    backend, _status = _loaded(NANO)
    backend.generate(prompt = "p", steps = 2, **_first_frame(first_frame))
    passed = backend._state.pipe.passed
    assert "add_resolution_template" not in passed and "add_duration_template" not in passed


@pytest.mark.parametrize(
    "repo,first_frame",
    [
        pytest.param(EDGE, (1920, 1080), id = "edge-i2v"),
        # A text-only render too: call_kwargs go with every request, unlike i2v_call_kwargs, which only a first frame
        # brings. Nano's row has none of its own, so it carries Edge's here.
        pytest.param(NANO, None, id = "nano-t2v"),
    ],
)
def test_cosmos3_call_kwargs_skip_a_key_the_pipeline_call_does_not_take(
    cosmos3_runtime, repo, first_frame
):
    # A row's call_kwargs reach __call__ only through its signature. vLLM-Omni spells Edge's flags
    # use_resolution_template / use_duration_template (Edge README.md:503-504); diffusers' __call__ has neither, and the
    # fake, like the real one, raises TypeError on a keyword it does not take.
    backend, _status = _loaded(repo)
    fam = backend._state.family
    vllm_spelling = (("use_resolution_template", False), ("use_duration_template", False))
    backend._state = dataclasses.replace(
        backend._state,
        family = dataclasses.replace(
            fam, call_kwargs = tuple(detect_video_family(EDGE).call_kwargs) + vllm_spelling
        ),
    )
    backend.generate(prompt = "p", steps = 2, **_first_frame(first_frame))
    passed = backend._state.pipe.passed
    assert ("image" in passed) is (first_frame is not None)
    assert not {"use_resolution_template", "use_duration_template"} & set(passed)
    assert (passed["add_resolution_template"], passed["add_duration_template"]) == (False, False)


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_cosmos3_generate_refuses_a_last_frame_before_the_render(cosmos3_runtime, repo):
    # Cosmos3OmniPipeline.__call__ takes one conditioning image (image=) and no last_image, so a first-and-last-frame
    # request is refused before the pipeline is called rather than failing inside it: by begin_generate, the route's
    # entry, before it reserves the job, and by a direct render.
    backend, _status = _loaded(repo)
    with pytest.raises(ValueError, match = "no last frame"):
        backend.begin_generate(
            prompt = "p",
            steps = 2,
            first_frame = _data_url(1920, 1080),
            last_frame = _data_url(1920, 1080),
        )
    with pytest.raises(ValueError, match = "no last frame"):
        backend.generate(
            prompt = "p",
            steps = 2,
            first_frame = _data_url(1920, 1080),
            last_frame = _data_url(1920, 1080),
        )
    assert backend._state.pipe.passed is None
    # The refusal must not leave the busy flag set, or every later render is turned away as busy.
    assert backend.generate_progress()["active"] is False


def test_cosmos3_cancel_interrupts_the_denoise_through_the_step_callback(cosmos3_runtime):
    # The cancel lands during the first step; Studio's callback sets _interrupt at that step's end, the pipeline skips
    # the remaining steps, and generate() discards the clip instead of exporting it.
    backend, _status = _loaded(NANO)
    pipe = backend._state.pipe
    pipe.on_step = lambda i: backend.cancel_generate() if i == 0 else None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "p", steps = 4)
    assert pipe.steps_computed == 1
    assert cosmos3_runtime == []
    assert backend.generate_progress()["active"] is False
    # The next render starts clean: the pipeline resets _interrupt itself.
    pipe.on_step = None
    backend.generate(prompt = "p", steps = 3)
    assert pipe.steps_computed == 4


# ── download plan and local folders ──────────────────────────────────────────


def _hub_listing(repo: str) -> types.SimpleNamespace:
    """``model_info`` as ``_base_download_files`` reads it: siblings with ``rfilename`` and ``size``."""
    return types.SimpleNamespace(
        siblings = [
            types.SimpleNamespace(rfilename = name, size = size)
            for name, size in _checkpoint(repo)["listing"]
        ]
    )


def _planned_files(repo: str) -> dict:
    """{path: bytes} the pipeline load would pre-download from ``repo``."""
    return dict(VideoBackend._base_download_files(_hub_listing(repo), "pipeline"))


def _write_stub_tree(root: Path, repo: str, names) -> Path:
    """``names`` (paths from ``repo``'s listing) under ``root`` as stand-ins: the fixture's own model_index.json, root
    config.json, transformer and scheduler configs; a shard index naming its folder's shards; ``{}`` for any other
    JSON; one byte for everything else. The completeness checks read presence, JSON and shard indexes, never weights."""
    entry = _checkpoint(repo)
    real = {
        "model_index.json": entry["model_index"],
        "config.json": entry["config"],
        "transformer/config.json": entry["transformer_config"],
        "scheduler/scheduler_config.json": entry["scheduler_config"],
    }
    names = sorted(names)
    for name in names:
        path = root / name
        path.parent.mkdir(parents = True, exist_ok = True)
        folder = name.rpartition("/")[0]
        if name in real:
            payload = real[name]
        elif name.endswith(".index.json"):
            shards = [
                n.rpartition("/")[2]
                for n in names
                if n.rpartition("/")[0] == folder and "-of-" in n and n.endswith(".safetensors")
            ]
            payload = {"weight_map": {f"w{i}": shard for i, shard in enumerate(shards)}}
        elif name.endswith(".json"):
            payload = {}
        else:
            path.write_bytes(b"\0")
            continue
        path.write_text(json.dumps(payload), encoding = "utf-8")
    return root


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_cosmos3_download_plan_stages_every_component_the_pipeline_reads(repo):
    """Both repos ship a modular_model_index.json beside model_index.json, which the planner used to read as
    MiniMax-H3's layout, staging scheduler/, transformer/ and vae/ and dropping text_tokenizer/ (the pipeline's only
    text component). Every file of those folders in the listing is planned, shards and shard index included."""
    listing = [name for name, _size in _checkpoint(repo)["listing"]]
    assert "modular_model_index.json" in listing
    planned = _planned_files(repo)
    assert "model_index.json" in planned
    for folder in ("scheduler/", "text_tokenizer/", "transformer/", "vae/"):
        wanted = {name for name in listing if name.startswith(folder)}
        assert wanted and wanted <= set(planned), folder


@pytest.mark.parametrize("repo,gigabytes", [(NANO, 3.14), (EDGE, 0.98)])
def test_cosmos3_download_plan_also_stages_sound_tokenizer_and_vision_encoder_weights(
    repo, gigabytes
):
    """What the plan does with the two folders the video load never opens. The load passes sound_tokenizer=None, and
    Cosmos3OmniPipeline takes no vision_encoder argument (Nano's index lists one, Edge's lists none), yet every file of
    both folders is planned: 3.14 GB for Nano (sound tokenizer 1.99, vision encoder 1.15) and 0.98 GB for Edge (its
    vision encoder). Pinned so a trim is a deliberate change, made together with the cache-completeness check below,
    which counts every model_index component."""
    folders = ("sound_tokenizer/", "vision_encoder/")
    listing = {
        name: size for name, size in _checkpoint(repo)["listing"] if name.startswith(folders)
    }
    staged = {name: size for name, size in _planned_files(repo).items() if name.startswith(folders)}
    assert staged == listing
    assert round(sum(staged.values()) / 1e9, 2) == gigabytes


@pytest.mark.parametrize("repo,components", [(NANO, 6), (EDGE, 4)])
def test_a_cosmos3_snapshot_of_exactly_the_planned_files_scores_complete(
    monkeypatch, tmp_path, repo, components
):
    """The cache check scores a cached snapshot where Studio picks between a repo and its unsloth/ mirror (video.py
    ``_video_fetch_repo``, ``_video_index_repo``, ``_video_offline_base_choice``). Cosmos3 has no mirror yet
    (test_video_unsloth_mirrors.py ``NO_UNSLOTH_REPO``), so nothing reads the score for it today; pinned so that a
    mirror added later does not read a finished download as partial. Nano's index has six non-null components
    (scheduler, sound_tokenizer, text_tokenizer, transformer, vae, vision_encoder); Edge's has four (its
    sound_tokenizer and safety_checker are null)."""
    revision, _entry = _pinned(repo)
    folder = tmp_path / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True)
    (folder / "refs" / "main").write_text(revision, encoding = "utf-8")
    snapshot = folder / "snapshots" / revision
    planned = _planned_files(repo)
    _write_stub_tree(snapshot, repo, planned)
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(tmp_path))
    assert video_mod._video_cached_completeness_detail(repo) == (components, True)


def _local_folders(repo: str) -> list:
    """A folder named after the repo, a renamed one, and the repo's HF-cache snapshot at the pinned revision, each with
    whether its path alone names the family (the renamed folder can resolve only through its configs)."""
    revision, _entry = _pinned(repo)
    name = repo.partition("/")[2]
    cached = f"models--{repo.replace('/', '--')}/snapshots/{revision}"
    return [
        pytest.param(name, True, id = "named"),
        pytest.param("renamed-video-model", False, id = "renamed"),
        pytest.param(cached, True, id = "hf-cache-snapshot"),
    ]


def _local_folder(tmp_path_factory, folder: str) -> Path:
    """``folder`` under a parent with no family token. pytest names ``tmp_path`` after the test, and these tests' names
    hold "cosmos3_edge" / "cosmos3_nano", which name detection reads off any path segment: under ``tmp_path`` the
    renamed folder resolved by its parent's name and never reached its configs."""
    return tmp_path_factory.mktemp("models") / folder


@pytest.mark.parametrize("folder,named", _local_folders(EDGE))
def test_a_local_cosmos3_edge_folder_validates_by_name_or_by_its_configs(
    cosmos3_runtime, monkeypatch, tmp_path_factory, folder, named
):
    # The files a pipeline load stages; a renamed folder resolves through its model_index.json class, root model_type
    # (cosmos3_edge) and transformer hidden_size (2048). No guardrail package either: the folder's own model_index.json
    # turns the checker off, and only the base repo id is in Studio's table of shipped values.
    _set_guardrail(monkeypatch, "absent")
    root = _write_stub_tree(_local_folder(tmp_path_factory, folder), EDGE, _planned_files(EDGE))
    assert (detect_video_family(str(root)) is not None) is named
    assert local_pipeline_components_are_complete(root, "model_index.json")
    assert VideoBackend().validate_load_request(str(root)).name == "cosmos3-edge"


@pytest.mark.parametrize("folder,named", _local_folders(NANO))
def test_a_local_cosmos3_nano_folder_validates_by_name_or_by_its_configs(
    cosmos3_runtime, tmp_path_factory, folder, named
):
    """A local copy of the official repo (``hf download nvidia/Cosmos3-Nano --local-dir ...``), renamed or not, or its
    cache snapshot. validate_load_request judges a local pipeline path with local_pipeline_components_are_complete,
    which reads Nano's sound_tokenizer/ as the model it is
    (test_the_local_completeness_check_reads_the_cosmos3_audio_tokenizer_as_a_model)."""
    root = _write_stub_tree(_local_folder(tmp_path_factory, folder), NANO, _planned_files(NANO))
    assert (detect_video_family(str(root)) is not None) is named
    assert VideoBackend().validate_load_request(str(root)).name == "cosmos3-nano"


def test_the_local_completeness_check_reads_the_cosmos3_audio_tokenizer_as_a_model(tmp_path):
    """Nano's model_index.json lists ``sound_tokenizer: [diffusers, Cosmos3AVAEAudioTokenizer]``, an audio VAE with a
    config.json and weights. The check's "tokenizer" rule (diffusion_families.py ``_LOCAL_PIPELINE_METADATA_CONFIGS``)
    applies to transformers-library tokenizers only, so it no longer asks that folder for a tokenizer_config.json. The
    same check sets the artifact kind of local and cached folders in the model catalog
    (hub/services/models/common.py ``_diffusers_pipeline_artifact_kind``)."""
    root = _write_stub_tree(tmp_path / "Cosmos3-Nano", NANO, _planned_files(NANO))
    # Every other component of the folder passes as it is.
    assert local_pipeline_components_are_complete(
        root, "model_index.json", excluded_components = ("sound_tokenizer",)
    )
    assert local_pipeline_components_are_complete(
        root, "model_index.json"
    ), "sound_tokenizer/ (Cosmos3AVAEAudioTokenizer) judged as a text tokenizer"


# ── validation refusals ──────────────────────────────────────────────────────


@pytest.mark.parametrize("guardrail", ["opted out", "installed"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_cosmos3_base_repos_validate_as_pipelines_once_the_guardrail_is_settled(
    cosmos3_runtime, monkeypatch, repo, guardrail
):
    _set_guardrail(monkeypatch, guardrail)
    fam = VideoBackend().validate_load_request(repo, model_kind = "pipeline")
    assert fam.name == _FAMILY[repo]
    assert fam.pipeline_class == "Cosmos3OmniPipeline"


def test_cosmos3_nano_without_the_guardrail_is_refused_before_the_load_starts(
    cosmos3_runtime, monkeypatch
):
    """Nano's model_index.json leaves enable_safety_checker unset, so Cosmos3OmniPipeline's __init__ default (True)
    builds NVIDIA's guardrail, and without cosmos_guardrail that raises ImportError (pipeline_cosmos3_omni.py:50-58,
    :437-441). Studio's environment does not ship the package, so the refusal has to come at validation, before ~35 GB
    are pulled and the resident model is evicted, and it has to name both ways out."""
    _set_guardrail(monkeypatch, "absent")
    started: list = []
    monkeypatch.setattr(VideoBackend, "_run_load", lambda self, **kwargs: started.append(kwargs))
    backend = VideoBackend()
    with pytest.raises(ValueError) as refused:
        backend.begin_load(NANO, model_kind = "pipeline")
    message = str(refused.value)
    # pip takes the package by either spelling.
    assert "cosmos_guardrail" in message and re.search(r"pip install cosmos[-_]guardrail", message)
    assert f"{_OPT_OUT_ENV}=0" in message
    assert started == [] and backend.loading_repo_ids() == ()
    # A direct load_pipeline call validates before anything else too, so nothing reaches from_pretrained.
    with pytest.raises(ValueError, match = "cosmos_guardrail"):
        backend.load_pipeline(NANO, model_kind = "pipeline")
    assert _FakeCosmos3Pipeline.loads == []


def test_cosmos3_edge_validates_without_the_guardrail_since_its_index_turns_it_off(
    cosmos3_runtime, monkeypatch
):
    # Edge's model_index.json ships "enable_safety_checker": false, so its pipeline never builds the guardrail and
    # there is nothing to refuse. With nothing cached, Studio reads that value from its table of what each base repo
    # ships (video_cosmos3._BASE_SAFETY_CHECKER).
    assert _checkpoint(EDGE)["model_index"]["enable_safety_checker"] is False
    assert "enable_safety_checker" not in _checkpoint(NANO)["model_index"]
    _set_guardrail(monkeypatch, "absent")
    assert VideoBackend().validate_load_request(EDGE, model_kind = "pipeline").name == "cosmos3-edge"


@pytest.mark.parametrize("guardrail", ["opted out", "installed"])
def test_an_untrusted_cosmos3_repack_is_refused_as_untrusted(
    cosmos3_runtime, monkeypatch, guardrail
):
    # Full pipelines load only from unsloth/*, the families' base repos and local paths; the name token makes this
    # repo a cosmos3-nano pick, not a trusted one.
    _set_guardrail(monkeypatch, guardrail)
    with pytest.raises(ValueError, match = "limited to"):
        VideoBackend().validate_load_request(_UNKNOWN_REPO, model_kind = "pipeline")


@pytest.mark.parametrize(
    "repo,filename,kind",
    [
        ("someone/Cosmos3-Nano-GGUF", "cosmos3-nano-Q4_K_M.gguf", None),
        ("someone/Cosmos3-Edge-GGUF", "cosmos3-edge-Q8_0.gguf", None),
        (NANO, "cosmos3_nano_bf16.safetensors", "single_file"),
    ],
    ids = ["nano-gguf", "edge-gguf", "nano-single-file"],
)
def test_cosmos3_gguf_and_single_file_picks_are_refused_at_validation(
    cosmos3_runtime, repo, filename, kind
):
    """A gguf / single_file load builds the denoiser with ``transformer_class.from_single_file`` (video.py's
    ``_load_expert``), which Cosmos3OmniTransformer does not have
    (test_diffusers_cosmos3_transformer_has_no_single_file_loader). Accepted here, the pick fails with AttributeError
    deep in the load, after the resident model was evicted and the checkpoint and the base companions were pulled."""
    with pytest.raises(ValueError, match = "loads only as its full Diffusers pipeline"):
        VideoBackend().validate_load_request(repo, gguf_filename = filename, model_kind = kind)


def test_diffusers_cosmos3_transformer_has_no_single_file_loader():
    diffusers = _diffusers_with_cosmos3()

    # Why a Cosmos3 gguf / single_file pick cannot load: the load builds the denoiser through
    # transformer_class.from_single_file, and Cosmos3OmniTransformer is no FromOriginalModelMixin.
    assert not hasattr(diffusers.Cosmos3OmniTransformer, "from_single_file")


@pytest.mark.parametrize("scheme", ["fp8", "int8"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_an_explicit_text_encoder_quant_on_cosmos3_is_refused_before_the_load(
    cosmos3_runtime, monkeypatch, repo, scheme
):
    """Cosmos3 has no separate text encoder (the omni transformer reads the prompt tokens itself), so an explicit
    encoder scheme can never engage. The load route validates and runs the precision preflight before it takes the GPU
    (routes/video.py), which is where an impossible explicit precision is refused: after it, the resident model is
    evicted and the checkpoint (~35 GB for Nano) is pulled and built. Both rows have no encoder, so both are refused.
    The refusal must not tell the user to keep a dense bf16 encoder that does not exist."""
    # A GPU that runs the scheme: on the fake CPU host the generic encoder checks refuse for a reason of their own and
    # would hide whether Cosmos3 is refused at all. The opt-in that lets a declined explicit precision fall back is off.
    monkeypatch.setattr(video_mod, "te_quant_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(video_mod, "torchao_quantize_importable", lambda: True)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", raising = False)
    backend = VideoBackend()
    checks_passed = False
    with pytest.raises((ValueError, RuntimeError)) as refused:
        # The route's two checks before it takes the GPU, in its order, then the load its worker runs.
        fam = backend.validate_load_request(repo, model_kind = "pipeline", text_encoder_quant = scheme)
        video_mod.assert_video_precision_available(
            fam, model_kind = "pipeline", text_encoder_quant = scheme
        )
        checks_passed = True
        backend.load_pipeline(repo, model_kind = "pipeline", text_encoder_quant = scheme)
    assert _FakeCosmos3Pipeline.loads == []
    # Refused by those checks, not by the load once the route has evicted the resident model.
    assert checks_passed is False
    message = str(refused.value)
    assert re.search(r"text[ _-]encoder", message), message
    assert "dense bf16 encoder" not in message, message


@pytest.mark.parametrize(
    "parent, repo, family",
    [
        ("Cosmos3-Nano-exports", EDGE, "cosmos3-edge"),
        ("cosmos3-edge-runs", NANO, "cosmos3-nano"),
    ],
)
def test_a_local_folders_cosmos3_configs_outrank_a_path_naming_the_other_size(
    tmp_path, parent, repo, family
):
    """Content beats the name, as #13376 decided for header-checked files: a folder whose own configs are one Cosmos3
    size loads as that size even under a path that names the other (its card settings and memory plan follow)."""
    import json as _json

    from core.inference.video import _resolve_load_family_with_source

    hidden, model_type = (2048, "cosmos3_edge") if repo == EDGE else (4096, "cosmos3_omni")
    folder = tmp_path / parent / "checkpoint"
    (folder / "transformer").mkdir(parents = True)
    (folder / "model_index.json").write_text(
        _json.dumps({"_class_name": "Cosmos3OmniPipeline"}), encoding = "utf-8"
    )
    (folder / "config.json").write_text(_json.dumps({"model_type": model_type}), encoding = "utf-8")
    (folder / "transformer" / "config.json").write_text(
        _json.dumps({"_class_name": "Cosmos3OmniTransformer", "hidden_size": hidden}),
        encoding = "utf-8",
    )
    assert detect_video_family(str(folder)).name != family  # the path alone names the other size
    fam, _by_content = _resolve_load_family_with_source(str(folder), None, None)
    assert fam.name == family


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_cosmos3_load_status_reports_no_separate_text_encoder(cosmos3_runtime, repo):
    """The resolved record of an unset encoder precision must not name a dense bf16 encoder Cosmos3 does not have."""
    _backend, status = _loaded(repo)
    resolved = status["resolved"]["text_encoder_quant"]
    assert resolved["reason"] == "not applicable (no separate text encoder)", resolved
