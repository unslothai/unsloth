# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano / Cosmos3-Edge: families, video_cosmos3 glue, catalog, and VideoBackend's load and render over
test_video_backend's fake runtime. Hub metadata: fixtures/cosmos3_checkpoints.json at pinned revisions; card values
from README.md of nvidia/Cosmos3-Nano@e59a53c2 and nvidia/Cosmos3-Edge@344d602b."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.machinery
import importlib.util
import inspect
import json
import shutil
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import video as video_mod
from core.inference import video_cosmos3 as vc3
from core.inference import video_families as vf
from core.inference.diffusion_component_sizes import (
    TABLE_ROUNDING_GB,
    video_family_has_no_text_encoder,
    video_required_roles,
)
from core.inference.diffusion_families import (
    _installed_at_least,
    local_pipeline_components_are_complete,
    pipeline_class_requirement,
)
from core.inference.diffusion_flow_shift import apply_scheduler_overrides
from core.inference.video import VideoBackend
from hub.services.models import catalog_classification as cc
from models.models import LocalModelInfo

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_video_backend import (  # noqa: E402,F401
    _FakeWanDiT,
    _FakeWanVae,
    _assume_the_restricted_load_is_available,
    _data_url,
    _pin_video_fetch_to_the_named_repo,
    _roomy_host_ram,
    fake_runtime,
)

NANO, EDGE = "nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Edge"
SUPER, EDGE_POLICY = "nvidia/Cosmos3-Super", "nvidia/Cosmos3-Edge-Policy-DROID"
_FAMILY = {NANO: "cosmos3-nano", EDGE: "cosmos3-edge"}
# The cards' scheduler rebuilds: Nano README.md:887, Edge README.md:759-761.
_CARD_SCHEDULER = {
    NANO: {"flow_shift": 10.0},
    EDGE: {"flow_shift": 12.0, "use_karras_sigmas": False},
}
_OPT_OUT_ENV = "UNSLOTH_COSMOS3_SAFETY_CHECKER"
_REASONER = "Cosmos3 Reasoner checkpoint"
_CONDITIONED = "Policy, Sim, Transfer or FP8 build"
_PIPELINE_ONLY = "loads only as its full Diffusers pipeline"
_CHECKPOINTS = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "cosmos3_checkpoints.json").read_text(
        encoding = "utf-8"
    )
)


def _pinned(repo):
    (key,) = [k for k in _CHECKPOINTS if k.partition("@")[0] == repo]
    return key.partition("@")[2], _CHECKPOINTS[key]


def _ckpt(repo):
    return _pinned(repo)[1]


def _fam(name):
    return vf.detect_video_family("", override = name)


def _snapshot_path(repo, root = "/home/me/.cache/huggingface/hub"):
    return f"{root}/models--{repo.replace('/', '--')}/snapshots/{_pinned(repo)[0]}"


def _write_json(path, payload):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


def _write_configs(folder, repo):
    entry = _ckpt(repo)
    _write_json(folder / "model_index.json", entry["model_index"])
    _write_json(folder / "config.json", entry["config"])
    _write_json(folder / "transformer" / "config.json", entry["transformer_config"])
    return folder


def _planned_files(repo):
    info = SimpleNamespace(
        siblings = [SimpleNamespace(rfilename = n, size = s) for n, s in _ckpt(repo)["listing"]]
    )
    return dict(VideoBackend._base_download_files(info, "pipeline"))


def _write_planned_tree(root, repo):
    """The files a pipeline load stages, as stubs: the fixture's configs, shard indexes naming their shards, one byte
    per weight."""
    names = sorted(_planned_files(repo))
    _write_configs(root, repo)
    _write_json(root / "scheduler" / "scheduler_config.json", _ckpt(repo)["scheduler_config"])
    for name in names:
        path, folder = root / name, name.rpartition("/")[0]
        if path.exists():
            continue
        if name.endswith(".index.json"):
            shards = [
                n.rpartition("/")[2] for n in names if n.startswith(folder + "/") and "-of-" in n
            ]
            _write_json(path, {"weight_map": {f"w{i}": s for i, s in enumerate(shards)}})
        elif name.endswith(".json"):
            _write_json(path, {})
        else:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(b"\0")
    return root


def _hub_cache(monkeypatch, answer):
    lookups = []

    def try_to_load_from_cache(repo_id, filename, *args, **kwargs):
        lookups.append((repo_id, filename))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", try_to_load_from_cache)
    return lookups


def _set_guardrail(monkeypatch, state):
    """ "opted out" (env 0, no package), "installed" (package, no env) or "absent" (neither)."""
    if state == "opted out":
        monkeypatch.setenv(_OPT_OUT_ENV, "0")
    else:
        monkeypatch.delenv(_OPT_OUT_ENV, raising = False)
    module = None
    if state == "installed":
        module = types.ModuleType("cosmos_guardrail")
        module.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", module)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args, **kwargs):
        self.lines.append(("info", fmt % args))

    def warning(self, fmt, *args, **kwargs):
        self.lines.append(("warning", fmt % args))


@pytest.mark.parametrize(
    "identifier, override, expected",
    [
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None, "wan2.2-ti2v-5b"),
        ("MiniMaxAI/MiniMax-H3", None, "minimax-h3"),
        (NANO, None, "cosmos3-nano"),
        ("NVIDIA/COSMOS3-EDGE", None, "cosmos3-edge"),
        (_snapshot_path(NANO), None, "cosmos3-nano"),
        (_snapshot_path(EDGE, "C:\\Users\\me\\hub").replace("/", "\\"), None, "cosmos3-edge"),
        ("reasoner-lab/Cosmos3-Edge", None, "cosmos3-edge"),
        ("/srv/cosmos3-edge-fp8/exports/Cosmos3-Edge", None, "cosmos3-edge"),
        ("/data/Cosmos3-Nano-simple", None, "cosmos3-nano"),
        (SUPER, None, None),
        ("nvidia/Cosmos-Reason1-7B", None, None),
        ("/data/models/Cosmos3", None, None),
        ("nvidia/Cosmos3-Nano-Reasoner", None, None),
        (EDGE_POLICY, None, None),
        ("/models/cosmos3_edge_fp8/", None, None),
        ("x", " COSMOS3.EDGE ", "cosmos3-edge"),
        # A bare token would claim Super, the Reasoners and Cosmos-Predict2's "cosmos" GGUF arch alike.
        ("x", "cosmos", None),
        ("x", "cosmos3", None),
        ("x", "cosmos3_omni", None),
    ],
)
def test_detection(identifier, override, expected):
    fam = vf.detect_video_family(identifier, override = override)
    assert (fam.name if fam else None) == expected
    if fam is not None and override is None:
        assert vf.video_checkpoint_refusal(fam, identifier) is None


@pytest.mark.parametrize(
    "repo, kwargs, reason",
    [
        ("/models/Cosmos3-Nano-Reasoner/", {}, _REASONER),
        ("nvidia/Cosmos3-Nano-Policy-DROID", {}, _CONDITIONED),
        ("/data/COSMOS3.NANO.FP8", {"model_kind": "pipeline"}, _CONDITIONED),
        ("nvidia/Cosmos3-Nano-Reasoner", {"family_override": "cosmos3-nano"}, _REASONER),
        (EDGE_POLICY, {"family_override": "wan2.2-ti2v-5b"}, _CONDITIONED),
        (EDGE, {"display_repo_id": "nvidia/Cosmos3-Edge-Reasoner"}, _REASONER),
        (
            "someone/files",
            {"gguf_filename": "cosmos3-nano-reasoner-Q4_K_M.gguf", "model_kind": "gguf"},
            _REASONER,
        ),
        (NANO, {"gguf_filename": "cosmos3-edge-Q8_0.gguf"}, _PIPELINE_ONLY),
        (
            NANO,
            {"gguf_filename": "cosmos3_nano.safetensors", "model_kind": "single_file"},
            _PIPELINE_ONLY,
        ),
    ],
)
def test_non_generator_builds_are_refused_by_name(repo, kwargs, reason):
    with pytest.raises(ValueError) as refused:
        VideoBackend().validate_load_request(repo, **kwargs)
    assert all(part in str(refused.value) for part in (reason, NANO, EDGE)), refused.value


@pytest.mark.parametrize(
    "folder, repo, expected",
    [
        ("my-render-model", NANO, "cosmos3-nano"),
        ("my-render-model", EDGE, "cosmos3-edge"),
        ("my-render-model", SUPER, None),
        ("my-render-model", EDGE_POLICY, None),
        # The shipped configs outrank a path naming the other size.
        ("Cosmos3-Nano-exports/checkpoint", EDGE, "cosmos3-edge"),
        ("cosmos3-edge-runs/checkpoint", NANO, "cosmos3-nano"),
    ],
)
def test_a_local_folder_is_named_by_its_shipped_configs(tmp_path, folder, repo, expected):
    from core.inference.video import _detect_load_family
    from core.training.diffusion_train_common import resolve_trainable_family

    path = str(_write_configs(tmp_path / folder, repo))
    by_name = vf.detect_video_family(path)
    assert by_name is None if folder == "my-render-model" else by_name.name != expected
    assert vf.local_cosmos3_pipeline_family(path) == expected
    fam = _detect_load_family(path, None, None)
    assert (fam.name if fam else None) == expected
    if expected:
        named = folder == "my-render-model"
        with pytest.raises(ValueError, match = f"{expected}.*ltx-2" if named else "can't train yet"):
            resolve_trainable_family(path)


@pytest.mark.parametrize("hidden", [4096, 2048, 5120])
@pytest.mark.parametrize("model_type", ["cosmos3_omni", "cosmos3_edge", "qwen3_vl", None])
def test_local_folder_matrix(tmp_path, model_type, hidden):
    entry = _ckpt(NANO)
    _write_json(tmp_path / "model_index.json", entry["model_index"])
    if model_type is not None:
        _write_json(tmp_path / "config.json", {"model_type": model_type})
    transformer = {**entry["transformer_config"], "hidden_size": hidden}
    _write_json(tmp_path / "transformer" / "config.json", transformer)
    served = {
        (t, h): f for t, h, f in (("cosmos3_omni", 4096, "nano"), ("cosmos3_edge", 2048, "edge"))
    }
    served.update({(None, 4096): "nano", (None, 2048): "edge"})
    name = served.get((model_type, hidden))
    assert vf.local_cosmos3_pipeline_family(str(tmp_path)) == (name and f"cosmos3-{name}")


@pytest.mark.parametrize(
    "broken, text",
    [("model_index.json", None), ("model_index.json", '{"_class_name": "WanPipeline"}')]
    + [("model_index.json", "[]"), ("transformer/config.json", "4096")],
)
def test_an_unreadable_or_foreign_folder_is_not_claimed(tmp_path, broken, text):
    folder = _write_configs(tmp_path / "renamed", EDGE)
    if text is None:
        (folder / broken).unlink()
    else:
        (folder / broken).write_text(text, encoding = "utf-8")
    for path in (str(folder), None, "", str(folder / "config.json"), EDGE):
        assert vf.local_cosmos3_pipeline_family(path) is None, path


@pytest.mark.parametrize("scheme", ["fp8", "int8"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_an_explicit_text_encoder_quant_is_refused_at_validation(
    fake_runtime, monkeypatch, repo, scheme
):
    _set_guardrail(monkeypatch, "opted out")
    monkeypatch.setattr(video_mod, "te_quant_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(video_mod, "torchao_quantize_importable", lambda: True)
    with pytest.raises(ValueError) as refused:
        VideoBackend().validate_load_request(repo, model_kind = "pipeline", text_encoder_quant = scheme)
    assert "no separate text" in str(refused.value)
    assert "dense bf16 encoder" not in str(refused.value)


@pytest.mark.parametrize(
    "table, expected",
    [((30.4, 0.0, 1.4), True), ((30.4, 0, 1.4), True), ((30.4, 0.05, 1.4), False)]
    + [((30.4, float("nan"), 1.4), False), ((30.4,), False), (None, False)],
)
def test_no_text_encoder_is_a_zero_text_term(table, expected):
    assert video_family_has_no_text_encoder(SimpleNamespace(bf16_components_gb = table)) is expected
    if table == (30.4, 0.0, 1.4):
        no_encoder = {f.name for f in vf._FAMILIES if video_family_has_no_text_encoder(f)}
        assert no_encoder == set(vf.COSMOS3_FAMILIES)
        assert video_required_roles(_fam("cosmos3-edge")) == ("transformer", "vae")
        assert "text_encoder" in video_required_roles(_fam("wan2.2-ti2v-5b"))


@pytest.mark.parametrize(
    "value, opted_out",
    [(v, True) for v in ("0", "FALSE", "no", " Off\n")]
    + [(v, False) for v in (None, "", "1", "true", "disabled")],
)
def test_the_opt_out_takes_only_explicit_off_values(monkeypatch, value, opted_out):
    if value is None:
        monkeypatch.delenv(_OPT_OUT_ENV, raising = False)
    else:
        monkeypatch.setenv(_OPT_OUT_ENV, value)
    assert vc3.safety_checker_opted_out() is opted_out


@pytest.mark.parametrize(
    "answer, installed",
    [(True, True), (False, False), (ImportError("finder"), False), (ValueError("no spec"), False)],
)
def test_guardrail_presence_is_find_spec_on_cosmos_guardrail(monkeypatch, answer, installed):
    real = importlib.util.find_spec

    def find_spec(name, package = None):
        if name != "cosmos_guardrail":
            return real(name, package)
        if isinstance(answer, Exception):
            raise answer
        return importlib.machinery.ModuleSpec(name, None) if answer else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert vc3.guardrail_installed() is installed


_NANO_INDEX = json.dumps(_ckpt(NANO)["model_index"])
_EDGE_INDEX = json.dumps(_ckpt(EDGE)["model_index"])
_EDGE_ON = json.dumps({**_ckpt(EDGE)["model_index"], "enable_safety_checker": True})


@pytest.mark.parametrize(
    "source, repo, content, enabled",
    [
        ("index", NANO, _NANO_INDEX, True),
        ("index", EDGE, _EDGE_INDEX, False),
        ("index", EDGE, _EDGE_ON, True),
        ("index", EDGE, _EDGE_INDEX.replace("false", '"false"'), True),
        ("folder", None, _NANO_INDEX, True),
        ("folder", None, _EDGE_INDEX, False),
        ("folder", None, None, True),
        ("folder", None, "{not json", True),
        ("cache", EDGE, _EDGE_ON, True),
        ("cache", NANO, None, True),
        ("cache", EDGE.upper(), None, False),
        ("cache", EDGE, ValueError("bad repo id"), False),
    ],
)
def test_the_checkpoints_index_decides_the_safety_checker(
    monkeypatch, tmp_path, source, repo, content, enabled
):
    monkeypatch.chdir(tmp_path)
    answer = content if isinstance(content, Exception) else None
    if source == "cache" and isinstance(content, str):
        answer = str(tmp_path / "blob.json")
        Path(answer).write_text(content, encoding = "utf-8")
    lookups = _hub_cache(monkeypatch, answer)
    if source == "index":
        assert vc3.checkpoint_enables_safety_checker(repo, json.loads(content)) is enabled
    elif source == "folder":
        folder = tmp_path / "Cosmos3-Edge"
        folder.mkdir()
        if content is not None:
            (folder / "model_index.json").write_text(content, encoding = "utf-8")
        assert vc3.checkpoint_enables_safety_checker(str(folder)) is enabled
    else:
        assert vc3.checkpoint_enables_safety_checker(repo) is enabled
    assert lookups == ([(repo, "model_index.json")] if source == "cache" else [])


@pytest.mark.parametrize("guardrail", ["opted out", "installed", "absent"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_guardrail_refusal_and_pipeline_kwargs(monkeypatch, repo, guardrail):
    _hub_cache(monkeypatch, None)
    _set_guardrail(monkeypatch, guardrail)
    log = _Log()
    monkeypatch.setattr(vc3, "logger", log)
    refusal = vc3.guardrail_refusal(repo)
    if repo == NANO and guardrail == "absent":
        for part in (f"'{NANO}'", "pip install cosmos-guardrail", f"{_OPT_OUT_ENV}=0"):
            assert part in refusal, part
        assert vc3.guardrail_refusal(NANO, index = _ckpt(EDGE)["model_index"]) is None
    else:
        assert refusal is None
    opted_out = guardrail == "opted out"
    expected = {"sound_tokenizer": None, **({"enable_safety_checker": False} if opted_out else {})}
    assert vc3.pipeline_kwargs(repo) == expected
    warned = [line for level, line in log.lines if level == "warning"]
    assert len(warned) == (repo == NANO and opted_out)
    assert all(_OPT_OUT_ENV in line and repo in line for line in warned)


def _listing(repo, *extra):
    return tuple(n for n, _s in _ckpt(repo)["listing"]) + extra


@pytest.mark.parametrize(
    "listing, expected",
    [
        (_listing(NANO), True),
        (_listing(EDGE), True),
        (("model_index.json", "text_encoder/c.json", "tokenizer/t.json"), False),
        (("modular_model_index.json", "text_encoder/c.json", "connectors/c.json"), False),
        (("modular_model_index.json", "text_encoder/m.bin", "transformer_ref/c.json"), False),
        ((), False),
        (_listing(EDGE, "text_encoder/model.safetensors"), False),
        (("model_index.json", "text_tokenizer.json"), False),
    ],
)
def test_is_cosmos3_listing(listing, expected):
    assert vc3.is_cosmos3_listing(iter(sorted(listing))) is expected


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_the_download_plan_stages_every_component_the_pipeline_reads(tmp_path, repo):
    # The root modular_model_index.json once made the planner read it as H3's layout and drop text_tokenizer/.
    planned = _planned_files(repo)
    assert "model_index.json" in planned
    for folder in ("scheduler/", "text_tokenizer/", "transformer/", "vae/"):
        wanted = {name for name in _listing(repo) if name.startswith(folder)}
        assert wanted and wanted <= set(planned), folder
    root = _write_planned_tree(tmp_path / "snapshot", repo)
    assert local_pipeline_components_are_complete(root, "model_index.json")
    # sound_tokenizer is passed as None, so diffusers' hub-id load skips it too; vision_encoder it still demands.
    assert not any(name.startswith("sound_tokenizer/") for name in planned)
    vision = {name for name in _listing(repo) if name.startswith("vision_encoder/")}
    assert vision <= set(planned)
    for unused in ("sound_tokenizer", "vision_encoder"):
        shutil.rmtree(root / unused, ignore_errors = True)
    assert local_pipeline_components_are_complete(root, "model_index.json")
    _write_json(root / "sound_tokenizer" / "config.json", {})
    assert local_pipeline_components_are_complete(root, "model_index.json")


class _FakeScheduler:
    def __init__(self, config):
        self.config = dict(config)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls({**dict(config), **kwargs})


class _RejectingScheduler(_FakeScheduler):
    @classmethod
    def from_config(cls, config, **kwargs):
        raise NotImplementedError("bh3 is not implemented")


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_scheduler_overrides(repo):
    shipped = _ckpt(repo)["scheduler_config"]
    overrides = _fam(_FAMILY[repo]).scheduler_overrides
    pipe, log = SimpleNamespace(scheduler = _FakeScheduler(shipped)), _Log()
    assert apply_scheduler_overrides(pipe, overrides, log) is True
    assert pipe.scheduler.config == {**shipped, **_CARD_SCHEDULER[repo]}
    rebuilt = pipe.scheduler
    assert apply_scheduler_overrides(pipe, overrides, log) is False and pipe.scheduler is rebuilt
    assert [level for level, _ in log.lines] == ["info"]
    for pipe, given in (
        (SimpleNamespace(scheduler = _FakeScheduler(shipped)), ()),
        (SimpleNamespace(scheduler = _FakeScheduler(shipped)), None),
        (SimpleNamespace(), overrides),
        (SimpleNamespace(scheduler = object()), overrides),
    ):
        before = getattr(pipe, "scheduler", None)
        assert apply_scheduler_overrides(pipe, given) is False
        assert getattr(pipe, "scheduler", None) is before
    pipe, log = SimpleNamespace(scheduler = _RejectingScheduler(shipped)), _Log()
    assert apply_scheduler_overrides(pipe, overrides, log) is False
    assert type(pipe.scheduler) is _RejectingScheduler
    ((level, line),) = log.lines
    assert level == "warning" and "flow_shift" in line and "not implemented" in line


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_the_overrides_sample_the_cards_schedule_on_the_real_unipc(repo):
    pytest.importorskip("diffusers", reason = "Backend CI installs no diffusers")
    import numpy as np
    import torch
    from diffusers import UniPCMultistepScheduler as UniPC

    shipped, steps = _ckpt(repo)["scheduler_config"], _fam(_FAMILY[repo]).default_steps
    pipe = SimpleNamespace(scheduler = UniPC.from_config(shipped))
    assert apply_scheduler_overrides(pipe, _fam(_FAMILY[repo]).scheduler_overrides) is True
    card = UniPC.from_config(shipped, **_CARD_SCHEDULER[repo])
    assert dict(pipe.scheduler.config) == dict(card.config)
    # Under use_native_flow_schedule (Edge's index) the pipeline passes its own linspace sigmas.
    native = _ckpt(repo)["model_index"].get("use_native_flow_schedule", False)
    t = np.linspace(1.0 - 1.0 / 1000, 0.0, steps + 1)[:-1]

    def sigmas(scheduler):
        scheduler.set_timesteps(steps, device = "cpu", **({"sigmas": t} if native else {}))
        return scheduler.sigmas.clone()

    ours, before = sigmas(pipe.scheduler), sigmas(UniPC.from_config(shipped))
    if repo == EDGE:
        expected = torch.from_numpy(12.0 * t / (1.0 + 11.0 * t))
        assert torch.allclose(ours[:-1].double(), expected, atol = 1e-6)
        assert not torch.equal(ours, before)
    else:
        # Nano's shipped Karras sigmas pre-empt the only branch reading flow_shift: the shift is inert.
        assert torch.equal(ours, before)


class _FakeCosmos3Vae(_FakeWanVae):
    def __init__(self):
        super().__init__()
        self.use_tiling, self.calls, self.encode_raises, self.strides = False, [], [], []
        self.spatial_compression_ratio, self.tile_sample_min_height, self.tile_sample_min_width = (
            16,
            256,
            256,
        )
        self.tile_sample_stride_height = self.tile_sample_stride_width = 192

    def enable_tiling(self):
        super().enable_tiling()
        self.use_tiling = True

    def encode(self, *args, **kwargs):
        self.calls.append("encode tiled" if self.use_tiling else "encode untiled")
        if self.encode_raises:
            raise self.encode_raises.pop(0)
        return ("latents", args, kwargs)

    def decode(self, *args, **kwargs):
        self.calls.append("decode tiled" if self.use_tiling else "decode untiled")
        self.strides.append((self.tile_sample_stride_height, self.tile_sample_stride_width))
        return super().decode(*args, **kwargs)


# torch.OutOfMemoryError by class name, without importing torch; a RuntimeError by its message.
_OOM_CLASS = type("OutOfMemoryError", (RuntimeError,), {})("x")
_OOM = RuntimeError("CUDA out of memory.")
_RETRIED = ["encode untiled", "empty_cache", "encode tiled"]


@pytest.mark.parametrize(
    "tiled, raises, calls, fails",
    [
        (True, [], ["encode untiled"], False),
        (False, [], ["encode untiled"], False),
        (True, [_OOM_CLASS], _RETRIED, False),
        (True, [_OOM], _RETRIED, False),
        (True, [_OOM, _OOM], _RETRIED, True),
        (True, [ValueError("expected a 5-D clip")], ["encode untiled"], True),
        (False, [_OOM_CLASS], ["encode untiled"], True),
    ],
)
def test_untiled_encode(monkeypatch, tiled, raises, calls, fails):
    vae = _FakeCosmos3Vae()
    if tiled:
        vae.enable_tiling()
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(empty_cache = lambda: vae.calls.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    log = _Log()
    assert vc3.install_untiled_encode(SimpleNamespace(vae = vae), logger = log) is True
    vae.encode_raises = list(raises)
    clip = object()
    if fails:
        with pytest.raises(type(raises[-1])) as raised:
            vae.encode(clip)
        assert raised.value is raises[-1]
    else:
        assert vae.encode(clip, return_dict = False) == ("latents", (clip,), {"return_dict": False})
    assert vae.calls == calls
    assert vae.use_tiling is tiled
    assert [level for level, _ in log.lines] == ["info"] + ["warning"] * (calls == _RETRIED)


def test_untiled_encode_installs_once_and_needs_a_callable_encode():
    vae = _FakeCosmos3Vae()
    pipe = SimpleNamespace(vae = vae)
    assert vc3.install_untiled_encode(pipe) is True
    wrapped = vae.encode
    assert vc3.install_untiled_encode(pipe) is True and vae.encode is wrapped
    bad = SimpleNamespace(use_tiling = True, encode = "encode")
    for pipe in (SimpleNamespace(), SimpleNamespace(vae = None), SimpleNamespace(vae = bad)):
        assert vc3.install_untiled_encode(pipe) is False


# Cosmos3OmniPipeline.__call__'s parameters (diffusers 0.41); unknown keywords raise TypeError as there.
_CALL_PARAMS = (
    "prompt negative_prompt image video condition_frame_indexes_vision condition_video_keep num_frames"
    " height width fps num_inference_steps guidance_scale enable_sound generator latents sound_latents"
    " action_latents action output_type return_dict use_system_prompt callback_on_step_end"
    " callback_on_step_end_tensor_inputs add_resolution_template add_duration_template"
    " enable_safety_check mixed_precision_format mixed_precision_first_steps mixed_precision_last_steps"
    " mixed_precision_reasoner_policy"
).split()


class _FakeCosmos3Pipeline:
    loads: list = []

    def __init__(self, scheduler_config):
        self.transformer, self.vae = _FakeWanDiT(), _FakeCosmos3Vae()
        self.scheduler = _FakeScheduler(scheduler_config)
        self.passed, self.outputs, self.steps_computed = None, [], 0
        self.on_step, self._interrupt = None, False

    @property
    def components(self):
        return {"transformer": self.transformer, "vae": self.vae, "scheduler": self.scheduler}

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.loads.append({"source": source, **kwargs})
        return cls(_ckpt(EDGE if str(source).lower() == EDGE.lower() else NANO)["scheduler_config"])

    def to(self, device):
        return self

    @contextlib.contextmanager
    def progress_bar(self, total = None):
        yield SimpleNamespace(update = lambda n = 1: None)

    def maybe_free_model_hooks(self):
        pass

    def __call__(self, **kwargs):
        if set(kwargs) - set(_CALL_PARAMS):
            raise TypeError(f"unexpected keywords {sorted(set(kwargs) - set(_CALL_PARAMS))}")
        self.passed, self._interrupt = dict(kwargs), False
        steps, callback = (
            int(kwargs.get("num_inference_steps", 35)),
            kwargs.get("callback_on_step_end"),
        )
        self.vae.encode(("clip", kwargs.get("image")))
        latents = object()
        with self.progress_bar(total = steps) as bar:
            for i in range(steps):
                if self._interrupt:
                    continue
                self.steps_computed += 1
                if self.on_step is not None:
                    self.on_step(i)
                if callback is not None:
                    latents = callback(self, i, 1000.0, {"latents": latents}).pop(
                        "latents", latents
                    )
                bar.update()
        self.vae.decode(latents)
        count = 1 + 4 * ((int(kwargs.get("num_frames") or 189) - 1) // 4)
        self.outputs.append(SimpleNamespace(video = [object() for _ in range(count)]))
        return self.outputs[-1]

    __call__.__signature__ = inspect.Signature(
        [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in ("self", *_CALL_PARAMS)
        ]
    )


def test_the_fakes_match_the_installed_pipeline():
    diffusers = pytest.importorskip("diffusers", reason = "Backend CI installs no diffusers")
    minimum, _ = pipeline_class_requirement("Cosmos3OmniPipeline")
    if not _installed_at_least(diffusers.__version__, minimum):
        pytest.skip(f"diffusers {diffusers.__version__} predates Cosmos3OmniPipeline ({minimum})")
    from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as module

    real = diffusers.Cosmos3OmniPipeline
    call = inspect.signature(real.__call__).parameters
    assert list(call)[1:] == _CALL_PARAMS
    assert call["enable_sound"].default is False
    assert {"sound_tokenizer", "enable_safety_checker"} <= set(
        inspect.signature(real.__init__).parameters
    )
    assert "sound_tokenizer" in real._optional_components
    assert "video" in {f.name for f in dataclasses.fields(module.Cosmos3OmniPipelineOutput)}
    assert not hasattr(diffusers.Cosmos3OmniTransformer, "from_single_file")
    tree = ast.parse(inspect.getsource(module))
    imports = {
        (n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert ("cosmos_guardrail", "CosmosSafetyChecker") in imports


@pytest.fixture
def cosmos3_runtime(fake_runtime, monkeypatch):
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "Cosmos3OmniPipeline", _FakeCosmos3Pipeline, raising = False)
    _FakeCosmos3Pipeline.loads = []
    _set_guardrail(monkeypatch, "opted out")
    encoded = []

    def _encode(frames, fps, audio, pipe):
        encoded.append({"frames": frames, "fps": fps, "audio": audio})
        return b"MP4"

    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(_encode))
    yield encoded


def _loaded(repo):
    backend = VideoBackend()
    return backend, backend.load_pipeline(repo, model_kind = "pipeline")


@pytest.mark.parametrize(
    "repo, guardrail, safety",
    [
        (NANO, "opted out", {"enable_safety_checker": False}),
        (EDGE, "opted out", {"enable_safety_checker": False}),
        (NANO, "installed", {}),
        (EDGE, "absent", {}),
    ],
)
def test_load(cosmos3_runtime, monkeypatch, repo, guardrail, safety):
    _set_guardrail(monkeypatch, guardrail)
    backend, status = _loaded(repo)
    (load,) = _FakeCosmos3Pipeline.loads
    assert load["source"] == repo and "sound_tokenizer" in load and load["sound_tokenizer"] is None
    assert {k: v for k, v in load.items() if k == "enable_safety_checker"} == safety
    pipe = backend._state.pipe
    assert pipe.scheduler.config == {**_ckpt(repo)["scheduler_config"], **_CARD_SCHEDULER[repo]}
    steps, frames, canvas = {NANO: (35, 189, [1280, 720]), EDGE: (20, 121, [832, 480])}[repo]
    got = status["defaults"]
    assert (got["steps"], got["guidance"], got["num_frames"], got["fps"]) == (
        steps,
        6.0,
        frames,
        24,
    )
    assert got["resolution_presets"][0] == canvas
    assert status["family"] == _FAMILY[repo] and status["has_audio"] is False
    assert status["supports_first_frame"] is True and status["supports_last_frame"] is False
    assert status["vae_tiling"] is True
    assert getattr(pipe.vae.encode, "_unsloth_untiled_encode", False) is True
    reason = status["resolved"]["text_encoder_quant"]["reason"]
    assert reason == "not applicable (no separate text encoder)"


def test_a_wan_load_keeps_the_stock_encode(cosmos3_runtime, monkeypatch):
    monkeypatch.setattr(_FakeWanVae, "encode", lambda self, *a, **k: a, raising = False)
    backend, status = _loaded("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert status["vae_tiling"] is True
    assert not getattr(backend._state.pipe.vae.encode, "_unsloth_untiled_encode", False)


def test_nano_without_the_guardrail_is_refused_before_the_load_starts(cosmos3_runtime, monkeypatch):
    _set_guardrail(monkeypatch, "absent")
    started = []
    monkeypatch.setattr(VideoBackend, "_run_load", lambda self, **kwargs: started.append(kwargs))
    backend = VideoBackend()
    with pytest.raises(ValueError, match = f"{_OPT_OUT_ENV}=0"):
        backend.begin_load(NANO, model_kind = "pipeline")
    assert started == [] and backend.loading_repo_ids() == ()
    with pytest.raises(ValueError, match = "cosmos_guardrail"):
        backend.load_pipeline(NANO, model_kind = "pipeline")
    assert _FakeCosmos3Pipeline.loads == []
    with pytest.raises(ValueError, match = "limited to"):
        backend.validate_load_request("someone/cosmos3-nano-repack", model_kind = "pipeline")


@pytest.mark.parametrize("folder", ["named", "renamed", "export", "snapshot"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_a_local_pipeline_folder_validates(
    cosmos3_runtime, monkeypatch, tmp_path_factory, repo, folder
):
    names = {"named": repo.partition("/")[2], "snapshot": _snapshot_path(repo, "hub")}
    root = _write_planned_tree(tmp_path_factory.mktemp("m") / names.get(folder, "my-model"), repo)
    if folder == "export":
        (root / "config.json").unlink()
    _set_guardrail(monkeypatch, "absent" if repo == EDGE else "opted out")
    assert VideoBackend().validate_load_request(str(root)).name == _FAMILY[repo]


def _first_frame(size):
    return {"first_frame": _data_url(*size)} if size else {}


@pytest.mark.parametrize(
    "repo, first_frame, steps, frames, size, stride",
    [
        (NANO, None, 35, 189, (1280, 720), 12),
        (NANO, (1920, 1080), 35, 189, (1280, 720), 12),
        (EDGE, None, 20, 121, (832, 480), 11),
        (EDGE, (1080, 1920), 20, 121, (480, 832), 11),
    ],
)
def test_generate(cosmos3_runtime, monkeypatch, repo, first_frame, steps, frames, size, stride):
    from core.inference import diffusion as diffusion_mod
    from core.inference import video_i2v

    decoded, decode = [], diffusion_mod.decode_b64_image
    monkeypatch.setattr(
        diffusion_mod,
        "decode_b64_image",
        lambda *a, **k: decoded.append(decode(*a, **k)) or decoded[-1],
    )
    offloaded, offload = [], video_i2v.offload_vae_after_encode
    monkeypatch.setattr(
        video_i2v, "offload_vae_after_encode", lambda p: offloaded.append(p) or offload(p)
    )
    backend, _status = _loaded(repo)
    pipe, seen = backend._state.pipe, []
    pipe.on_step = lambda i: seen.append(backend.generate_progress())
    result = backend.generate(prompt = "a robot arm", **_first_frame(first_frame))
    passed = pipe.passed
    assert (passed["num_inference_steps"], passed["guidance_scale"], passed["fps"]) == (
        steps,
        6.0,
        24.0,
    )
    assert (passed["width"], passed["height"], passed["num_frames"]) == (*size, frames)
    assert (result["width"], result["height"], result["num_frames"]) == (*size, frames)
    assert result["conditioning"] == ("i2va" if first_frame else "t2va")
    if first_frame:
        (image,) = decoded
        assert passed["image"] is image and image.size == first_frame
    assert [(g["phase"], g["total_steps"]) for g in seen] == [("denoise", steps)] * steps
    (export,) = cosmos3_runtime
    assert export["frames"] is pipe.outputs[-1].video and export["audio"] is None
    assert offloaded == [pipe] and pipe.vae.calls == ["encode untiled", "decode tiled"]
    assert pipe.vae.strides == [(stride * 16, stride * 16)]
    assert (pipe.vae.tile_sample_stride_height, pipe.vae.tile_sample_stride_width) == (192, 192)

    pipe.on_step = None
    request = {"negative_prompt": "blurry", "steps": 2, "fps": 16, "guidance_2": 3.5}
    result = backend.generate(prompt = "a robot arm", **request, **_first_frame(first_frame))
    passed = pipe.passed
    expected = {"prompt", "negative_prompt", "num_inference_steps", "guidance_scale", "width"}
    expected |= {"height", "num_frames", "fps", "generator", "callback_on_step_end"}
    expected |= {"image"} if first_frame else set()
    off = {"add_resolution_template": False, "add_duration_template": False} if repo == EDGE else {}
    assert set(passed) == expected | set(off) and {k: passed[k] for k in off} == off
    assert passed["fps"] == 16.0 and type(passed["fps"]) is float
    assert passed["negative_prompt"] == "blurry"
    assert result["fps"] == 16 and cosmos3_runtime[-1]["fps"] == 16
    backend.generate(prompt = "p", steps = 1)
    assert "image" not in pipe.passed


@pytest.mark.parametrize("first_frame, recorded", [(None, True), ((1920, 1080), False)])
def test_a_pending_background_compile_records_only_text_to_video(
    cosmos3_runtime, monkeypatch, first_frame, recorded
):
    from core.inference import diffusion_bg_compile as bg_compile

    skipped = []

    @contextlib.contextmanager
    def no_record():
        skipped.append(True)
        yield

    monkeypatch.setattr(bg_compile, "no_record", no_record)
    backend, _status = _loaded(EDGE)
    bg = bg_compile.BackgroundCompile(object())
    assert bg.pending()
    backend._state = dataclasses.replace(backend._state, bg_compile = bg)
    backend.generate(prompt = "a robot arm", steps = 1, **_first_frame(first_frame))
    assert (not skipped) is recorded


@pytest.mark.parametrize(
    "repo, requested, frames", [(NANO, 400, 397), (EDGE, 30, 53), (EDGE, 200, 149)]
)
def test_generate_snaps_frames_and_size_onto_the_family_lattice(
    cosmos3_runtime, repo, requested, frames
):
    backend, _status = _loaded(repo)
    backend.generate(prompt = "p", steps = 1, num_frames = requested, width = 1000, height = 700)
    passed = backend._state.pipe.passed
    assert _ckpt(repo)["vae_config"]["scale_factor_spatial"] == 16
    assert (passed["num_frames"], passed["width"], passed["height"]) == (frames, 992, 688)


def test_call_kwargs_skip_a_key_the_pipeline_does_not_take(cosmos3_runtime):
    backend, _status = _loaded(NANO)
    extra = (("use_resolution_template", False), ("add_duration_template", False))
    family = dataclasses.replace(backend._state.family, call_kwargs = extra)
    backend._state = dataclasses.replace(backend._state, family = family)
    backend.generate(prompt = "p", steps = 2)
    passed = backend._state.pipe.passed
    assert "use_resolution_template" not in passed and passed["add_duration_template"] is False


def test_a_last_frame_is_refused_and_a_cancel_interrupts_the_denoise(cosmos3_runtime):
    from core.inference.video_families import VIDEO_CANCELLED_MSG

    backend, _status = _loaded(EDGE)
    pipe = backend._state.pipe
    frames = {"first_frame": _data_url(64, 64), "last_frame": _data_url(64, 64)}
    for call in (backend.begin_generate, backend.generate):
        with pytest.raises(ValueError, match = "no last frame"):
            call(prompt = "p", steps = 2, **frames)
    assert pipe.passed is None and backend.generate_progress()["active"] is False
    pipe.on_step = lambda i: backend.cancel_generate() if i == 0 else None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "p", steps = 4)
    assert pipe.steps_computed == 1 and cosmos3_runtime == []
    assert backend.generate_progress()["active"] is False
    pipe.on_step = None
    backend.generate(prompt = "p", steps = 3)
    assert pipe.steps_computed == 4


# Nano card :887-898, :113 (5-400 frames); Edge card :759-776, :102 (50-150). Windows: the 4k+1 counts inside.
_CARD = {"cosmos3-nano": (NANO, (5, 397)), "cosmos3-edge": (EDGE, (53, 149))}


@pytest.mark.parametrize("name", sorted(_CARD))
def test_rows_follow_the_card_and_the_shipped_checkpoint(name):
    repo, window = _CARD[name]
    fam, vae, listing = _fam(name), _ckpt(repo)["vae_config"], _ckpt(repo)["listing"]
    assert (fam.pipeline_class, fam.i2v_pipeline_class) == ("Cosmos3OmniPipeline",) * 2
    assert fam.base_repo == repo and (fam.min_num_frames, fam.max_num_frames) == window
    assert (fam.frame_step, fam.resolution_multiple) == (
        vae["scale_factor_temporal"],
        vae["scale_factor_spatial"],
    )
    assert fam.trainable is False and fam.i2v_fits_first_frame is False
    for term, folder in ((0, "transformer/"), (2, "vae/")):
        folder_gb = sum(s for n, s in listing if n.startswith(folder)) / 1e9
        assert abs(fam.bf16_components_gb[term] - folder_gb) <= TABLE_ROUNDING_GB, folder
    vf.validate_video_request_shape(fam, *fam.resolution_presets[0], fam.default_num_frames)


@pytest.mark.parametrize(
    "name, shape, fragment",
    [
        ("cosmos3-nano", {"num_frames": 188}, "185 and 189"),
        ("cosmos3-nano", {"num_frames": 1}, "5 to 397"),
        ("cosmos3-edge", {"width": 840, "height": 480}, "multiples of 16"),
        ("cosmos3-edge", {"width": 1280, "height": 720}, "409,600 pixels"),
    ],
)
def test_shapes_off_the_lattice_window_or_grid_are_refused(name, shape, fragment):
    with pytest.raises(vf.VideoShapeError, match = fragment):
        vf.validate_video_request_shape(_fam(name), **shape)


@pytest.mark.parametrize("fam", vf._FAMILIES, ids = lambda f: f.name)
def test_every_window_end_and_duration_preset_is_a_valid_count(fam):
    from routes.video import _frames_for_seconds

    step, offset = max(1, fam.frame_step), max(1, fam.frame_offset)
    floor = max(offset, fam.min_num_frames)
    for end in (floor,) if fam.max_num_frames is None else (floor, fam.max_num_frames):
        assert (end - offset) % step == 0, end
        vf.validate_video_request_shape(fam, num_frames = end)
    defaults = {"fps": fam.default_fps, "frame_step": step, "frame_offset": fam.frame_offset}
    for seconds in fam.duration_presets:
        page = max(1, math.floor((seconds * fam.default_fps - offset) / step + 0.5)) * step + offset
        for frames in {_frames_for_seconds(seconds, defaults), page}:
            vf.validate_video_request_shape(fam, num_frames = frames)


@pytest.mark.parametrize(
    "ids, expected",
    [
        ((None, None, None, _snapshot_path(NANO)), (35, 6.0)),
        ((None, EDGE), (20, 6.0)),
        (("/data/fusionx/Cosmos3-Edge",), (20, 6.0)),
        (("/data/hunyuanvideo/Cosmos3-Nano",), (35, 6.0)),
        (("/models/Cosmos3-Nano/wan2.1-t2v-14b-Q4_K_M.gguf",), (20, 6.0)),
        (("/data/cosmos3-edge/LTX-2",), (40, 4.0)),
    ],
)
def test_generation_defaults(ids, expected):
    assert vf.default_video_generation_params(*ids) == expected


@pytest.mark.parametrize(
    "installed, available",
    [("0.39.0", False), ("0.40.0", False), ("0.41.0", True), ("0.41.0.dev0", True)],
)
def test_rows_hide_on_a_diffusers_without_edges_native_flow_schedule(
    monkeypatch, installed, available
):
    from core.inference import diffusion_families as df

    assert _ckpt(EDGE)["model_index"]["use_native_flow_schedule"] is True
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    monkeypatch.setattr(df, "_installed_diffusers_version", lambda: installed)
    assert {df.family_pipeline_available(_fam(n)) for n in vf.COSMOS3_FAMILIES} == {available}


_PIPELINE = ("model_index.json", "transformer/diffusion_pytorch_model.safetensors")
_SHARDS = ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors")


@pytest.mark.parametrize(
    "repo_id, files, video",
    [
        (EDGE, _PIPELINE, True),
        ("unsloth/Cosmos3-Edge", _PIPELINE, True),
        ("someone/Cosmos3-Edge", _PIPELINE, False),
        (SUPER, _PIPELINE, False),
        (EDGE, _SHARDS, False),
        ("unsloth/Cosmos3-Nano-unsloth-bnb-4bit", _SHARDS, False),
        ("nvidia/Cosmos3-Edge-Reasoner", _SHARDS, False),
    ],
)
def test_the_cached_listing_offers_only_trusted_generator_pipelines_as_video(repo_id, files, video):
    entries = [SimpleNamespace(file_name = n, size_on_disk = 1, blob_path = None) for n in files]
    repo = SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = Path("/hf-cache/x"),
        revisions = [SimpleNamespace(files = entries)],
    )
    assert cc._cached_repo_task(repo) == (cc._VIDEO_GEN_TASK if video else None)
    assert cc._repo_is_diffusers(repo) is (files == _PIPELINE)


def test_the_local_listing_offers_only_generator_pipelines_as_video(tmp_path):
    def row(path, model_id = None):
        kwargs = {"display_name": path.name, "path": str(path), "model_id": model_id}
        return LocalModelInfo(id = str(path), source = "models_dir", **kwargs)

    def shards(path):
        for name in _SHARDS:
            _write_json(path / name, {})
        return path

    video = cc._VIDEO_GEN_TASK
    assert cc._local_model_task(row(_write_planned_tree(tmp_path / "Cosmos3-Edge", EDGE))) == video
    renamed = _write_configs(tmp_path / "my-render-model", NANO)
    assert cc._local_model_task(row(renamed)) == video
    reasoner = shards(tmp_path / "Cosmos3-Nano-Reasoner")
    assert cc._local_model_task(row(reasoner, "nvidia/Cosmos3-Nano-Reasoner")) is None
    assert cc._local_is_diffusers(row(reasoner)) is False
    assert cc._local_model_task(row(shards(tmp_path / "Cosmos3-Nano-finetune"))) is None
    fp8 = _write_planned_tree(tmp_path / "Cosmos3-Edge-FP8", EDGE)
    assert cc._local_model_task(row(fp8)) is None


def test_no_cosmos3_gguf_or_loose_file_is_offered_as_video(tmp_path):
    from hub.utils import comfy_models

    assert (
        cc._unhydrated_gguf_task(("unsloth/Cosmos3-Nano-GGUF", "Cosmos3-Nano-Q4_K_M.gguf")) is None
    )
    unsupported = cc._UNSUPPORTED_DIFFUSION_TASK
    assert cc._name_hint_media_task(("Cosmos3-Edge-Q8_0.gguf",), unsupported) == unsupported
    for name, offered in (
        ("cosmos3_edge_bf16.safetensors", False),
        ("wan2.1_t2v_1.3B.safetensors", True),
    ):
        (tmp_path / name).write_bytes(b"weights")
        assert comfy_models._offer_loose_checkpoint(tmp_path / name) is offered, name


@pytest.mark.parametrize("policy", ["none", "model", "group", "streaming", "sequential"])
@pytest.mark.parametrize("speed", ["off", "default"])
def test_cosmos3_decodes_untiled_when_it_fits_even_at_speed_off(policy, speed):
    gate = video_mod._decodes_untiled_when_fits
    placed = policy != "sequential"
    for family in ("cosmos3-edge", "cosmos3-nano"):
        assert gate(family, policy, "cuda", speed) is placed
        assert gate(family, policy, "cpu", speed) is False
    assert gate("wan2.2-ti2v-5b", policy, "cuda", speed) is (placed and speed != "off")


def _diffusers_tiles(length, tile, stride):
    """AutoencoderKLWan.tiled_decode's slices along one axis, in latents."""
    return [(a, min(a + tile, length)) for a in range(0, length, stride)]


@pytest.mark.parametrize("family", sorted(vf.COSMOS3_FAMILIES))
def test_seam_safe_tiles_clear_the_wan_seam_floor(family):
    import test_vae_tile_geometry as G

    _, config = G._family_configs()[family]
    vae, _ = G.build_vae(config, None)
    floor = G.SMALLER_GEOMETRY_OK["AutoencoderKLWan@16x"]
    stock = (vae.tile_sample_min_height, vae.tile_sample_stride_height)
    # Every preset side through the stock tiled decoder; 400 px (25 latents) needs a wider tile.
    for side in sorted({*G._sides(family), 400}):
        with vc3.seam_safe_tile_strides(SimpleNamespace(vae = vae), side, side):
            tiles = G.probe_axis(vae, config, side // 16, G.MIN_SIDE // 16, True)
        assert G.check_axis(tiles, floor) == [], (side, tiles)
        assert (vae.tile_sample_min_height, vae.tile_sample_stride_height) == stock
    for length in range(17, 161):
        tile, stride = vc3.seam_safe_tile(length, 16, 12)
        assert G.check_axis(_diffusers_tiles(length, tile, stride), floor) == [], length
    assert vc3.seam_safe_tile(45, 16, 12) == vc3.seam_safe_tile(80, 16, 12) == (16, 12)
    assert vc3.seam_safe_tile(25, 16, 12) == (17, 13)


def test_wide_tiles_keep_the_seam_floor_for_cosmos3():
    import test_vae_tile_geometry as G
    from core.inference import video_vae_untiled as U

    floor, planned = G.SMALLER_GEOMETRY_OK["AutoencoderKLWan@16x"], 0
    for family in sorted(vf.COSMOS3_FAMILIES):
        for width, height in ((640, 640), (832, 480), (480, 832), (1280, 720), (400, 640)):
            for free_gib in (2, 4, 8, 16, 32):
                shape = (1, 48, 31, height // 16, width // 16)
                plan = U.wide_tile_plan(family, shape, 16, free_gib * 2**30)
                if plan is None:
                    continue
                planned += 1
                tile_h, stride_h, tile_w, stride_w = (v // 16 for v in plan)
                for length, tile, stride in (
                    (height // 16, tile_h, stride_h),
                    (width // 16, tile_w, stride_w),
                ):
                    if tile < length:
                        tiles = _diffusers_tiles(length, tile, stride)
                        assert G.check_axis(tiles, floor) == [], (family, width, height, plan)
    assert planned
