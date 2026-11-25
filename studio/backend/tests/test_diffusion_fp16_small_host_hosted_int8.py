# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AUTO on an fp16-only small host reads the hosted int8 ConvRot DiT instead of quantising per row at load."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_fp16_hosted_int8 as FH  # noqa: E402
from core.inference import diffusion_small_host as SH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

BASE = "Qwen/Qwen-Image-2.1"
T4 = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
L4 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
# Colab T4 standard (12.7 GB) and high-RAM (51 GB) hosts, as the loader read them.
STANDARD = (12976, 11427)
HIGH_RAM = (52223, 50118)


def _fam():
    fam = detect_family(BASE, override = "qwen-image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    return fam


@pytest.fixture
def host(monkeypatch):
    for env in (
        FH.FP16_CARD_HOSTED_INT8_ENV,
        SH.SMALL_HOST_ENV,
        SH.HOST_RAM_LIMIT_ENV,
        SH.HOST_RAM_CHECK_ENV,
        "UNSLOTH_NATIVE_INT8_ACT",
        "UNSLOTH_NATIVE_INT8_ROT",
        "UNSLOTH_DIFFUSION_INT8_CONVROT",
        "UNSLOTH_DIFFUSION_FP16_CARD_INT8",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (7, 5))
    monkeypatch.setattr(SH, "resolve_snapshot_dir", lambda *a, **k: None)
    monkeypatch.setattr(SH, "host_commit_mib", lambda: None)
    monkeypatch.setattr(FH, "fp16_card_hosted_int8_cached", lambda *a, **k: "/hub/x.safetensors")
    ram = {"now": STANDARD}
    monkeypatch.setattr(SH, "host_ram_mib", lambda: ram["now"])
    return ram


def test_small_host_auto_reads_the_hosted_build(host):
    why = FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE, require_cached = True)
    assert why is not None and "11427 MiB available" in why


def test_a_host_the_dense_dit_fits_keeps_auto_unchanged(host):
    host["now"] = HIGH_RAM
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE) is None


def test_unreadable_host_ram_keeps_auto_unchanged(host):
    host["now"] = (None, None)
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE) is None


def test_bf16_cards_are_untouched(host):
    assert FH.small_host_hosted_int8_reason(_fam(), L4, base = BASE) is None


@pytest.mark.parametrize(
    "base", ["/models/Qwen-Image-2.1", "someone/Qwen-Image-2.1-finetune", None]
)
def test_only_the_released_repo(host, base):
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = base) is None


def test_lora_keeps_the_dense_route(host):
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE, has_lora = True) is None


def test_load_needs_the_file_in_the_cache(host, monkeypatch):
    monkeypatch.setattr(FH, "fp16_card_hosted_int8_cached", lambda *a, **k: None)
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE, require_cached = True) is None
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE) is not None


@pytest.mark.parametrize(
    "env",
    [
        (FH.FP16_CARD_HOSTED_INT8_ENV, "0"),
        (SH.SMALL_HOST_ENV, "0"),
        ("UNSLOTH_DIFFUSION_FP16_CARD_INT8", "0"),
    ],
)
def test_kill_switches(host, monkeypatch, env):
    monkeypatch.setenv(*env)
    assert FH.small_host_hosted_int8_reason(_fam(), T4, base = BASE) is None


def test_the_download_plan_stages_the_hosted_file_on_a_small_host(host):
    planned = FH.fp16_card_hosted_int8_planned_scheme(_fam(), T4, None, kind = "pipeline", base = BASE)
    assert planned == "int8"
    host["now"] = HIGH_RAM
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(_fam(), T4, None, kind = "pipeline", base = BASE)
        is None
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"speed_mode": "off"},
        {"forces_offload": True},
        {"has_lora": True},
    ],
)
def test_the_download_plan_keeps_the_dense_shards_where_auto_does(host, kw):
    assert (
        FH.fp16_card_hosted_int8_planned_scheme(_fam(), T4, None, kind = "pipeline", base = BASE, **kw)
        is None
    )


def test_snapshot_pricing_matches_the_small_host_decision(host, monkeypatch, tmp_path):
    comp = SH.StoredComponent("transformer", 13030, "bfloat16")
    monkeypatch.setattr(SH, "resolve_snapshot_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(SH, "stored_components", lambda snap: {"transformer": comp})
    monkeypatch.setattr(SH, "snapshot_weights_complete", lambda *a, **k: True)
    fits, why = FH._dense_dit_fits_host(_fam(), BASE, None)
    decision = SH.decide_small_host(
        {"transformer": comp},
        torch.float16,
        device = "cuda",
        host_total_mib = STANDARD[0],
        host_available_mib = STANDARD[1],
    )
    assert fits is False and decision.engaged
    assert f"~{decision.dense_host_mib} MiB (snapshot)" in why


def test_the_loader_consults_the_small_host_reason():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    start = src.index("turing_auto_int8 = (")
    block = src[start : src.index("if turing_auto_int8 is not None:", start)]
    assert "turing_auto_int8 = turing_auto_int8 or small_host_hosted" in block
    assert "small_host_hosted_int8_reason(" in block
    assert "require_cached = True" in block
    assert "has_lora = _has_active_lora(loras)" in block


def test_the_hosted_dit_is_marked_and_placed_as_the_small_host_routes():
    """Without the mark a 12.7 GB T4 streamed the hosted DiT unprefetched at ~14 s/step vs 4 s."""
    from core.inference import diffusion as D
    from core.inference.diffusion_memory import pipe_small_host_int8

    pipe = types.SimpleNamespace()
    seen = {}

    class Backend:
        def _small_host_plan(self, pipe, plan, fam, logger):
            seen["info"] = SH.engaged_on(pipe)
            return "placed"

    log = types.SimpleNamespace(info = lambda *a, **k: None)
    got = D.DiffusionBackend._mark_small_host_hosted(Backend(), pipe, "plan", "why", None, log)
    assert got == "placed"
    assert seen["info"]["components"]["transformer"] == D.SMALL_HOST_HOSTED_INT8_COMPONENT
    assert D._small_host_int8(pipe) and not pipe_small_host_int8(pipe)
    assert D._lora_load_block_reason(pipe) is not None


def test_a_route_that_engaged_for_the_encoders_keeps_its_components():
    from core.inference import diffusion as D

    pipe = types.SimpleNamespace()
    SH.mark(pipe, {"reason": "r", "components": {"vae": "converted"}})

    class Backend:
        def _small_host_plan(self, pipe, plan, fam, logger):
            return plan

    D.DiffusionBackend._mark_small_host_hosted(
        Backend(), pipe, "plan", "why", None, types.SimpleNamespace(info = lambda *a, **k: None)
    )
    info = SH.engaged_on(pipe)
    assert info["reason"] == "r" and info["components"]["vae"] == "converted"
    assert info["components"]["transformer"] == D.SMALL_HOST_HOSTED_INT8_COMPONENT


def test_the_loader_marks_the_hosted_dit_after_the_pipeline_is_built():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    call = src.index("plan = self._mark_small_host_hosted(")
    guard = src.rindex("small_host_hosted is not None", 0, call)
    assert "hosted_native_scheme is not None" in src[guard:call]
    assert src.rindex("plan = self._apply_small_host_route(", 0, call) < guard


def _hosted_read_statements():
    """The loader's statements from the hosted read up to the skipped-shard restore, as written in diffusion.py."""
    import ast

    path = BACKEND / "core" / "inference" / "diffusion.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    bodies = (
        getattr(node, field, None)
        for node in ast.walk(tree)
        for field in ("body", "orelse", "finalbody")
    )
    for body in bodies:
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if (
                isinstance(stmt, ast.If)
                and "fp16_card_hosted_int8_reason(" in ast.unparse(stmt.test)
                and "fp16_card_hosted_int8_transformer(" in ast.unparse(stmt)
            ):
                end = next(
                    j
                    for j in range(i + 1, len(body))
                    if isinstance(body[j], ast.If)
                    and "_pipeline_prequant_skipped" in ast.unparse(body[j].test)
                    and "hosted_native_scheme is None" in ast.unparse(body[j].test)
                )
                return ast.Module(body = body[i:end], type_ignores = []), str(path)
    raise AssertionError("hosted read not found in the loader")


@pytest.mark.parametrize("read", ["declined", "read"])
def test_a_declined_hosted_read_returns_to_the_auto_per_row_route(read):
    """A declined hosted read restores AUTO: no pinned int8, so the per-row build is not re-armed."""
    module, path = _hosted_read_statements()
    logged = []
    hosted = types.SimpleNamespace(_unsloth_prequant_artifact = "prequant:x")
    ns = dict(
        pipeline_seed_scheme = None,
        _has_active_lora = lambda loras: False,
        loras = None,
        _pipeline_prequant_skipped = (),
        _fp16_hosted_planned = False,
        pipe_kwargs = {},
        fp16_card_hosted_int8_reason = lambda *a: "hosted",
        fp16_card_hosted_int8_transformer = lambda *a, **k: None if read == "declined" else hosted,
        fam = None,
        target = T4,
        transformer_cls = object,
        base = BASE,
        _base_local_dir = None,
        fetch_base = BASE,
        hf_token = None,
        hub_cache_dir = lambda: None,
        logger = types.SimpleNamespace(info = lambda *a, **k: logged.append(a)),
        TQ_AUTO = tq.TQ_AUTO,
        TQ_INT8 = tq.TQ_INT8,
        transformer_quant = tq.TQ_INT8,
        transformer_quant_pinned = tq.TQ_INT8,
        native_scheme = tq.TQ_INT8,
        turing_auto_int8 = "small host",
        small_host_hosted = "small host",
        hosted_native_scheme = None,
        transformer_quant_artifact = None,
    )
    exec(compile(module, path, "exec"), ns)
    if read == "declined":
        assert ns["transformer_quant"] == tq.TQ_AUTO
        assert ns["transformer_quant_pinned"] is None and ns["native_scheme"] is None
        assert ns["turing_auto_int8"] is None and ns["small_host_hosted"] is None
        assert "transformer" not in ns["pipe_kwargs"]
        assert any("was not read" in str(a[0]) for a in logged)
    else:
        assert ns["pipe_kwargs"]["transformer"] is hosted
        assert ns["native_scheme"] == tq.TQ_INT8 and ns["hosted_native_scheme"] == tq.TQ_INT8
        assert ns["small_host_hosted"] == "small host"
