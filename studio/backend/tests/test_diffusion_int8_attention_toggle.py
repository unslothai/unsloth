# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The per-load ``int8_attention`` toggle (auto / on / off) and MiniMax-H3 video's raced INT8 attention default (CPU):
precedence of the per-load value over the env switch over the family default, the policy resolved once per load,
``off`` on every route, ``on`` like the request by name, an explicit attention pick kept in Auto, H3 video's own
race sizes and token floor, the H3 install order and status, the H3 sink mode, FlashInfer standing aside, the
resolved record, the deferred speed profile, and old clients that send nothing."""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
import torch

from core.inference import diffusion_attention as A
from core.inference import diffusion_int8_attention as I8
from core.inference import diffusion_int8_attention_dispatch as D

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV = I8.INT8_ATTENTION_ENV


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    # pythonpath puts the test's own tree first: a base arm must not import head code (or the reverse)
    assert Path(I8.__file__).resolve().is_relative_to(BACKEND_DIR)
    assert Path(D.__file__).resolve().is_relative_to(BACKEND_DIR)
    assert Path(A.__file__).resolve().is_relative_to(BACKEND_DIR)
    monkeypatch.delenv(ENV, raising = False)


# ---------------------------------------------------------------- policy and precedence


@pytest.mark.parametrize(
    "env, request_value, expected",
    [
        (None, None, "auto"),
        (None, "auto", "auto"),
        (None, "on", "force"),
        (None, "off", "off"),
        ("0", None, "off"),
        ("0", "auto", "off"),
        ("0", "on", "force"),  # a per-load On beats the env kill switch
        ("0", "off", "off"),
        ("1", None, "force"),
        ("1", "auto", "force"),
        ("1", "on", "force"),
        ("1", "off", "off"),  # a per-load Off beats the env force
    ],
)
def test_precedence_matrix(monkeypatch, env, request_value, expected):
    if env is not None:
        monkeypatch.setenv(ENV, env)
    assert I8.mode(request_value) == expected
    assert I8.resolve_policy(request_value).mode == expected
    # no override: the env switch alone, as before
    assert I8.mode() == {"0": "off", "1": "force", None: "auto"}[env]


def test_request_is_normalised_and_invalid_refused():
    assert I8.normalize_int8_attention(None) is None
    assert I8.normalize_int8_attention("") is None
    assert I8.normalize_int8_attention(" ON ") == "on"
    for bad in ("force", "1", "maybe", 1):
        with pytest.raises(ValueError):
            I8.normalize_int8_attention(bad)


def test_policy_is_resolved_once(monkeypatch):
    policy = I8.resolve_policy(None)
    assert policy.mode == "auto" and policy.source == "default"
    monkeypatch.setenv(ENV, "1")
    # a later env change does not reach a policy already resolved for a load
    assert I8.mode(policy) == "auto"
    assert I8.resolve_policy(policy) is policy
    assert I8.resolve_policy(None).source == "env"
    assert I8.resolve_policy("off").source == "load"


def test_per_load_on_is_not_the_env_force(monkeypatch):
    # the env switch 1 alone skips the Triton stand-in's race; a per-load On races it like the name does
    assert not I8.env_forced("on")
    monkeypatch.setenv(ENV, "1")
    assert I8.env_forced(None)
    assert I8.env_forced("auto")
    assert not I8.env_forced("on")
    assert not I8.env_forced("off")


def test_status_reports_the_load_policy(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    st = I8.status(None, I8.resolve_policy("off"))
    assert st["mode"] == "off" and st["requested"] == "off" and st["source"] == "load"
    assert I8.status()["mode"] == "force"


# ---------------------------------------------------------------- off / on on every route


def _pipe_with_transformer():
    return types.SimpleNamespace(transformer = torch.nn.Linear(2, 2))


def test_off_beats_env_force_and_the_name_on_every_route(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    called = []
    monkeypatch.setattr(I8, "install_int8_attention", lambda *a, **k: called.append(k) or 3)
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda:0", dtype = torch.bfloat16)
    pipe = _pipe_with_transformer()
    assert I8.install_requested(pipe, "unsloth_int8", target, policy = "off") == 0
    assert I8.install_auto(pipe, target, policy = "off") == 0
    assert (
        I8.install_family_auto(
            pipe.transformer, device = "cuda:0", dtype = torch.bfloat16, policy = "off"
        )
        == 0
    )
    assert D._install(pipe, target, "unsloth_int8", None, True, False, "off") == 0
    assert called == []


def test_on_beats_env_off_like_the_request_by_name(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    called = []
    monkeypatch.setattr(I8, "install_int8_attention", lambda t, **k: called.append(k) or 3)
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda:1", dtype = torch.bfloat16)
    pipe = _pipe_with_transformer()
    # env 0 alone: nothing, even by name
    assert I8.install_requested(pipe, "unsloth_int8", target) == 0
    assert I8.install_requested(pipe, None, target, policy = "on") == 3
    assert called[-1]["force"] is True and called[-1]["device"] == "cuda:1"
    assert I8.resolve_policy(called[-1]["policy"]).request == "on"


def test_install_returns_zero_with_off_policy_before_touching_the_card(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    t = torch.nn.Linear(2, 2)
    assert I8._install(t, "cuda:0", torch.bfloat16, None, None, True, None, None, "off") == 0


# ---------------------------------------------------------------- explicit attention pick in Auto


def _route_recorder(monkeypatch, engaged):
    calls = {"dispatch": 0, "auto": 0, "removed": 0}
    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: engaged)
    monkeypatch.setattr(I8, "install_requested", lambda *a, **k: 0)

    def auto(*a, **k):
        calls["auto"] += 1
        return 0

    def dispatch(*a, **k):
        calls["dispatch"] += 1
        calls["policy"] = k.get("policy")
        return 0

    def removed(pipe):
        calls["removed"] += 1
        return 0

    monkeypatch.setattr(I8, "install_auto", auto)
    monkeypatch.setattr(D, "install", dispatch)
    monkeypatch.setattr(D, "remove", removed)
    return calls


@pytest.mark.parametrize(
    "requested, engaged", [("native", None), ("sage", "sage_hub"), ("cudnn", "_native_cudnn")]
)
def test_auto_keeps_an_explicit_attention_pick(monkeypatch, requested, engaged):
    calls = _route_recorder(monkeypatch, engaged)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    out = A.apply_attention_backend(object(), engaged, target = target, requested = requested)
    assert out == engaged
    # neither the strided race nor the dispatch race runs over the user's pick
    assert calls["dispatch"] == 0 and calls["auto"] == 0
    assert calls["removed"] == 1
    # an attention pick left to Studio still races both routes
    A.apply_attention_backend(object(), engaged, target = target, requested = None)
    assert calls["dispatch"] == 1
    assert calls["auto"] == (1 if engaged is None or engaged.startswith("_native") else 0)


@pytest.mark.parametrize("requested, engaged", [("native", None), ("sage", "sage_hub")])
def test_on_layers_the_kernel_over_an_explicit_pick(monkeypatch, requested, engaged):
    calls = _route_recorder(monkeypatch, engaged)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    A.apply_attention_backend(
        object(), engaged, target = target, requested = requested, int8_policy = "on"
    )
    assert calls["dispatch"] == 1
    assert I8.resolve_policy(calls["policy"]).mode == "force"


# ---------------------------------------------------------------- H3 video Auto: its own race


class MiniMaxH3Transformer3DModel(torch.nn.Module):
    pass


class MiniMaxH3ImageTransformer2DModel(torch.nn.Module):
    pass


def test_h3_video_races_at_its_own_lengths_and_h3_image_does_not():
    assert I8.family_race_tokens(MiniMaxH3Transformer3DModel()) == (16384, 8192)
    assert min(I8.FAMILY_RACE_TOKENS["MiniMaxH3Transformer3DModel"]) > max(I8.RACE_TOKENS)
    assert I8.family_race_tokens(MiniMaxH3ImageTransformer2DModel()) is None
    assert "MiniMaxH3Transformer3DModel" not in D.AUTO_DITS
    assert "MiniMaxH3ImageTransformer2DModel" not in D.AUTO_DITS
    # the sink path waits on its own measurement
    assert "MiniMaxH3Transformer3DModel" not in I8.SINK_DITS


def _family_floor(
    monkeypatch,
    wins: dict,
    impl: str = "cuda",
    ok: bool = True,
):
    """Run install_family_auto with the card check and the race faked; returns (floor result, raced sizes)."""
    raced = []
    monkeypatch.setattr(I8, "_is_rocm", lambda: False)
    monkeypatch.setattr(I8, "implementation", lambda index, hd: impl)
    monkeypatch.setattr(
        I8,
        "verdict",
        lambda dev, dt, heads, hd, fn, name, timed = True: {"ok": ok, "impl": impl, "timed": timed},
    )

    def race(device, dtype, heads, head_dim, incumbent, name, tokens):
        raced.append(tokens)
        return wins.get(tokens, False)

    monkeypatch.setattr(D, "_race_wins", race)
    got = {}

    def fake_install(t, **k):
        got["kwargs"] = k
        got["floor"] = k["floor"](
            torch.device("cuda", 0), torch.bfloat16, 24, 128, None, "_native_flash"
        )
        return 0 if got["floor"][1] is None else 40

    monkeypatch.setattr(I8, "install_int8_attention", fake_install)
    n = I8.install_family_auto(
        MiniMaxH3Transformer3DModel(), device = "cuda:2", dtype = torch.bfloat16, policy = None
    )
    return n, got, raced


def test_h3_auto_engages_where_it_wins_the_long_race(monkeypatch):
    n, got, raced = _family_floor(monkeypatch, {16384: True, 8192: False})
    assert n == 40
    # raced at H3's lengths, never at the 4096-token card race that rejects A100
    assert raced == [16384, 8192]
    found, floor = got["floor"]
    assert found["timed"] is False  # the accuracy check alone, untimed
    # the floor keeps the token refiner (20-200 tokens) on the incumbent
    assert floor >= D.MIN_TOKENS and floor == (16384 + 8192) // 2
    assert got["kwargs"]["device"] == "cuda:2"
    assert "force" not in got["kwargs"] or got["kwargs"]["force"] is False


def test_h3_auto_floor_when_it_wins_both_sizes(monkeypatch):
    n, got, _ = _family_floor(monkeypatch, {16384: True, 8192: True})
    assert n == 40 and got["floor"][1] >= D.MIN_TOKENS


def test_h3_auto_keeps_the_incumbent_where_it_loses(monkeypatch):
    n, got, raced = _family_floor(monkeypatch, {})
    assert n == 0 and got["floor"][1] is None and raced == [16384]


def test_h3_auto_needs_the_accuracy_check(monkeypatch):
    n, got, raced = _family_floor(monkeypatch, {16384: True, 8192: True}, ok = False)
    assert n == 0 and raced == []


def test_h3_auto_cuda_kernel_only(monkeypatch):
    n, got, raced = _family_floor(monkeypatch, {16384: True, 8192: True}, impl = "triton")
    assert n == 0 and raced == [] and got["floor"][0]["error"] == "CUDA kernel only"


def test_h3_auto_does_not_run_under_force_or_off(monkeypatch):
    monkeypatch.setattr(I8, "install_int8_attention", lambda *a, **k: pytest.fail("installed"))
    for policy in ("on", "off"):
        assert (
            I8.install_family_auto(
                MiniMaxH3Transformer3DModel(), device = "cuda:0", dtype = torch.bfloat16, policy = policy
            )
            == 0
        )


def test_long_verdict_takes_family_sizes(monkeypatch):
    raced = []
    monkeypatch.setattr(D, "_race_wins", lambda *a: raced.append(a[-1]) or True)
    D.long_verdict("cuda:0", torch.bfloat16, 2, 128, None, "x", tokens = (32768, 12288))
    assert raced == [32768, 12288]
    raced.clear()
    D.long_verdict("cuda:0", torch.bfloat16, 2, 128, None, "x")
    assert raced == list(D.LONG_RACE_TOKENS)


# ---------------------------------------------------------------- H3 video load block (source order)


def _h3_block() -> str:
    src = (BACKEND_DIR / "core" / "inference" / "video.py").read_text()
    start = src.index("h3_attn_backend = select_attention_backend(")
    end = src.index("from .video_minimax_h3_flashinfer import install_h3_flashinfer", start)
    return src[start:end]


def test_h3_block_installs_strided_then_int8_then_reports_it():
    block = _h3_block()
    # the stock processors' routes never take the kernel first (that would cost the fused q/k norm + RoPE)
    assert 'int8_policy = int8_attention.Int8Policy(request = "off"' in block
    strided = block.index("install_strided_attention(")
    assert strided < block.index("int8_attention.install_int8_attention(")
    assert strided < block.index("int8_attention.install_family_auto(")
    # status names the kernel when it engaged
    assert "attention_engaged = int8_attention.BACKEND" in block
    # the probe keys on the load's real card
    assert "device = h3_torch_device" in block
    assert 'getattr(umem_target, "torch_device", None)' in block


def test_h3_block_auto_is_gated_to_the_evidenced_configuration():
    block = _h3_block()
    auto = block[block.index("int8_auto = (") : block.index("h3_torch_device =")]
    for gate in (
        'int8_mode == "auto"',
        "effective_speed != SPEED_OFF",
        "_auto_request(attention_backend)",
        'transformer_quant_engaged == "int8"',
        "not comfy_checkpoint",
        "not baked_loras",
    ):
        assert gate in auto, gate


def test_h3_load_status_carries_int8_attention():
    src = (BACKEND_DIR / "core" / "inference" / "video.py").read_text()
    h3 = src[src.index("def _load_h3_modular_pipeline(") :]
    h3 = h3[: h3.index("\n    def ", 10)]
    assert '"int8_attention": int8_resolved_entry(' in h3
    assert "int8_annotate_resolved(resolved, int8_policy, speed_view)" in h3
    assert "int8_policy = int8_attention_policy(int8_policy)" in h3


# ---------------------------------------------------------------- H3 strided processor sink mode, FlashInfer


def test_h3_strided_processor_forwards_the_sink_mode(monkeypatch):
    h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    from core.inference import video_minimax_h3_attn as HA

    torch.manual_seed(0)
    model = h3.MiniMaxH3Transformer3DModel(
        num_attention_heads = 2,
        attention_head_dim = 32,
        hidden_size = 64,
        num_layers = 1,
        num_refiner_layers = 1,
        ffn_dim = 128,
        in_channels = 4,
        audio_in_channels = 4,
        patch_size = (1, 2, 2),
        text_dim = 32,
        freq_dim = 32,
        time_embed_hidden_dim = 64,
        time_embed_dim = 32,
        rope_freq_dim = 4,
    ).eval()
    assert HA.install_strided_attention(model, qk_rope = False) > 0
    attn = next(
        m for m in model.modules() if hasattr(type(getattr(m, "processor", None)), "_unsloth_int8")
    )
    proc = attn.processor
    assert proc._unsloth_int8_sinks == 0
    modes = []

    def call(
        q,
        k,
        v,
        mode = 0,
    ):
        modes.append(mode)
        return v.clone()

    monkeypatch.setattr(HA._i8, "int8_attention_call", call)
    proc._unsloth_int8 = True
    proc._unsloth_int8_min_tokens = 0
    hidden = torch.randn(1, 8, 64)
    with torch.no_grad():
        proc(attn, hidden)
        proc._unsloth_int8_sinks = I8.MODE_RACE
        proc(attn, hidden)
    assert modes == [0, I8.MODE_RACE]


def test_flashinfer_stands_aside_for_an_engaged_int8_processor(monkeypatch):
    from core.inference import video_minimax_h3_flashinfer as fi

    monkeypatch.setenv(fi.MASTER_ENV, "1")
    monkeypatch.setattr(fi, "flashinfer_unavailable", lambda device = None: None)
    monkeypatch.setattr(fi, "_scheme_of", lambda w: "fp8")
    monkeypatch.setattr(fi, "register_ops", lambda: True)
    monkeypatch.setattr(fi, "_pack", lambda *a, **k: pytest.fail("packed over the INT8 kernel"))
    blocks = []
    for _ in range(2):
        blk = torch.nn.Module()
        blk.attn = torch.nn.Module()
        blk.attn.to_q = torch.nn.Linear(8, 8, bias = False)
        blk.attn.processor = types.SimpleNamespace(
            _attention_backend = "_native_cudnn", _unsloth_int8 = True
        )
        blocks.append(blk)
    out = fi.install_h3_flashinfer(types.SimpleNamespace(transformer_blocks = blocks))
    assert out["blocks"] == 0 and "INT8 attention" in out["reason"]


# ---------------------------------------------------------------- resolved record


class _Proc:
    _unsloth_int8 = False

    def __init__(
        self,
        on: bool,
        impl: str = "cuda",
    ):
        self._unsloth_int8 = on
        self._unsloth_int8_impl = impl


class _Attn(torch.nn.Module):
    def __init__(self, on: bool):
        super().__init__()
        self.heads = 2
        self.processor = _Proc(on)


def _pipe(n_on: int):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attn(i < n_on) for i in range(3)])
    return types.SimpleNamespace(transformer = dit)


def _record(policy, pipe, **kw):
    from core.inference.diffusion_auto_policy import build_resolved_record
    rec = build_resolved_record({"int8_attention": I8.resolved_entry(policy, pipe, **kw)})
    return I8.annotate_resolved(rec, policy, pipe)["int8_attention"]


def test_resolved_entry_auto_engaged():
    e = _record(None, _pipe(2))
    assert e["value"] == "on" and e["source"] == "auto" and e["status"] == "applied"
    assert e["engaged"] is True and e["modules"] == 2 and e["impl"] == "cuda"
    assert "unsloth_int8" in e["reason"]


def test_resolved_entry_explicit_on_declined_reads_fell_back():
    e = _record("on", _pipe(0))
    assert e["value"] == "off" and e["requested"] == "on" and e["status"] == "fell_back"
    assert e["engaged"] is False and e["impl"] is None
    e = _record("on", _pipe(0), supported = False)
    assert e["status"] == "unsupported"


def test_resolved_entry_explicit_off():
    e = _record("off", _pipe(0))
    assert e["value"] == "off" and e["status"] == "applied" and e["mode"] == "off"


def test_resolved_entry_counts_a_list_of_views():
    e = _record(None, [_pipe(1), _pipe(2)])
    assert e["modules"] == 3


# ---------------------------------------------------------------- load plumbing: requests, begin_load, deferred profile


def test_request_models_carry_the_field_and_old_clients_send_nothing():
    from pydantic import ValidationError

    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    for model in (DiffusionLoadRequest, VideoLoadRequest):
        assert model(model_path = "x/y").int8_attention is None
        assert model(model_path = "x/y", int8_attention = " Off ").int8_attention == "off"
        assert model(model_path = "x/y", int8_attention = "").int8_attention is None
        with pytest.raises(ValidationError):
            model(model_path = "x/y", int8_attention = "force")


def test_every_begin_load_accepts_it():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend
    from core.inference.video import VideoBackend

    for cls in (DiffusionBackend, VideoBackend, SdCppDiffusionBackend):
        params = inspect.signature(cls.begin_load).parameters
        assert "int8_attention" in params and params["int8_attention"].default is None, cls
    from core.inference.mlx_diffusion_backend import MlxDiffusionBackend

    kinds = {p.kind for p in inspect.signature(MlxDiffusionBackend.begin_load).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD in kinds


def test_routes_forward_it_only_when_set():
    for rel in ("routes/inference.py", "routes/video.py"):
        src = (BACKEND_DIR / rel).read_text()
        assert (
            '**({"int8_attention": request.int8_attention} if request.int8_attention else {})'
            in src
        ), rel


def test_deferred_profile_keeps_the_load_policy():
    import dataclasses

    from core.inference import diffusion as Dm

    assert "int8_policy" in {f.name for f in dataclasses.fields(Dm._LoadState)}
    src = inspect.getsource(Dm.DiffusionBackend._engage_deferred_profile)
    assert 'int8_policy = getattr(state, "int8_policy", None)' in src
    load_src = inspect.getsource(Dm.DiffusionBackend.load_pipeline)
    assert "int8_policy = int8_policy" in load_src


def test_video_state_keeps_the_load_policy():
    import dataclasses

    from core.inference import video as V
    assert "int8_policy" in {f.name for f in dataclasses.fields(V._VideoLoadState)}
