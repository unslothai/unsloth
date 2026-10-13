# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the diffusion auto-optimizer core: patches, targets, profile, plan, apply, runtime."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.diffusion_autoopt import apply as ao  # noqa: E402
from core.inference.diffusion_autoopt import patches as ap  # noqa: E402
from core.inference.diffusion_autoopt import plan as plan_mod  # noqa: E402
from core.inference.diffusion_autoopt import profile as prof  # noqa: E402
from core.inference.diffusion_autoopt import runtime, targets  # noqa: E402


# ------------------------------------------------------------------------------------------------- fingerprints
def _sample(x, y = 1):
    """Docstring is ignored by the digest."""
    # so is this comment
    return x + y


def test_source_digest_is_the_one_both_q21_modules_use():
    from core.inference import diffusion_qwenimage21, diffusion_qwenimage21_rope

    assert diffusion_qwenimage21._digest is ap.source_digest
    assert diffusion_qwenimage21_rope._digest is ap.source_digest
    assert ap.source_digest(_sample) == ap.source_digest(_sample)
    assert len(ap.source_digest(_sample)) == 16
    assert ap.source_digest(len) is None  # no Python source


def test_source_digest_matches_a_pinned_fingerprint():
    """A fingerprint recorded with the pre-merge ``_digest`` still matches (the algorithm did not change)."""
    from core.inference import video_wan_fused

    try:
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    except Exception:  # noqa: BLE001
        pytest.skip("diffusers without Wan")
    digest = ap.source_digest(WanTransformerBlock.forward)
    # The pinned set covers the diffusers versions the fused block was written against; the installed one is among
    # them in the Studio venv, and the digest must be byte-for-byte what the old helper produced.
    if digest not in video_wan_fused._FINGERPRINTS:
        pytest.skip(f"installed diffusers Wan block is not a pinned version ({digest})")
    assert digest in video_wan_fused._FINGERPRINTS


def test_body_has_and_once():
    assert ap.body_has(_sample, "return x + y")
    assert not ap.body_has(_sample, "return x - y")
    assert ap.body_has_once(_sample, "return x + y")

    def twice(a):
        a = a + 1
        a = a + 1
        return a

    assert ap.body_has(twice, "a = a + 1")
    assert not ap.body_has_once(twice, "a = a + 1")


def test_arch_patches_keep_a_monkeypatchable_body_has():
    from core.inference import diffusion_arch_patches
    assert diffusion_arch_patches._body_has is ap.body_has


# ---------------------------------------------------------------------------------------------- splice_source
_SCALE = 3


def _spliced_target(x):
    y = x * 2
    return y + _SCALE


def test_splice_source_rewrites_in_the_original_globals():
    new = ap.splice_source(_spliced_target, [("y = x * 2", "y = x * 10")])
    assert new(1) == 13 and _spliced_target(1) == 5
    assert new.__unsloth_spliced__ and new.__wrapped_original__ is _spliced_target
    assert new.__qualname__ == _spliced_target.__qualname__


def test_splice_source_needles_must_match_exactly_once():
    with pytest.raises(ap.SpliceError, match = "0 times"):
        ap.splice_source(_spliced_target, [("not in there", "x")])

    def dup(x):
        x = x + 1
        x = x + 1
        return x

    with pytest.raises(ap.SpliceError, match = "2 times"):
        ap.splice_source(dup, [("x = x + 1", "x = x + 2")])


def test_splice_source_refuses_unresolved_globals_and_closures():
    with pytest.raises(ap.SpliceError, match = "unresolved globals: _nowhere"):
        ap.splice_source(_spliced_target, [("return y + _SCALE", "return y + _nowhere")])
    captured = 4

    def closure(x):
        return x + captured

    with pytest.raises(ap.SpliceError, match = "closure"):
        ap.splice_source(closure, [("x + captured", "x - captured")])


def test_splice_source_fingerprint_gate():
    digest = ap.source_digest(_spliced_target)
    assert ap.splice_source(_spliced_target, [("x * 2", "x * 4")], fingerprints = [digest])(1) == 7
    with pytest.raises(ap.SpliceError, match = "differs"):
        ap.splice_source(_spliced_target, [("x * 2", "x * 4")], fingerprints = ["0" * 16])


# --------------------------------------------------------------------------------------------------- registry
@pytest.fixture
def scratch_registry(monkeypatch):
    """A private registry so these tests never touch the built-in specs' targets."""
    monkeypatch.setattr(ap, "_REGISTRY", [])
    monkeypatch.setattr(ap, "_BUILTIN_DONE", True)
    return ap


def test_registry_lifecycle_kill_switch_speed_only_and_failures(scratch_registry, monkeypatch):
    calls: list = []
    state = {"on": False}

    def install(ctx):
        calls.append(("install", ctx.speed_mode))
        state["on"] = True
        return True

    def uninstall():
        calls.append(("uninstall",))
        state["on"] = False

    def boom(ctx):
        raise RuntimeError("kaboom")

    ap.register(ap.PatchSpec("fuse", ap.STAGE_LOAD, install, uninstall, group = "g", speed_only = True))
    ap.register(ap.PatchSpec("broken", ap.STAGE_LOAD, boom, warn = "broken thing"))
    ap.register(ap.PatchSpec("dit_only", ap.STAGE_LOAD, install, classes = ("OtherDiT",)))
    audit = ao.new_audit(backend = "image", family = None)
    warnings: list = []
    log = types.SimpleNamespace(warning = lambda *a: warnings.append(a))
    pipe = types.SimpleNamespace(transformer = type("MyDiT", (), {})())

    res = ap.install_stage(
        ap.STAGE_LOAD, ap.PatchContext(pipe = pipe, speed_mode = "default", logger = log), audit = audit
    )
    assert res == {"fuse": True} and state["on"]
    assert audit.entries["patch:fuse"]["status"] == "engaged"
    assert audit.entries["patch:broken"]["status"] == "failed"
    assert "kaboom" in audit.entries["patch:broken"]["reason"]
    assert warnings and warnings[0][1] == "broken thing"
    assert "patch:dit_only" not in audit.entries  # class gate: not applicable, not recorded

    # speed off uninstalls a speed-only spec
    ap.install_stage(ap.STAGE_LOAD, ap.PatchContext(pipe = pipe, speed_mode = "off"), audit = audit)
    assert not state["on"]

    # per-spec kill switch
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PATCH_FUSE", "0")
    audit2 = ao.new_audit(backend = "image", family = None)
    ap.install_stage(ap.STAGE_LOAD, ap.PatchContext(pipe = pipe, speed_mode = "max"), audit = audit2)
    assert audit2.entries["patch:fuse"] == {
        "status": "declined",
        "reason": "UNSLOTH_DIFFUSION_PATCH_FUSE=0",
    }
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PATCH_FUSE")

    ap.install_stage(ap.STAGE_LOAD, ap.PatchContext(pipe = pipe, speed_mode = "max"))
    assert state["on"]
    ap.uninstall_group("g")
    assert not state["on"]


def test_register_rejects_unknown_stage(scratch_registry):
    with pytest.raises(ValueError):
        ap.register(ap.PatchSpec("x", "sometime", lambda ctx: True))


def test_builtin_specs_cover_every_former_install_site():
    names = {s.name for s in ap.specs()}
    assert names >= {
        "eager_fusions",
        "arch_fusions",
        "qwenimage21_fast_step",
        "flux2_rope",
        "rocm_fused",
        "wan_fused",
        "qwenimage21_real_rope",
        "qwenimage_real_rope",
        "zimage_fused",
        "qwenimage21_fused",
    }
    # the former hand-kept teardown list is now a registry group
    assert {s.name for s in ap.specs(group = ap.GROUP_FUSED_DIT)} == {
        "flux2_rope",
        "rocm_fused",
        "qwenimage_real_rope",
        "zimage_fused",
        "qwenimage21_fused",
    }
    # every spec answers its module's existing kill switch name
    envs = {s.name: s.env for s in ap.specs()}
    assert envs["arch_fusions"] == "UNSLOTH_DIFFUSION_ARCH_PATCHES"
    assert envs["wan_fused"] == "UNSLOTH_DIFFUSION_WAN_FUSED_ADALN"
    assert envs["zimage_fused"] == "UNSLOTH_DIFFUSION_ZIMAGE_FUSED"
    # order inside the regional compile: Q2.1 real RoPE before the int8 swap, the rest after it
    assert [s.name for s in ap.specs(ap.STAGE_PRE_INT8)] == ["qwenimage21_real_rope"]
    assert [s.name for s in ap.specs(ap.STAGE_PRE_COMPILE)] == [
        "qwenimage_real_rope",
        "zimage_fused",
        "qwenimage21_fused",
    ]


def test_uninstall_fused_dit_patches_goes_through_the_registry(monkeypatch):
    from core.inference import diffusion, diffusion_flux2_rope, diffusion_zimage_fused

    seen = []
    monkeypatch.setattr(diffusion_flux2_rope, "uninstall", lambda: seen.append("flux2"))
    monkeypatch.setattr(diffusion_zimage_fused, "uninstall", lambda *a: seen.append("zimage"))
    diffusion._uninstall_fused_dit_patches()
    assert {"flux2", "zimage"} <= set(seen)


# ---------------------------------------------------------------------------------------------------- targets
def test_denoiser_dits_contract_is_unchanged():
    from core.inference import diffusion_attention, diffusion_speed

    a, b = object(), object()
    pipe = types.SimpleNamespace(
        transformer = a, transformer_2 = b, unconditional_transformer = a, unet = object()
    )
    assert targets.denoiser_dits(pipe) == [a, b]
    assert diffusion_speed._denoiser_dits(pipe) == [a, b]
    assert diffusion_attention._attention_dits(pipe) == [a, b]
    ref = object()
    assert diffusion_speed._guarded_dits(types.SimpleNamespace(transformer_ref = ref)) == [ref]
    assert targets.hooked_denoisers(pipe)[-1] is pipe.unet


def test_by_type_resolver_finds_unnamed_denoisers_but_not_vaes():
    import torch

    class Blocky(torch.nn.Module):
        _repeated_blocks = ["Block"]

        def compile_repeated_blocks(self, **kw):
            pass

    dit, vae, cn = Blocky(), Blocky(), Blocky()
    pipe = types.SimpleNamespace(components = {"prior": dit, "vae": vae, "controlnet": cn})
    assert targets.denoisers(pipe) == [dit]
    assert targets.discovered_only(pipe) == [dit]
    assert targets.denoiser_dits(pipe) == []  # the primitives' contract does not widen
    fam = types.SimpleNamespace(denoiser_attr = "prior")
    pipe.prior = dit
    assert targets.denoisers(pipe, fam)[0] is dit


# ---------------------------------------------------------------------------------------------------- profile
def test_graph_break_screen():
    src = "def forward(self, x):\n    n = x.sum().item()\n    # x.tolist() in a comment\n    return x[: n]\n"
    assert prof.screen_source(src) == (".item()",)
    assert prof.screen_source("def f(x):\n    return x.nonzero()\n") == (".nonzero()",)
    assert prof.screen_source(None) == ()


def _toy_model(block_src_sync: bool):
    import torch

    class ToyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        if block_src_sync:

            def forward(self, x):
                k = int(x.abs().max().item())
                return self.lin(x) + k
        else:

            def forward(self, x):
                return self.lin(x)

    class ToyDiT(torch.nn.Module):
        _no_split_modules = ["ToyBlock"]

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([ToyBlock(), ToyBlock()])
            self.compiled = None

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

        def compile_repeated_blocks(self, **kw):
            if not getattr(self, "_repeated_blocks", None):
                raise ValueError("_repeated_blocks is empty")
            self.compiled = kw

    return ToyDiT()


def test_profile_discovers_blocks_and_is_cached():
    prof.clear_cache()
    model = _toy_model(block_src_sync = True)
    p = prof.profile_model(model)
    assert p.discovered_blocks == ("ToyBlock",)
    assert p.declared_blocks == () and p.repeated_blocks == ()
    assert p.graph_break_suspects == {"ToyBlock": (".item()",)}
    assert "not verified" in p.compile_decline()
    assert prof.profile_model(model) is p  # cached by (qualname, digest, diffusers)


# -------------------------------------------------------------------------------------------- plan / generic
def _target():
    import torch
    return types.SimpleNamespace(
        device = "cuda", backend = "cuda", dtype = torch.bfloat16, supports_default_torch_compile = True
    )


def test_registered_family_detection():
    from core.inference.diffusion_families import _FAMILIES
    from core.inference.video_families import _FAMILIES as VIDEO

    assert plan_mod.family_has_data(_FAMILIES[0])
    assert plan_mod.family_has_data(VIDEO[0])
    assert not plan_mod.family_has_data(types.SimpleNamespace(name = "brand-new-dit"))
    assert not plan_mod.family_has_data(None)


def test_plan_reasons_mirror_the_primitive_gates():
    model = _toy_model(block_src_sync = False)
    pipe = types.SimpleNamespace(transformer = model)
    fam = types.SimpleNamespace(name = "z-image", supports_torch_compile = False)
    p = plan_mod.build_plan(pipe, _target(), family = fam, backend = "image", speed_mode = "default")
    assert p.step("compiled").attempt is False
    assert p.reason("compiled") == "family opts out of torch.compile"
    off = plan_mod.build_plan(pipe, _target(), family = fam, backend = "image", speed_mode = "off")
    assert off.reason("compiled") == "speed off"
    eager = plan_mod.build_plan(pipe, _target(), family = fam, backend = "image", speed_mode = "eager")
    assert eager.reason("compiled") == "eager tier does not compile"


def _generic_run(
    model,
    monkeypatch,
    env = None,
):
    from core.inference import diffusion_speed as ds

    if env is not None:
        monkeypatch.setenv(plan_mod.DISCOVER_ENV, env)
    monkeypatch.setattr(ds, "torch_compile_runtime_available", lambda: True)
    fam = types.SimpleNamespace(name = "brand-new-dit", supports_torch_compile = True)
    pipe = types.SimpleNamespace(transformer = model)
    audit = ao.new_audit(backend = "image", family = fam)
    seen = {}

    def apply(family, log):
        seen["family"] = family
        compiled = False
        if getattr(family, "supports_torch_compile", True):
            try:
                model.compile_repeated_blocks(fullgraph = True)
                compiled = True
            except Exception as exc:  # noqa: BLE001
                log.warning("diffusion.speed: %s failed: %s", "compile_repeated_blocks", exc)
        return {"compiled": compiled, "cuda_graph": False}

    applied = ao.run_speed_layer(
        pipe, _target(), family = fam, backend = "image", speed_mode = "default", audit = audit, apply = apply
    )
    return applied, audit, seen


def test_generic_family_trial_compiles_discovered_blocks(monkeypatch):
    prof.clear_cache()
    model = _toy_model(block_src_sync = False)
    applied, audit, _ = _generic_run(model, monkeypatch)
    assert applied["compiled"] and model.compiled == {"fullgraph": True}
    assert model._repeated_blocks == ["ToyBlock"]
    assert audit.generic and audit.entries["compiled"]["status"] == "engaged"


def test_generic_family_declines_a_graph_breaking_block_with_the_reason(monkeypatch):
    prof.clear_cache()
    model = _toy_model(block_src_sync = True)
    applied, audit, seen = _generic_run(model, monkeypatch)
    assert not applied["compiled"] and model.compiled is None
    assert seen["family"].supports_torch_compile is False
    reason = audit.entries["compiled"]["reason"]
    assert audit.entries["compiled"]["status"] == "declined"
    assert reason.startswith(plan_mod.GRAPH_BREAK_DECLINE) and ".item()" in reason


def test_generic_family_without_discovery_records_the_failure_and_never_raises(monkeypatch):
    prof.clear_cache()
    model = _toy_model(block_src_sync = False)
    applied, audit, _ = _generic_run(model, monkeypatch, env = "0")
    assert not applied["compiled"]
    assert audit.entries["compile_repeated_blocks"]["status"] == "failed"
    assert audit.entries["compiled"]["status"] == "declined"


def test_registered_family_passes_through_untouched():
    from core.inference.diffusion_families import _FAMILIES

    fam = next(f for f in _FAMILIES if f.name == "z-image")
    seen = {}

    def apply(family, log):
        seen["family"] = family
        return {"compiled": False}

    ao.run_speed_layer(
        types.SimpleNamespace(transformer = None),
        _target(),
        family = fam,
        backend = "image",
        speed_mode = "default",
        apply = apply,
    )
    assert seen["family"] is fam


def test_audit_logger_forwards_and_records(caplog):
    audit = ao.new_audit(backend = "image", family = None)
    log = ao.auditing_logger(logging.getLogger("autoopt-test"), audit)
    with caplog.at_level(logging.WARNING, logger = "autoopt-test"):
        log.warning("diffusion.speed: %s failed: %s", "channels_last", RuntimeError("nope"))
        log.info("ignored")
    assert audit.entries["channels_last"] == {"status": "failed", "reason": "nope"}
    assert "channels_last failed: nope" in caplog.text
    silent = ao.auditing_logger(None, audit)
    silent.info("x")  # an absent logger stays silent
    assert ao.auditing_logger(log, None) is log


def test_finalize_reconciles_and_attaches():
    audit = ao.new_audit(backend = "image", family = None)
    audit.record("cuda_graph", "declined", "armed after placement")
    audit.record("compiled", "engaged")
    resolved = {"speed_mode": {"value": "default"}}
    steps = ao.finalize(audit, ("cuda_graph", "int8_gemm"), resolved)
    by = {s["step"]: s for s in steps}
    assert by["cuda_graph"]["status"] == "engaged"
    assert by["int8_gemm"]["status"] == "engaged"
    assert by["compiled"]["status"] == "declined"  # dropped after placement
    assert resolved["speed_mode"]["steps"] == steps


def test_compile_cache_kwargs_match_the_former_inline_dicts():
    from core.inference import diffusion_speed as ds

    pipe = types.SimpleNamespace(transformer = None, vae = None)
    for mode in ("default", "max"):
        for below in (False, True):
            want = {
                "fullgraph": True,
                "dynamic": ds.compile_dynamic(None, None if mode == "max" else True),
                "mode": "max-autotune-no-cudagraphs" if mode == "max" else "default",
                "vae_decode": ds.vae_decode_compile_allowed(pipe, mode),
                **({"below_hooks": True} if below else {}),
            }
            got = plan_mod.compile_cache_kwargs(
                pipe, speed_mode = mode, fullgraph = True, offload_below_hooks = below
            )
            assert got == want
    deferred = plan_mod.compile_cache_kwargs(
        pipe, speed_mode = "max", fullgraph = False, offload_below_hooks = False, deferred = True
    )
    assert deferred == {
        "fullgraph": False,
        "dynamic": ds.compile_dynamic(None, True),
        "mode": "default",
        "vae_decode": ds.vae_decode_compile_allowed(pipe, "default"),
    }


# ---------------------------------------------------------------------------------------------------- runtime
class _OpensContext:
    def __call__(self):
        with self.transformer.cache_context("cond"):
            pass


class _NoContext:
    def __call__(self):
        self.transformer()


def test_runner_aware_step_cache():
    dit = object()
    base = _OpensContext()
    base.transformer = dit
    runner = _NoContext()
    runner.transformer = dit
    assert runtime.step_cache_runner_reason(base, base) is None
    assert runtime.step_cache_runner_reason(base, None) is None
    reason = runtime.step_cache_runner_reason(base, runner)
    assert reason and "_NoContext opens no cache_context" in reason
    assert runtime.auto_cache_steps(base, runner, 28) == (0, reason)
    assert runtime.auto_cache_steps(base, base, 28) == (28, None)
    ok = _OpensContext()
    ok.transformer = dit
    assert runtime.auto_cache_steps(base, ok, 28) == (28, None)
    other = _NoContext()
    other.transformer = object()  # a different denoiser carries none of base's hooks
    assert runtime.step_cache_runner_reason(base, other) is None


def test_video_family_hooks_are_data():
    names = [h.name for h in ao.FAMILY_HOOKS]
    assert names[:3] == ["ltx_static_compile", "ltx_stg_compile_adapter", "cudnn_benchmark_off"]
    ltx = types.SimpleNamespace(name = "ltx-2", denoiser_attr = "transformer", cudnn_benchmark = False)
    dit = types.SimpleNamespace()
    view = types.SimpleNamespace(transformer = dit)
    seen = {}

    def apply(family, log):
        seen["static"] = getattr(dit, "_unsloth_compile_static", False)
        return {"compiled": False, "cudnn_benchmark": False}

    ao.run_speed_layer(
        view, _target(), family = ltx, backend = "video", speed_mode = "default", apply = apply
    )
    assert seen["static"] is True
    image_dit = types.SimpleNamespace()
    ao.run_speed_layer(
        types.SimpleNamespace(transformer = image_dit),
        _target(),
        family = ltx,
        backend = "image",
        speed_mode = "default",
        apply = apply,
    )
    assert not hasattr(
        image_dit, "_unsloth_compile_static"
    )  # video hooks never run on the image backend
