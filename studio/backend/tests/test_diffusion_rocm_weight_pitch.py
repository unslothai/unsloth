# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11xx Linear weight row pitch: arch / env gate, INT8 and other skips, the TunableOp table rule (no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_families as fams  # noqa: E402
from core.inference import diffusion_rocm_tunableop as tunableop  # noqa: E402
from core.inference import diffusion_rocm_weight_pitch as wp  # noqa: E402

TABLE_ROWS = [
    (
        "GemmTunableOp_BFloat16_TN",
        "tn_3072_4096_3072_ld_3072_3072_3072",
        "Gemm_Hipblaslt_629",
        "2.2",
    ),
    (
        "GemmTunableOp_BFloat16_TN",
        "tn_3072_512_3072_ld_3072_3072_3072",
        "Gemm_Hipblaslt_629",
        "0.3",
    ),
    (
        "GemmTunableOp_BFloat16_TN",
        "tn_12288_4096_4096_ld_4096_4096_12288",
        "Gemm_Hipblaslt_629",
        "9.1",
    ),
    ("GemmTunableOp_BFloat16_NN", "nn_3072_3840_9216_ld_3072_9216_3072", "Gemm_Hipblaslt_1", "1.0"),
]


@pytest.fixture
def gfx(monkeypatch, tmp_path):
    """CPU tensors stand in for the card; gfx1151 unless a test says otherwise; no table, no local cache."""
    monkeypatch.delenv(wp.PITCH_ENV, raising = False)
    monkeypatch.delenv(tunableop.TUNABLEOP_ENV, raising = False)
    monkeypatch.setattr(wp, "_DEVICE_TYPES", ("cpu",))
    state = {"arch": "gfx1151"}
    monkeypatch.setattr(wp, "_arch", lambda _d: state["arch"])
    monkeypatch.setattr(tunableop, "is_active", lambda: False)
    monkeypatch.setattr(tunableop, "results_file", lambda arch: tmp_path / f"{arch}_none.csv")
    for k in wp.STATS:
        wp.STATS[k] = 0
    return state


def _table_on(monkeypatch, rows = TABLE_ROWS):
    monkeypatch.setattr(tunableop, "is_active", lambda: True)
    tun = type("T", (), {"get_results": staticmethod(lambda: rows)})
    monkeypatch.setattr(torch.cuda, "tunable", tun, raising = False)


class Dit(torch.nn.Module):
    def __init__(self, dtype = torch.bfloat16):
        super().__init__()
        self.attn = torch.nn.Linear(3072, 3072, dtype = dtype)  # (N, K) in the table
        self.ffn = torch.nn.Linear(3072, 14336, dtype = dtype)  # not in the table
        self.small = torch.nn.Linear(256, 3072, dtype = dtype)  # K below MIN_IN_FEATURES

    def forward(self, x):
        return self.attn(x) + self.ffn(x)[..., :3072]


def _padded(lin) -> bool:
    return lin.weight.stride() == (lin.weight.shape[1] + 64, 1)


def test_pitch_gate_by_arch_and_env(monkeypatch):
    monkeypatch.delenv(wp.PITCH_ENV, raising = False)
    assert wp.pitch("gfx1151") == 64
    assert wp.pitch("gfx1100") == 0  # gfx11xx but unmeasured: opt-in only
    assert wp.pitch(None) == 0 and wp.pitch("gfx942") == 0
    monkeypatch.setenv(wp.PITCH_ENV, "0")
    assert wp.pitch("gfx1151") == 0
    monkeypatch.setenv(wp.PITCH_ENV, "128")
    assert wp.pitch("gfx1100") == 128 and wp.pitch("gfx1151") == 128
    assert wp.pitch("gfx942") == 0 and wp.pitch(None) == 0  # never off gfx11xx / off ROCm
    for bad in ("63", "-64", "abc"):
        monkeypatch.setenv(wp.PITCH_ENV, bad)
        assert wp.pitch("gfx1151") == 0


def test_pads_exactly_and_counts(gfx):
    dit = Dit()
    x = torch.randn(4, 3072, dtype = torch.bfloat16)
    ref_w = dit.attn.weight.detach().clone()
    ref = dit(x)
    assert wp.pad_weights(dit, family = "wan2.2-ti2v-5b") == 2
    assert _padded(dit.attn) and _padded(dit.ffn) and dit.small.weight.is_contiguous()
    assert torch.equal(dit.attn.weight, ref_w)
    assert isinstance(dit.attn.weight, torch.nn.Parameter)
    torch.testing.assert_close(dit(x), ref, rtol = 0, atol = 0)
    assert wp.STATS["padded"] == 2 and wp.STATS["bytes"] == (3072 + 14336) * 64 * 2


def test_releases_replaced_storage_once(gfx, monkeypatch):
    calls = []
    monkeypatch.setattr(wp, "_release_old_storage", lambda _t: calls.append(1))
    wp.pad_weights(Dit())
    assert calls == [1]
    gfx["arch"] = None
    wp.pad_weights(Dit())
    assert calls == [1]  # nothing re-stored, nothing to release


def test_first_forward_under_inference_mode_keeps_normal_weights(gfx):
    dit = Dit()
    assert wp.install(dit, family = "wan2.2-ti2v-5b") == 1
    with torch.inference_mode():
        dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert _padded(dit.attn) and not dit.attn.weight.is_inference()
    x = torch.randn(2, 3072, dtype = torch.bfloat16, requires_grad = True)
    dit(x).float().sum().backward()
    assert x.grad is not None


def test_off_arches_and_env_zero(gfx, monkeypatch):
    for arch in (None, "gfx1100"):
        gfx["arch"] = arch
        dit = Dit()
        assert wp.pad_weights(dit) == 0 and dit.attn.weight.is_contiguous()
    gfx["arch"] = "gfx1151"
    monkeypatch.setenv(wp.PITCH_ENV, "0")
    assert wp.pad_weights(Dit()) == 0


def test_skips_int8_fp32_subclass_and_noncontiguous(gfx):
    class Custom(torch.nn.Linear):
        pass

    m = torch.nn.Module()
    m.fp32 = torch.nn.Linear(2048, 2048)
    m.custom = Custom(2048, 2048, dtype = torch.bfloat16)
    m.int8 = torch.nn.Linear(2048, 2048, dtype = torch.bfloat16)
    m.int8.weight = torch.nn.Parameter(
        torch.zeros(2048, 2048, dtype = torch.int8), requires_grad = False
    )
    m.strided = torch.nn.Linear(2048, 2048, dtype = torch.bfloat16)
    m.strided.weight.data = torch.empty(2048, 4096, dtype = torch.bfloat16)[:, :2048]
    assert wp.pad_weights(m) == 0
    assert wp.STATS["skipped_other"] == 3  # Custom is not a plain Linear: never visited


def test_table_family_keeps_covered_weights(gfx, monkeypatch):
    _table_on(monkeypatch)
    dit = Dit()
    assert wp.pad_weights(dit, family = "qwen-image-2.1") == 1
    assert dit.attn.weight.is_contiguous()  # tn_3072_*_3072 rows: kept on the tuned solution
    assert _padded(dit.ffn)  # no row for (14336, 3072)
    assert wp.STATS["skipped_table"] == 1


def test_default_keeps_every_covered_shape_for_any_family(gfx, monkeypatch):
    """Env unset: a GEMM main served from the table keeps its contiguous weight, so the render stays byte-identical."""
    _table_on(monkeypatch)
    dit = Dit()
    assert wp.pad_weights(dit, family = "wan2.2-ti2v-5b") == 1
    assert dit.attn.weight.is_contiguous() and _padded(dit.ffn)
    assert wp.STATS["skipped_table"] == 1


def test_explicit_pitch_uses_the_family_rule(gfx, monkeypatch):
    """Env set: a family the table was not tuned for pads a shared (N, K) as without a table."""
    _table_on(monkeypatch)
    monkeypatch.setenv(wp.PITCH_ENV, "64")
    dit = Dit()
    assert wp.pad_weights(dit, family = "wan2.2-ti2v-5b") == 2
    assert wp.STATS["skipped_table"] == 0
    dit = Dit()
    assert (
        wp.pad_weights(dit, family = "qwen-image-2.1") == 1
    )  # table family keeps its rows either way


def test_no_table_pads_table_families(gfx):
    dit = Dit()
    assert wp.pad_weights(dit, family = "flux.3-klein") == 2


@pytest.mark.parametrize("how", ["tune", "cache"])
def test_user_tuned_rows_guard_every_family(gfx, monkeypatch, tmp_path, how):
    _table_on(monkeypatch)
    if how == "tune":
        monkeypatch.setenv(tunableop.TUNABLEOP_ENV, "tune")
    else:
        cache = tmp_path / "gfx1151_local.csv"
        cache.write_text("Validator,PT_VERSION,2.11.0\n", encoding = "utf-8")
        monkeypatch.setattr(tunableop, "results_file", lambda arch: cache)
    dit = Dit()
    assert wp.pad_weights(dit, family = "wan2.2-ti2v-5b") == 1
    assert dit.attn.weight.is_contiguous()


def test_unreadable_active_table_keeps_everything(gfx, monkeypatch):
    monkeypatch.setattr(tunableop, "is_active", lambda: True)

    def boom():
        raise RuntimeError("no results")

    monkeypatch.setattr(
        torch.cuda, "tunable", type("T", (), {"get_results": staticmethod(boom)}), raising = False
    )
    assert wp.pad_weights(Dit(), family = "qwen-image-2.1") == 0


def _user_tunable(
    monkeypatch,
    enabled = True,
    rows = TABLE_ROWS,
):
    """TunableOp enabled outside Studio (PYTORCH_TUNABLEOP_ENABLED=1 + a loaded file): Studio stays inactive."""
    monkeypatch.setattr(tunableop, "is_active", lambda: False)

    def results():
        if isinstance(rows, BaseException):
            raise rows
        return rows

    tun = type(
        "T",
        (),
        {"is_enabled": staticmethod(lambda: enabled), "get_results": staticmethod(results)},
    )
    monkeypatch.setattr(torch.cuda, "tunable", tun, raising = False)


@pytest.mark.parametrize("env", [None, "64"])
@pytest.mark.parametrize("family", ["qwen-image-2.1", "wan2.2-ti2v-5b"])
def test_user_enabled_tunableop_keeps_covered_rows(gfx, monkeypatch, env, family):
    _user_tunable(monkeypatch)
    if env is not None:
        monkeypatch.setenv(wp.PITCH_ENV, env)
    dit = Dit()
    assert wp.pad_weights(dit, family = family) == 1
    assert dit.attn.weight.is_contiguous()
    assert _padded(dit.ffn)


def test_user_enabled_tunableop_unreadable_pads_nothing(gfx, monkeypatch):
    _user_tunable(monkeypatch, rows = RuntimeError("no results"))
    dit = Dit()
    assert wp.pad_weights(dit, family = "wan2.2-ti2v-5b") == 0
    assert dit.attn.weight.is_contiguous() and dit.ffn.weight.is_contiguous()


@pytest.mark.parametrize("probe", ["off", "raises"])
def test_tunableop_disabled_unchanged(gfx, monkeypatch, probe):
    if probe == "off":
        _user_tunable(monkeypatch, enabled = False)
    else:

        def boom():
            raise RuntimeError("no TunableOp in this build")

        tun = type(
            "T",
            (),
            {"is_enabled": staticmethod(boom), "get_results": staticmethod(lambda: TABLE_ROWS)},
        )
        monkeypatch.setattr(torch.cuda, "tunable", tun, raising = False)
    assert wp.table_shapes() is None
    assert wp.pad_weights(Dit(), family = "wan2.2-ti2v-5b") == 2


def test_parse_row():
    assert wp.parse_row("tn_3072_4096_9216_ld_9216_9216_3072") == (3072, 9216, 4096)
    assert wp.parse_row("tn_3072_4096_9216_ld_9280_9216_3072") is None
    assert wp.parse_row("nn_3072_3840_9216_ld_3072_9216_3072") is None
    assert wp.parse_row("garbage") is None


def test_every_shipped_table_maps_to_known_families():
    names = {f.name for f in fams._FAMILIES}
    assert wp.TABLE_FAMILIES <= names
    assert {f for v in wp.TABLE_STEM_TOKENS.values() for f in v} == set(wp.TABLE_FAMILIES)
    tables = sorted(Path(tunableop._TABLE_DIR).glob("*.csv"))
    assert tables
    for path in tables:
        # gfx1151_torch2.11.0_hip713_<models...>[_1024][_train]
        tokens = path.stem.split("_")[3:]
        models = [t for t in tokens if not t.isdigit() and t != tunableop.TRAIN_SUFFIX.strip("_")]
        assert models, path.name
        for t in models:
            assert (
                t in wp.TABLE_STEM_TOKENS
            ), f"{path.name}: add {t!r} to TABLE_STEM_TOKENS / TABLE_FAMILIES"
        rows = [
            r.split(",")
            for r in path.read_text(encoding = "utf-8").splitlines()
            if r.startswith("Gemm")
        ]
        assert any(wp.parse_row(r[1]) for r in rows), path.name


def test_install_arms_once_and_pads_on_first_forward(gfx, monkeypatch):
    dit = Dit()
    assert wp.install(dit, family = "wan2.2-ti2v-5b") == 1
    assert wp.install(dit) == 0
    assert dit.attn.weight.is_contiguous()
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert _padded(dit.attn) and wp.STATS["calls"] == 1
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert wp.STATS["calls"] == 1


def test_install_off_card(gfx, monkeypatch):
    gfx["arch"] = None
    assert wp.install(Dit()) == 0
    gfx["arch"] = "gfx1151"
    monkeypatch.setenv(wp.PITCH_ENV, "0")
    assert wp.install(Dit()) == 0


def test_speed_optims_installs_only_on_rocm():
    src = (Path(wp.__file__).with_name("diffusion_speed.py")).read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "apply_speed_optims"
    )
    hits = [
        n for n in ast.walk(fn) if isinstance(n, ast.If) and "rocm_weight_pitch" in ast.unparse(n)
    ]
    assert hits and any("'rocm'" in ast.unparse(h.test) for h in hits)


def test_hook_stays_armed_under_graph_capture(gfx, monkeypatch):
    dit = Dit()
    assert wp.install(dit, family = "wan2.2-ti2v-5b") == 1
    capturing = {"on": True}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["on"])
    monkeypatch.setattr(wp, "_release_old_storage", lambda _t: None)
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert dit.attn.weight.is_contiguous() and wp.STATS["calls"] == 0
    assert dit.__dict__[wp._HOOK_ATTR]  # still armed
    assert wp.pad_weights(dit) == 0  # direct callers are gated too
    capturing["on"] = False
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert _padded(dit.attn) and wp.STATS["calls"] == 1
    assert dit.__dict__[wp._HOOK_ATTR] is False


def test_hook_stays_armed_while_compiling(gfx, monkeypatch):
    dit = Dit()
    assert wp.install(dit) == 1
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))
    assert dit.__dict__[wp._HOOK_ATTR] and wp.STATS["calls"] == 0


def _fail_on_second_empty(monkeypatch, exc_type):
    real = torch.empty
    seen = {"n": 0}

    def empty(*a, **k):
        if isinstance(a[0], tuple) and len(a[0]) == 2 and a[0][1] == 3072 + 64:
            seen["n"] += 1
            if seen["n"] == 2:
                raise exc_type("boom")
        return real(*a, **k)

    monkeypatch.setattr(torch, "empty", empty)


def test_oom_stops_padding_releases_and_logs(gfx, monkeypatch):
    calls, warns = [], []
    monkeypatch.setattr(wp, "_release_old_storage", lambda _t: calls.append(1))
    _fail_on_second_empty(monkeypatch, torch.cuda.OutOfMemoryError)
    log = type(
        "L", (), {"warning": lambda self, *a: warns.append(a), "info": lambda self, *a: None}
    )()
    dit = Dit()
    x = torch.randn(2, 3072, dtype = torch.bfloat16)
    ref = dit(x)
    assert wp.pad_weights(dit, log) == 1
    assert _padded(dit.attn) and dit.ffn.weight.is_contiguous()  # first padded, second untouched
    assert calls == [1] and wp.STATS["oom"] == 1 and wp.STATS["padded"] == 1
    assert warns and "out of memory" in warns[0][0]
    torch.testing.assert_close(dit(x), ref, rtol = 0, atol = 0)


def test_other_error_releases_storage_and_hook_logs(gfx, monkeypatch):
    calls, warns = [], []
    monkeypatch.setattr(wp, "_release_old_storage", lambda _t: calls.append(1))
    _fail_on_second_empty(monkeypatch, RuntimeError)
    log = type(
        "L", (), {"warning": lambda self, *a: warns.append(a), "info": lambda self, *a: None}
    )()
    dit = Dit()
    assert wp.install(dit, log) == 1
    dit(torch.randn(2, 3072, dtype = torch.bfloat16))  # render goes on
    assert _padded(dit.attn) and dit.ffn.weight.is_contiguous()
    assert calls == [1] and wp.STATS["padded"] == 1
    assert warns and "skipped" in warns[0][0]
    assert dit.__dict__[wp._HOOK_ATTR] is False  # a real attempt disarms


def test_arch_accepts_amd_sdk_wheel_without_version_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocm7.1.0")
    props = type("P", (), {"gcnArchName": "gfx1151:sramecc-:xnack-"})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _d: props)
    assert wp._arch(0) == "gfx1151"
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu130")
    assert wp._arch(0) is None
