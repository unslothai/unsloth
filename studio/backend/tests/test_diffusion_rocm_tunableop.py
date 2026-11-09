# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm TunableOp scoping for diffusion denoisers (a fake torch.cuda.tunable; no GPU)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_rocm_tunableop as rt  # noqa: E402


class FakeTunable:
    def __init__(self):
        self.enabled = False
        self.tuning = False
        self.calls = []
        self.results = {}
        self.filename = None

    def enable(self, v = True):
        self.enabled = bool(v)
        self.calls.append(("enable", bool(v)))

    def tuning_enable(self, v = True):
        self.tuning = bool(v)
        self.calls.append(("tuning", bool(v)))

    def set_filename(
        self,
        name,
        insert_device_ordinal = False,
    ):
        self.filename = name

    def set_max_tuning_duration(self, ms):
        self.calls.append(("duration", ms))

    def set_max_tuning_iterations(self, n):
        self.calls.append(("iters", n))

    def read_file(self, name = None):
        self.calls.append(("read", name))
        return True

    def get_results(self):
        return tuple(
            ("GemmTunableOp_BFloat16_TN", k, "Gemm_Hipblaslt_1", v) for k, v in self.results.items()
        )

    def get_validators(self):
        return (("PT_VERSION", "2.11.0"), ("GCN_ARCH_NAME", "gfx1151"))


class Denoiser(nn.Module):
    def __init__(self, fake):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.fake = fake
        self.seen = []

    def forward(self, x, **_kw):
        self.seen.append(self.fake.tuning)
        self.fake.results[f"shape{len(self.fake.results)}"] = 1.0
        return self.lin(x)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeTunable()
    for name in (
        "enable",
        "tuning_enable",
        "set_filename",
        "set_max_tuning_duration",
        "set_max_tuning_iterations",
        "read_file",
        "get_results",
        "get_validators",
    ):
        monkeypatch.setattr(torch.cuda.tunable, name, getattr(f, name))
    monkeypatch.setattr(rt, "is_rocm_torch", lambda _t: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1151")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(rt.TUNABLEOP_ENV, "tune")
    monkeypatch.setattr(rt, "_TABLE_DIR", tmp_path / "tables")
    monkeypatch.setattr(rt, "_prime", lambda _torch: None)
    for k in [k for k in __import__("os").environ if k.startswith("PYTORCH_TUNABLEOP_")]:
        monkeypatch.delenv(k)
    rt._STATE.update(active = False, depth = 0, written = 0, file = None, tune = False)
    yield f
    rt._STATE.update(active = False, depth = 0, written = 0, file = None, tune = False)


def _pipe(fake):
    return types.SimpleNamespace(transformer = Denoiser(fake), vae = nn.Linear(4, 4))


def test_tunes_only_inside_the_denoiser_and_persists(fake, tmp_path):
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe)
    assert fake.enabled and not fake.tuning
    assert fake.filename.startswith(str(tmp_path / "tunableop"))
    assert ("duration", rt.MAX_TUNING_MS) in fake.calls
    x = torch.randn(2, 4)
    pipe.transformer(x)  # a render's first call: lookups only
    assert pipe.transformer.seen == [False]
    pipe.transformer(x)  # the next step, same shapes: tunes
    assert pipe.transformer.seen == [False, True]
    assert not fake.tuning
    from pathlib import Path

    text = Path(fake.filename).read_text(encoding = "utf-8").splitlines()
    assert text[0] == "Validator,PT_VERSION,2.11.0"
    assert text[-1] == "GemmTunableOp_BFloat16_TN,shape1,Gemm_Hipblaslt_1,1.0"
    mtime = Path(fake.filename).stat().st_mtime_ns
    pipe.transformer.forward = lambda x: pipe.transformer.lin(x)  # nothing new tuned: no rewrite
    pipe.transformer(torch.randn(2, 4))
    assert Path(fake.filename).stat().st_mtime_ns == mtime


def test_nested_calls_keep_tuning_on_until_the_outer_returns(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)
    rt._pre(None, (torch.zeros(1, 4),), {})
    rt._post(None, (), {}, None)
    rt._pre(None, (torch.zeros(1, 4),), {})  # steady: tuning on
    pipe.transformer(torch.randn(1, 4))
    assert fake.tuning
    rt._post(None, (), {}, None)
    assert not fake.tuning


def test_exception_in_the_forward_turns_tuning_off(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)

    def boom(x):
        raise RuntimeError("x")

    pipe.transformer.forward = boom
    for _ in range(2):  # the second call is steady, so tuning is on when it raises
        with pytest.raises(RuntimeError):
            pipe.transformer(torch.randn(1, 4))
    assert not fake.tuning and rt._STATE["depth"] == 0


@pytest.mark.parametrize("why", ["kill", "user", "arch", "not_rocm"])
def test_declines(fake, monkeypatch, why):
    if why == "kill":
        monkeypatch.setenv(rt.TUNABLEOP_ENV, "0")
    elif why == "user":
        monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    elif why == "arch":
        monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1100")
    else:
        monkeypatch.setattr(rt, "is_rocm_torch", lambda _t: False)
    pipe = _pipe(fake)
    assert not rt.install_for_pipe(pipe)
    assert not fake.enabled
    assert not pipe.transformer._forward_pre_hooks


def test_uninstall_restores_the_default_pick(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)
    rt.uninstall(pipe)
    assert not fake.enabled and not fake.tuning
    assert not pipe.transformer._forward_pre_hooks and not pipe.transformer._forward_hooks
    assert not rt.is_active()


def test_results_file_keys_on_arch_torch_and_hip(fake, tmp_path):
    p = rt.results_file("gfx1151")
    assert p.parent == tmp_path / "tunableop"
    assert p.name.startswith("gfx1151_") and p.suffix == ".csv"


def test_speed_layer_installs_on_rocm_tiers_and_removes_on_off(monkeypatch):
    from core.inference import diffusion_speed as ds

    seen = []
    monkeypatch.setattr(
        rt, "install_for_pipe", lambda pipe, device = None, logger = None: seen.append("install")
    )
    monkeypatch.setattr(rt, "uninstall", lambda pipe = None: seen.append("uninstall"))
    pipe = types.SimpleNamespace(transformer = nn.Linear(2, 2), vae = None)
    rocm = types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, rocm, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]
    cuda = types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, cuda, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]


def test_prefill_and_new_prompt_shapes_do_not_tune(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)
    lat = torch.randn(1, 4)
    for text in (5, 9):
        pipe.transformer(
            lat, encoder_hidden_states = torch.zeros(1, text, 4), kv_cache_mode = "extract"
        )
        pipe.transformer(lat, encoder_hidden_states = torch.zeros(1, text, 4), kv_cache_mode = "cached")
        pipe.transformer(lat, encoder_hidden_states = torch.zeros(1, text, 4), kv_cache_mode = "cached")
    assert pipe.transformer.seen == [False, False, True, False, False, True]


def _table(tmp_path, validators):
    d = tmp_path / "tables"
    d.mkdir(exist_ok = True)
    p = d / "gfx1151_test.csv"
    lines = [f"Validator,{k},{v}" for k, v in validators]
    lines.append(
        "GemmTunableOp_BFloat16_TN,tn_4096_4096_4096_ld_4096_4096_4096,Gemm_Hipblaslt_547,3.8"
    )
    p.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return p


def test_default_reads_a_matching_table_and_never_tunes(fake, monkeypatch, tmp_path):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    table = _table(tmp_path, fake.get_validators())
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe)
    assert fake.enabled and ("read", str(table)) in fake.calls
    assert not pipe.transformer._forward_pre_hooks
    for _ in range(3):
        pipe.transformer(torch.randn(2, 4))
    assert pipe.transformer.seen == [False, False, False]
    assert ("tuning", True) not in fake.calls


@pytest.mark.parametrize("case", ["no_table", "other_build"])
def test_default_without_a_matching_table_leaves_tunableop_off(fake, monkeypatch, tmp_path, case):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    if case == "other_build":
        _table(tmp_path, (("PT_VERSION", "2.12.0"), ("GCN_ARCH_NAME", "gfx1151")))
    pipe = _pipe(fake)
    assert not rt.install_for_pipe(pipe)
    assert not fake.enabled and fake.calls[-1] == ("enable", False)
    assert not any(c[0] == "read" for c in fake.calls)
    assert not rt.is_active()


def test_table_with_a_missing_validator_does_not_match(fake, tmp_path):
    _table(tmp_path, (("PT_VERSION", "2.11.0"),))
    assert rt.shipped_tables("gfx1151", dict(fake.get_validators())) == []
    assert rt.shipped_tables("gfx1100", dict(fake.get_validators())) == []


def test_shipped_tables_are_torch_result_files_with_full_validators():
    from pathlib import Path

    tables = sorted(Path(rt.__file__).resolve().parent.joinpath("rocm_gemm_tables").glob("*.csv"))
    assert tables
    for path in tables:
        text = path.read_text(encoding = "utf-8")
        validators = rt._validators(text)
        assert set(validators) >= {
            "PT_VERSION",
            "HIP_VERSION",
            "HIPBLASLT_VERSION",
            "ROCBLAS_VERSION",
            "GCN_ARCH_NAME",
        }
        assert path.name.startswith(validators["GCN_ARCH_NAME"] + "_")
        assert validators["GCN_ARCH_NAME"] in rt.TUNED_ARCHES
        rows = [
            line.split(",")
            for line in text.splitlines()
            if line and not line.startswith("Validator,")
        ]
        assert rows and all(len(r) == 4 and r[0].startswith("GemmTunableOp_") for r in rows)
        assert rt.shipped_tables(validators["GCN_ARCH_NAME"], validators) == [path]


def test_shipped_tables_are_package_data():
    from pathlib import Path

    root = Path(rt.__file__).resolve().parents[4]
    text = (root / "pyproject.toml").read_text(encoding = "utf-8")
    assert '"backend/core/inference/rocm_gemm_tables/*.csv"' in text


def test_mixed_arch_hosts_decline(fake, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1100" if device == 1 else "gfx1151")
    ok, why = rt.eligible(0)
    assert not ok and "mixed archs" in why
    assert not rt.install_for_pipe(_pipe(fake), 0)
    assert not fake.enabled


def test_a_load_on_another_arch_drops_the_earlier_table(fake, monkeypatch, tmp_path):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _table(tmp_path, fake.get_validators())
    assert rt.install_for_pipe(_pipe(fake))
    assert fake.enabled and rt.is_active()
    monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1201")
    assert not rt.install_for_pipe(_pipe(fake))
    assert not fake.enabled and not rt.is_active()
