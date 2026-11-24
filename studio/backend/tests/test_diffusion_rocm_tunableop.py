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
        self.tuning = True
        self.calls = []
        self.results = {}
        self.filename = "user.csv"
        self.iterations = 77
        self.duration = 91
        self.read_ok = True

    def state(self):
        return (self.enabled, self.tuning, self.filename, self.iterations, self.duration)

    def is_enabled(self):
        return self.enabled

    def tuning_is_enabled(self):
        return self.tuning

    def get_filename(self):
        return self.filename

    def get_max_tuning_iterations(self):
        return self.iterations

    def get_max_tuning_duration(self):
        return self.duration

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
        self.duration = ms
        self.calls.append(("duration", ms))

    def set_max_tuning_iterations(self, n):
        self.iterations = n
        self.calls.append(("iters", n))

    def read_file(self, name = None):
        self.calls.append(("read", name))
        return self.read_ok

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
        "is_enabled",
        "tuning_is_enabled",
        "get_filename",
        "get_max_tuning_iterations",
        "get_max_tuning_duration",
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
    for k in ("HIPBLASLT_WORKSPACE_SIZE", "CUBLASLT_WORKSPACE_SIZE"):
        monkeypatch.delenv(k, raising = False)
    rt._STATE["owners"].clear()
    rt._STATE.update(active = False, depth = 0, written = 0, file = None, tune = False)
    yield f
    rt._STATE["owners"].clear()
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
    before = fake.state()
    rt.install_for_pipe(pipe)
    rt.uninstall(pipe)
    assert fake.state() == before
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


class _UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_tune_hooks_survive_a_whole_module_fullgraph_compile(fake, monkeypatch):
    # The max tier compiles a U-Net whole (Module.compile, fullgraph), which traces the root module's hooks.
    pipe = types.SimpleNamespace(unet = _UNet(), vae = nn.Linear(4, 4))
    assert rt.install_for_pipe(pipe)
    assert pipe.unet._forward_pre_hooks and pipe.unet._forward_hooks
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: False
    )  # dynamo would save a CUDA RNG state
    pipe.unet.compile(fullgraph = True, dynamic = False, backend = "eager")
    x = torch.randn(2, 4)
    for _ in range(3):
        out = pipe.unet(x)
    torch.testing.assert_close(out, pipe.unet._call_impl(x))
    assert rt._STATE["depth"] == 0 and not fake.tuning
    # Eager calls still tune on a steady shape.
    pipe.unet._compiled_call_impl = None
    pipe.unet(x)
    pipe.unet(x)
    assert ("tuning", True) in fake.calls and not fake.tuning


def test_windows_and_linux_gfx1151_builds_each_get_their_own_table():
    """Windows Studio ships torch 2.11.0+rocm7.14.0 (multi-arch index), Linux 2.11.0+rocm7.13.0 (gfx1151 index):
    hipBLASLt differs, so each build reads only the table measured on it and never the other's solution ids."""
    from pathlib import Path

    root = Path(rt.__file__).resolve().parent.joinpath("rocm_gemm_tables")
    by_hip = {}
    for path in sorted(root.glob("gfx1151_*.csv")):
        validators = rt._validators(path.read_text(encoding = "utf-8"))
        by_hip[validators["HIP_VERSION"]] = (path, validators)
    assert {"713", "714"} <= set(by_hip)
    for hip, (path, validators) in by_hip.items():
        assert rt.shipped_tables("gfx1151", validators) == [path]
        for other, (other_path, _) in by_hip.items():
            if other != hip:
                assert other_path not in rt.shipped_tables("gfx1151", validators)
    windows = by_hip["714"][1]
    assert windows["PT_VERSION"] == "2.11.0" and windows["HIPBLASLT_VERSION"].startswith("100401-")


def _write_table(
    tmp_path,
    fake,
    name = "gfx1151_train.csv",
):
    rt._STATE["owners"].clear()  # earlier tests' pipes may still hold claims
    d = tmp_path / "tables"
    d.mkdir(exist_ok = True)
    rows = [f"Validator,{k},{v}" for k, v in fake.get_validators()]
    rows.append(
        "GemmTunableOp_BFloat16_NN,nn_3072_3840_3072_ld_3072_3072_3072,Gemm_Hipblaslt_386,1.96"
    )
    (d / name).write_text("\n".join(rows) + "\n", encoding = "utf-8")
    return d / name


def test_training_reads_the_shipped_table_never_tunes_and_release_restores(
    fake, tmp_path, monkeypatch
):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    table = _write_table(tmp_path, fake)
    assert rt.install_for_training()
    assert fake.enabled and not fake.tuning
    assert ("read", str(table)) in fake.calls
    assert rt.is_active()
    rt.release_training()
    assert not fake.enabled and not rt.is_active()


def test_training_without_a_matching_table_leaves_torch_alone(fake, monkeypatch):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    assert not rt.install_for_training()
    assert not fake.enabled and not rt.is_active()


def test_training_release_keeps_a_pipe_claim(fake, tmp_path, monkeypatch):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _write_table(tmp_path, fake)
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe)
    assert rt.install_for_training()
    rt.release_training()
    assert rt.is_active() and fake.enabled  # the render pipe still owns the table
    rt.release(pipe)
    assert not rt.is_active() and not fake.enabled


@pytest.mark.parametrize("why", ["kill", "user", "arch"])
def test_training_declines(fake, tmp_path, monkeypatch, why):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _write_table(tmp_path, fake)
    if why == "kill":
        monkeypatch.setenv(rt.TUNABLEOP_ENV, "0")
    elif why == "user":
        monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    else:
        monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1100")
    assert not rt.install_for_training()
    assert not fake.enabled


def test_trainer_installs_only_on_rocm_cuda_and_releases(monkeypatch):
    from core.training import diffusion_dit_trainer as ddt

    seen = []
    monkeypatch.setattr(
        rt, "install_for_training", lambda device = None, logger = None: seen.append("i") or True
    )
    monkeypatch.setattr(rt, "release_training", lambda: seen.append("r"))
    monkeypatch.setattr(ddt, "torch_is_rocm", lambda: False)
    assert not ddt._install_rocm_tunableop("cuda")
    monkeypatch.setattr(ddt, "torch_is_rocm", lambda: True)
    assert not ddt._install_rocm_tunableop("cpu")
    assert seen == []
    assert ddt._install_rocm_tunableop("cuda")
    ddt._release_rocm_tunableop()
    assert seen == ["i", "r"]


def test_shipped_training_table_is_well_formed():
    from pathlib import Path

    (path,) = sorted((Path(rt.__file__).parent / "rocm_gemm_tables").glob("gfx1151_*_train.csv"))
    rows = [
        ln.split(",")
        for ln in path.read_text(encoding = "utf-8").splitlines()
        if ln and not ln.startswith("Validator,")
    ]
    keys = [(r[0], r[1]) for r in rows]
    assert len(keys) == len(set(keys))
    assert ("GemmTunableOp_BFloat16_NT", "nt_16_3072_3840_ld_16_3072_16") in keys
    assert ("GemmTunableOp_BFloat16_NN", "nn_3072_3840_3072_ld_3072_3072_3072") in keys


@pytest.mark.parametrize("var", ["HIPBLASLT_WORKSPACE_SIZE", "CUBLASLT_WORKSPACE_SIZE"])
def test_a_user_workspace_cap_declines(fake, tmp_path, monkeypatch, var):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _write_table(tmp_path, fake)
    monkeypatch.setenv(var, "0")
    ok, why = rt.eligible()
    assert not ok and var in why
    assert not rt.install_for_training()
    assert not rt.install_for_pipe(_pipe(fake))
    assert not fake.enabled and not any(c[0] == "read" for c in fake.calls)


def test_an_incompatible_table_leaves_every_setting_alone(fake, monkeypatch, tmp_path):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _table(tmp_path, (("PT_VERSION", "2.12.0"), ("GCN_ARCH_NAME", "gfx1151")))
    before = fake.state()
    assert not rt.install_for_pipe(_pipe(fake))
    assert not rt.install_for_training()
    assert fake.state() == before


@pytest.mark.parametrize("who", ["pipe", "training"])
def test_last_release_restores_every_setting(fake, tmp_path, monkeypatch, who):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    _write_table(tmp_path, fake)
    before = fake.state()
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe) if who == "pipe" else rt.install_for_training()
    assert fake.enabled and fake.iterations == rt.MAX_TUNING_ITERS and not fake.tuning
    rt.release(pipe) if who == "pipe" else rt.release_training()
    assert fake.state() == before and not rt.is_active()


def test_trainer_with_a_malformed_local_cache_keeps_state_consistent(fake, tmp_path, monkeypatch):
    from core.training import diffusion_dit_trainer as ddt

    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    monkeypatch.setattr(ddt, "torch_is_rocm", lambda: True)
    _write_table(tmp_path, fake)
    cache = rt.results_file("gfx1151")
    cache.parent.mkdir(parents = True, exist_ok = True)
    cache.write_bytes(b"\xff\xfe\x00bad")
    before = fake.state()
    ok = ddt._install_rocm_tunableop("cuda")
    assert ok == fake.enabled == rt.is_active()
    ddt._release_rocm_tunableop()
    assert fake.state() == before and not rt.is_active()


def test_a_failed_read_is_a_failure_and_restores(fake, tmp_path, monkeypatch):
    from core.training import diffusion_dit_trainer as ddt

    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    monkeypatch.setattr(ddt, "torch_is_rocm", lambda: True)
    _write_table(tmp_path, fake)
    fake.read_ok = False
    before = fake.state()
    assert not ddt._install_rocm_tunableop("cuda")
    assert fake.state() == before and not rt.is_active()
    with pytest.raises(RuntimeError):
        rt.install_for_pipe(_pipe(fake))
    assert fake.state() == before and not rt.is_active()


def test_training_skips_inference_only_tables(fake, tmp_path, monkeypatch):
    monkeypatch.delenv(rt.TUNABLEOP_ENV)
    table = _write_table(tmp_path, fake, name = "gfx1151_windows.csv")
    before = fake.state()
    assert not rt.install_for_training()
    assert fake.state() == before and not any(c[0] == "read" for c in fake.calls)
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe) and ("read", str(table)) in fake.calls
    assert (
        not rt.install_for_training()
    )  # an inference-only table already active: no training claim
    rt.release(pipe)
    assert fake.state() == before and not rt.is_active()


def test_only_the_linux_shipped_table_serves_training():
    from pathlib import Path

    root = Path(rt.__file__).resolve().parent.joinpath("rocm_gemm_tables")
    train = {}
    for path in sorted(root.glob("gfx1151_*.csv")):
        hip = rt._validators(path.read_text(encoding = "utf-8"))["HIP_VERSION"]
        train[hip] = path.stem.endswith(rt.TRAIN_SUFFIX)
    assert train == {"713": True, "714": False}
