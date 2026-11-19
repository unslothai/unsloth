# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_decode_segments: expandable allocator segments for the span of an image VAE decode (CPU, fake torch)."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_decode_segments as D  # noqa: E402
from core.inference import diffusion_vae_oom as O  # noqa: E402


class _FakeTorch:
    def __init__(self, *, segments = (), backend = "native", hip = None, compiling = False, capturing = False,
                 setter = True, legacy = False, fail = False, settings = ""):  # fmt: skip
        self.calls = []
        self.settings = settings
        self.version = types.SimpleNamespace(hip = hip)

        self.expandable = D._conf_keys(settings).get("expandable_segments") == "true"

        def set_settings(value):
            if fail:
                raise RuntimeError("refused")
            self.calls.append(value)
            self.settings = value
            flag = D._conf_keys(value).get("expandable_segments")
            if flag is not None:
                self.expandable = flag == "true"

        c = types.SimpleNamespace()
        if setter:
            setattr(c, "_cuda_cudaCachingAllocator_set_allocator_settings" if legacy
                    else "_accelerator_setAllocatorSettings", set_settings)  # fmt: skip
        self._C = c
        self.compiler = types.SimpleNamespace(is_compiling = lambda: compiling)
        self.cuda = types.SimpleNamespace(
            is_current_stream_capturing = lambda: capturing,
            memory_snapshot = lambda: [dict(s) for s in segments],
            memory = types.SimpleNamespace(
                get_allocator_backend = lambda: backend,
                _snapshot = lambda: {
                    "segments": [dict(s) for s in segments],
                    "allocator_settings": {
                        "PYTORCH_CUDA_ALLOC_CONF": self.settings,
                        "expandable_segments": self.expandable,
                    },
                },
            ),
        )


class _Vae:
    pass


def _cuda_z():
    return types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda"))


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    monkeypatch.setattr(D, "_platform_ok", lambda: True)
    for name in D._ALLOC_CONF_VARS + (D.DECODE_EXPANDABLE_ENV,):
        monkeypatch.delenv(name, raising = False)

    def install(fake):
        monkeypatch.setattr(D, "_torch", lambda: fake)
        return fake

    return install


def _vae():
    vae = _Vae()
    assert D.enable(vae)
    return vae


def _decode(
    vae,
    fake,
    z = None,
):
    seen = []
    with D.expandable_decode(vae, z if z is not None else _cuda_z()):
        seen.append(list(fake.calls))
    return seen[0]


def test_flips_on_for_the_decode_and_back_off(fresh):
    fake = fresh(_FakeTorch())
    inside = _decode(_vae(), fake)
    assert inside == ["expandable_segments:True"]
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]
    assert fake.settings == ""


def test_legacy_setter_on_older_torch(fresh):
    fake = fresh(_FakeTorch(legacy = True))
    _decode(_vae(), fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]


def test_explicit_off_is_restored_as_it_was(fresh):
    fake = fresh(_FakeTorch(settings = "expandable_segments:False"))
    _decode(_vae(), fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]
    assert fake.settings == "expandable_segments:False"


@pytest.mark.parametrize(
    "live",
    [
        "expandable_segments:True",  # opted in at runtime before any expandable segment exists
        "expandable_segments:True,pinned_num_register_threads:1",
        "max_split_size_mb:128,garbage_collection_threshold:0.6",  # a set would reset these
        "expandable_segments:False,max_split_size_mb:128",
    ],
)
def test_runtime_allocator_settings_are_left_alone(fresh, live):
    fake = fresh(_FakeTorch(settings = live))
    _decode(_vae(), fake)
    assert fake.calls == [] and fake.settings == live


def test_a_runtime_opt_in_after_an_engaged_decode_is_kept(fresh):
    fake = fresh(_FakeTorch())
    vae = _vae()
    _decode(vae, fake)
    fake.calls.clear()
    fake.settings, fake.expandable = "expandable_segments:True", True
    _decode(vae, fake)
    assert fake.calls == [] and fake.settings == "expandable_segments:True"


def test_segments_an_earlier_decode_left_do_not_stop_the_next(fresh):
    # The flag is read live, so expandable segments still cached from the last scope are not "already on".
    fake = fresh(_FakeTorch(segments = [{"is_expandable": True}]))
    vae = _vae()
    _decode(vae, fake)
    _decode(vae, fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""] * 2


def test_a_flag_left_on_by_an_earlier_string_is_kept(fresh):
    # "expandable_segments:True" then "" leaves the flag on with an empty string: only the live flag shows it
    fake = fresh(_FakeTorch())
    fake.expandable = True
    _decode(_vae(), fake)
    assert fake.calls == [] and fake.expandable is True


def test_unreadable_live_settings_never_engage(fresh):
    fake = fresh(_FakeTorch())
    fake.cuda.memory._snapshot = lambda: {"segments": []}
    _decode(_vae(), fake)
    assert fake.calls == []


def test_nested_decodes_flip_once(fresh):
    fake = fresh(_FakeTorch())
    vae = _vae()
    with D.expandable_decode(vae, _cuda_z()):
        with D.expandable_decode(vae, _cuda_z()):
            assert fake.calls == ["expandable_segments:True"]
        assert fake.calls == ["expandable_segments:True"]
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]


def test_restores_after_a_failing_decode(fresh):
    fake = fresh(_FakeTorch())
    with pytest.raises(ValueError):
        with D.expandable_decode(_vae(), _cuda_z()):
            raise ValueError("decode failed")
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]


@pytest.mark.parametrize(
    "fake_kwargs",
    [
        {"backend": "cudaMallocAsync"},
        {"hip": "6.4"},
        {"compiling": True},
        {"capturing": True},
        {"setter": False},
    ],
)
def test_never_touches_the_allocator_when_not_allowed(fresh, fake_kwargs):
    fake = fresh(_FakeTorch(**fake_kwargs))
    _decode(_vae(), fake)
    assert fake.calls == []


@pytest.mark.parametrize(
    "conf",
    [
        "max_split_size_mb:128",
        "expandable_segments:False",
        "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]",
        "backend:cudaMallocAsync",
        "garbage_collection_threshold:0.6",
    ],
)
def test_user_allocator_config_wins(fresh, monkeypatch, conf):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", conf)
    fake = fresh(_FakeTorch())
    _decode(_vae(), fake)
    assert fake.calls == []


def test_bare_expandable_true_export_after_init_still_engages(fresh, monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    fake = fresh(_FakeTorch(segments = [{"is_expandable": False}]))
    _decode(_vae(), fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]


def test_kill_switch_and_opt_in(fresh, monkeypatch):
    fake = fresh(_FakeTorch())
    _decode(_Vae(), fake)
    assert fake.calls == []
    vae = _vae()
    monkeypatch.setenv(D.DECODE_EXPANDABLE_ENV, "0")
    _decode(vae, fake)
    assert fake.calls == []
    assert not D.enable(_Vae())


def test_cpu_and_mps_inputs_untouched(fresh):
    fake = fresh(_FakeTorch())
    for kind in ("cpu", "mps", "xpu"):
        _decode(_vae(), fake, types.SimpleNamespace(device = types.SimpleNamespace(type = kind)))
    assert fake.calls == []


def test_refused_setting_disables_for_the_process(fresh):
    fake = fresh(_FakeTorch(fail = True))
    vae = _vae()
    ran = []
    with D.expandable_decode(vae, _cuda_z()):
        ran.append(1)
    assert ran == [1] and D._usable is False


def test_platform_gate(monkeypatch):
    monkeypatch.setattr(D.sys, "platform", "win32")
    assert not D._platform_ok()
    monkeypatch.setattr(D.sys, "platform", "linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert not D._platform_ok()
    monkeypatch.delenv("WSL_DISTRO_NAME")
    monkeypatch.delenv("WSL_INTEROP", raising = False)
    monkeypatch.setattr(
        D.os,
        "uname",
        lambda: types.SimpleNamespace(release = "5.15.167.4-microsoft-standard-WSL2"),
        raising = False,
    )
    assert not D._platform_ok()
    monkeypatch.setattr(
        D.os, "uname", lambda: types.SimpleNamespace(release = "6.8.0-1015-aws"), raising = False
    )
    monkeypatch.setattr(D.os.path, "exists", lambda p: False)
    assert D._platform_ok()


def test_conf_keys_keeps_bracketed_lists():
    keys = D._conf_keys("expandable_segments:True, roundup_power2_divisions:[32:256,64:128]")
    assert keys == {"expandable_segments": "true", "roundup_power2_divisions": "[32:256,64:128]"}


class _ScopeVae(torch.nn.Module):
    def __init__(self, fake):
        super().__init__()
        self.use_tiling = False
        self.fake = fake
        self.during = []

    def enable_tiling(self):
        self.use_tiling = True

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        self.during.append(list(self.fake.calls))
        return z

    def decode(
        self,
        z,
        return_dict = True,
    ):
        return self._decode(z, return_dict = return_dict)


def test_oom_scope_wraps_the_whole_decode(fresh):
    fake = fresh(_FakeTorch())
    vae = _ScopeVae(fake)
    assert O.install(vae)
    D.enable(vae)
    z = _cuda_z()
    assert vae.decode(z) is z
    assert vae.during == [["expandable_segments:True"]]
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False", ""]
    O.uninstall(vae)


def _real_live():
    try:
        return torch.cuda.memory._snapshot()["allocator_settings"]
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.skipif(
    _real_live() is None or D._setter(torch) is None,
    reason = "torch build without the CUDA allocator settings",
)
@pytest.mark.parametrize(
    "first, live, engages",
    [
        (None, "", True),
        (None, "expandable_segments:False", True),
        (None, "max_split_size_mb:128,garbage_collection_threshold:0.6", False),
        (
            None,
            "expandable_segments:True",
            False,
        ),
        (
            "expandable_segments:True",
            "",
            False,
        ),
    ],
)
def test_real_allocator_settings_come_back_exactly(fresh, first, live, engages):
    """torch's own setter and live settings (they parse on a GPU-less CUDA build too); only the CUDA probes are faked."""
    real_set = D._setter(torch)
    before = _real_live()
    fake = _FakeTorch()
    fake._C = types.SimpleNamespace(_accelerator_setAllocatorSettings = real_set)
    fake.cuda.memory._snapshot = lambda: {"segments": [], "allocator_settings": _real_live()}
    fresh(fake)
    try:
        real_set("expandable_segments:False")
        if first is not None:
            real_set(first)
        real_set(live)
        want = dict(_real_live())
        engaged = D.STATS["engaged"]
        _decode(_vae(), fake)
        assert dict(_real_live()) == want
        assert D.STATS["engaged"] - engaged == int(engages)
    finally:
        real_set(
            "expandable_segments:True"
            if before["expandable_segments"]
            else "expandable_segments:False"
        )
        real_set(before["PYTORCH_CUDA_ALLOC_CONF"])


_real_gpu = torch.cuda.is_available() and sys.platform.startswith("linux")


@pytest.mark.skipif(
    not (_real_gpu and getattr(torch.version, "hip", None)), reason = "ROCm device only"
)
def test_rocm_device_never_engages(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    monkeypatch.setattr(D, "STATS", {"engaged": 0, "skipped": 0})
    vae = _Vae()
    D.enable(vae)
    z = torch.zeros(1, device = "cuda")
    with D.expandable_decode(vae, z):
        torch.empty(64 << 20, dtype = torch.uint8, device = "cuda")
    assert D._usable is False and D.STATS["engaged"] == 0


@pytest.mark.skipif(
    not (_real_gpu and not getattr(torch.version, "hip", None))
    or not D._platform_ok()
    or D._allocator_configured(),
    reason = "Linux CUDA device with the default allocator only",
)
def test_cuda_device_maps_the_decode_into_an_expandable_segment(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    torch.cuda.init()
    if any(seg.get("is_expandable") for seg in torch.cuda.memory_snapshot()):
        pytest.skip("expandable segments already on in this process")
    vae = _Vae()
    D.enable(vae)
    z = torch.zeros(1, device = "cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # a cached fixed block would serve the allocation below

    def expandable(t):
        ptr = t.data_ptr()
        for seg in torch.cuda.memory_snapshot():
            if seg["address"] <= ptr < seg["address"] + seg["total_size"]:
                return bool(seg.get("is_expandable"))
        raise AssertionError("tensor not in any segment")

    with D.expandable_decode(vae, z):
        inside = torch.empty(96 << 20, dtype = torch.uint8, device = "cuda")
        assert expandable(inside)
    after = torch.empty(80 << 20, dtype = torch.uint8, device = "cuda")
    assert not expandable(after)
    del inside, after


@pytest.mark.skipif(
    not (_real_gpu and not getattr(torch.version, "hip", None))
    or not D._platform_ok()
    or D._allocator_configured(),
    reason = "Linux CUDA device with the default allocator only",
)
def test_cuda_device_keeps_a_runtime_opt_in_and_restores_the_settings(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    torch.cuda.init()
    before = torch.cuda.memory._snapshot()["allocator_settings"]
    vae = _Vae()
    D.enable(vae)
    z = torch.zeros(1, device = "cuda")
    try:
        D._setter(torch)("expandable_segments:True")
        with D.expandable_decode(vae, z):
            pass
        assert torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"] is True
        D._setter(torch)("expandable_segments:False")
        D._setter(torch)("garbage_collection_threshold:0.6")
        live = dict(torch.cuda.memory._snapshot()["allocator_settings"])
        with D.expandable_decode(vae, z):
            pass
        assert dict(torch.cuda.memory._snapshot()["allocator_settings"]) == live
    finally:
        D._setter(torch)(
            "expandable_segments:True"
            if before["expandable_segments"]
            else "expandable_segments:False"
        )
        D._setter(torch)(before["PYTORCH_CUDA_ALLOC_CONF"])


@pytest.mark.skipif(
    not (_real_gpu and not getattr(torch.version, "hip", None))
    or not D._platform_ok()
    or D._allocator_configured(),
    reason = "Linux CUDA device with the default allocator only",
)
def test_cuda_device_engages_every_decode_not_only_the_first(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    torch.cuda.init()
    if torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]:
        pytest.skip("expandable segments already on in this process")
    vae = _Vae()
    D.enable(vae)
    z = torch.zeros(1, device = "cuda")
    engaged = D.STATS["engaged"]
    for _ in range(3):
        with D.expandable_decode(vae, z):
            held = torch.empty(64 << 20, dtype = torch.uint8, device = "cuda")
            del held
    assert D.STATS["engaged"] - engaged == 3
