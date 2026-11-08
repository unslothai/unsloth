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
    """Allocator settings as ``AllocatorConfig::parseArgs`` keeps them: each call resets max_split_size /
    garbage_collection_threshold / roundup to defaults, applies the keys it names and records the string."""

    def __init__(self, *, backend = "native", hip = None, compiling = False, capturing = False,
                 setter = True, legacy = False, fail = False, live = "", settings = True):  # fmt: skip
        self.calls = []
        self.version = types.SimpleNamespace(hip = hip)
        self.state = {"expandable_segments": False, "release_lock_on_cudamalloc": False}
        self._parse(live)

        def set_settings(value):
            if fail:
                raise RuntimeError("refused")
            self.calls.append(value)
            self._parse(value)

        def snapshot():
            return (
                {"segments": [], "allocator_settings": dict(self.state)}
                if settings
                else {"segments": []}
            )

        c = types.SimpleNamespace()
        if setter:
            setattr(c, "_cuda_cudaCachingAllocator_set_allocator_settings" if legacy
                    else "_accelerator_setAllocatorSettings", set_settings)  # fmt: skip
        self._C = c
        self.compiler = types.SimpleNamespace(is_compiling = lambda: compiling)
        self.cuda = types.SimpleNamespace(
            is_current_stream_capturing = lambda: capturing,
            memory = types.SimpleNamespace(get_allocator_backend = lambda: backend, _snapshot = snapshot),
        )

    def _parse(self, value):
        self.state.update(
            {
                "max_split_size": -1,
                "garbage_collection_threshold": 0.0,
                "roundup_power2_divisions": None,
            }
        )
        self.state["PYTORCH_CUDA_ALLOC_CONF"] = value
        for part in D._conf_parts(value):
            key, _, val = part.partition(":")
            if key == "max_split_size_mb":
                self.state["max_split_size"] = int(val) << 20
            elif key == "garbage_collection_threshold":
                self.state["garbage_collection_threshold"] = float(val)
            elif key == "roundup_power2_divisions":
                self.state["roundup_power2_divisions"] = val
            elif key in ("expandable_segments", "release_lock_on_cudamalloc"):
                self.state[key] = val == "True"

    def settings(self):
        return {k: v for k, v in self.state.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}


class _Vae:
    pass


def _cuda_z():
    return types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda"))


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    monkeypatch.setattr(D, "_saved", None)
    monkeypatch.setattr(D, "_written", (None, None))
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
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]


def test_legacy_setter_on_older_torch(fresh):
    fake = fresh(_FakeTorch(legacy = True))
    _decode(_vae(), fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]


def test_nested_decodes_flip_once(fresh):
    fake = fresh(_FakeTorch())
    vae = _vae()
    with D.expandable_decode(vae, _cuda_z()):
        with D.expandable_decode(vae, _cuda_z()):
            assert fake.calls == ["expandable_segments:True"]
        assert fake.calls == ["expandable_segments:True"]
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]


def test_restores_after_a_failing_decode(fresh):
    fake = fresh(_FakeTorch())
    with pytest.raises(ValueError):
        with D.expandable_decode(_vae(), _cuda_z()):
            raise ValueError("decode failed")
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]


@pytest.mark.parametrize(
    "fake_kwargs",
    [
        {"live": "expandable_segments:True"},  # already on process-wide
        {"live": "expandable_segments:False"},  # switched off on purpose
        {"backend": "cudaMallocAsync"},
        {"hip": "6.4"},
        {"compiling": True},
        {"capturing": True},
        {"setter": False},
        {"settings": False},  # live settings unreadable: they could not be restored
    ],
)
def test_never_touches_the_allocator_when_not_allowed(fresh, fake_kwargs):
    fake = fresh(_FakeTorch(**fake_kwargs))
    _decode(_vae(), fake)
    assert fake.calls == []


@pytest.mark.parametrize(
    "live",
    [
        "max_split_size_mb:128,garbage_collection_threshold:0.7",
        "roundup_power2_divisions:[32:256,64:128,256:64,>:32]",
        "release_lock_on_cudamalloc:True,max_split_size_mb:512",
    ],
)
def test_runtime_allocator_settings_survive_the_decode(fresh, live):
    """Settings applied through the runtime setter (no environment variable) are kept inside and after."""
    fake = fresh(_FakeTorch(live = live))
    before = fake.settings()
    vae = _vae()
    seen = []
    for _ in range(2):  # the second decode reads the first one's restore as the user's string
        with D.expandable_decode(vae, _cuda_z()):
            inside = fake.settings()
            seen.append(inside.pop("expandable_segments"))
            assert inside == {k: v for k, v in before.items() if k != "expandable_segments"}
        assert fake.settings() == before
    assert seen == [True, True] and len(fake.calls) == 4 and D._usable is True


def test_env_allocator_config_is_kept_too(fresh, monkeypatch):
    conf = "max_split_size_mb:128,garbage_collection_threshold:0.6"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", conf)
    fake = fresh(_FakeTorch(live = conf))
    before = fake.settings()
    inside = _decode(_vae(), fake)
    assert inside == [conf + ",expandable_segments:True"]
    assert fake.settings() == before


def test_bare_expandable_true_export_after_init_still_engages(fresh, monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    fake = fresh(_FakeTorch())
    _decode(_vae(), fake)
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]


def test_a_restore_that_reads_back_different_ends_it(fresh, monkeypatch):
    fake = fresh(_FakeTorch(live = "max_split_size_mb:128"))
    real = fake._parse
    fake._parse = lambda value: real(value.replace("max_split_size_mb:128,", ""))
    _decode(_vae(), fake)
    assert D._usable is False


def test_kill_switch_and_opt_in(fresh, monkeypatch):
    fake = fresh(_FakeTorch())
    _decode(_Vae(), fake)  # not opted in: video and other VAEs
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
    assert fake.calls == ["expandable_segments:True", "expandable_segments:False"]
    O.uninstall(vae)


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
    not (_real_gpu and not getattr(torch.version, "hip", None)) or not D._platform_ok(),
    reason = "Linux CUDA device with the default allocator only",
)
def test_cuda_device_maps_the_decode_into_an_expandable_segment(monkeypatch):
    monkeypatch.setattr(D, "_usable", None)
    monkeypatch.setattr(D, "_depth", 0)
    monkeypatch.setattr(D, "_saved", None)
    monkeypatch.setattr(D, "_written", (None, None))
    torch.cuda.init()
    if any(seg.get("is_expandable") for seg in torch.cuda.memory_snapshot()):
        pytest.skip("expandable segments already on in this process")
    live = D._live_settings(torch)
    if not D._decide(torch) or live is None or D._base_conf(live) is None:
        pytest.skip(
            "this process's allocator settings rule the toggle out (backend, or expandable set)"
        )
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
