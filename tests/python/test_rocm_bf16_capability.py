import inspect
import types
from pathlib import Path

import pytest

from unsloth.device_type import arch_lacks_bf16


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"


@pytest.mark.parametrize(
    "arch",
    ["gfx1010", "gfx1012", "gfx1030", "gfx1031", "gfx1032:sramecc-:xnack-", "GFX1036", " gfx1030 "],
)
def test_gfx10_lacks_bf16(arch):
    assert arch_lacks_bf16(arch) is True


@pytest.mark.parametrize(
    "arch",
    ["gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx90a", "gfx942", "gfx908"],
)
def test_newer_rdna_and_cdna_keep_bf16(arch):
    assert arch_lacks_bf16(arch) is False


@pytest.mark.parametrize("arch", ["", None, "unknown"])
def test_unreadable_arch_does_not_disable_bf16(arch):
    assert arch_lacks_bf16(arch) is False


def test_one_unreadable_device_keeps_the_others(monkeypatch):
    """Only an unreadable device COUNT may empty the list; a wedged device must not (#7922)."""
    import types

    import unsloth.device_type as dt

    if not hasattr(dt, "torch"):
        pytest.skip("device_type stub or MLX host; the real HIP probe is not loaded")

    class _Props:
        gcnArchName = "gfx1032"

    def _props(i):
        if i == 1:
            raise RuntimeError("device wedged")
        return _Props()

    monkeypatch.setattr(
        dt,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(device_count = lambda: 2, get_device_properties = _props)
        ),
    )
    assert dt.hip_visible_archs() == ["gfx1032"]

    def _count_raises():
        raise RuntimeError("no HIP runtime")

    monkeypatch.setattr(
        dt,
        "torch",
        types.SimpleNamespace(cuda = types.SimpleNamespace(device_count = _count_raises)),
    )
    assert dt.hip_visible_archs() == []


def test_gpu_init_gates_on_every_visible_device():
    source = GPU_INIT.read_text(encoding = "utf-8")
    hip_branch = source.split('elif DEVICE_TYPE == "hip":', 1)[1].split("\nelif ", 1)[0]
    assert "arch_lacks_bf16" in hip_branch
    assert "hip_visible_archs()" in hip_branch
    assert "get_device_properties(0)" not in hip_branch


def test_model_utils_uses_the_patched_hip_probe():
    source = MODEL_UTILS.read_text(encoding = "utf-8")
    hip_branch = source.split('elif DEVICE_TYPE == "hip":', 1)[1].split("\nelif ", 1)[0]
    assert "SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()" in hip_branch
    assert "SUPPORTS_BFLOAT16 = True" not in hip_branch


# The tests below exec the real bf16 chain: no CI has gfx10, and a text assert only checks spelling.

_CHAIN_START = 'if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():'
_CHAIN_END = "\n# For Gradio HF Spaces?"


def _fake_torch(
    archs,
    base_bf16 = True,
    count_raises = False,
    props_raises_on = (),
):
    def device_count():
        if count_raises:
            raise RuntimeError("no HIP runtime")
        return len(archs)

    def get_device_properties(i):
        if i in props_raises_on:
            raise RuntimeError("device wedged")
        return types.SimpleNamespace(gcnArchName = archs[i])

    # Not *args: the cuda branch sniffs this signature with inspect.signature and would fall back.
    def is_bf16_supported(including_emulation = True):
        return base_bf16

    return types.SimpleNamespace(
        version = types.SimpleNamespace(hip = "6.2.4", cuda = None),
        cuda = types.SimpleNamespace(
            device_count = device_count,
            get_device_properties = get_device_properties,
            is_bf16_supported = is_bf16_supported,
            is_available = lambda: True,
            get_device_capability = lambda: (9, 0),
        ),
        xpu = types.SimpleNamespace(is_bf16_supported = lambda: True),
    )


def _namespace(fake_torch, device_type):
    from unsloth.device_type import hip_visible_archs
    return {
        "torch": fake_torch,
        "inspect": inspect,
        "DEVICE_TYPE": device_type,
        "arch_lacks_bf16": arch_lacks_bf16,
        # Reads unsloth.device_type's own `torch`, not this fake, so the caller must monkeypatch.
        "hip_visible_archs": hip_visible_archs,
    }


def _run_chain(monkeypatch, fake_torch, device_type):
    import unsloth.device_type as dt

    monkeypatch.setattr(dt, "torch", fake_torch, raising = False)
    source = GPU_INIT.read_text(encoding = "utf-8")
    body = _CHAIN_START + source.split(_CHAIN_START, 1)[1].split(_CHAIN_END, 1)[0]
    namespace = _namespace(fake_torch, device_type)
    exec(compile(body, str(GPU_INIT), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((True,), {}),
        ((False,), {}),
        ((), {"including_emulation": True}),
        ((), {"including_emulation": False}),
        ((), {"a_future_kwarg": 1}),
    ],
)
@pytest.mark.parametrize("archs,expected", [(["gfx1032"], False), (["gfx1100"], True)])
def test_patched_probe_accepts_every_call_form(monkeypatch, archs, expected, args, kwargs):
    """including_emulation=False must not reopen the gate: ROCm torch returns True regardless."""
    fake = _fake_torch(archs)
    namespace = _run_chain(monkeypatch, fake, "hip")
    assert namespace["SUPPORTS_BFLOAT16"] is expected
    assert fake.cuda.is_bf16_supported(*args, **kwargs) is expected


@pytest.mark.parametrize(
    "archs,expected",
    [
        (["gfx1030", "gfx1100"], False),
        (["gfx1100", "gfx1030"], False),
        (["gfx1100", "gfx1101"], True),
    ],
)
def test_mixed_host_disables_bf16_process_wide(monkeypatch, archs, expected):
    """SUPPORTS_BFLOAT16 is one module constant, so a mixed host cannot be judged per card."""
    namespace = _run_chain(monkeypatch, _fake_torch(archs), "hip")
    assert namespace["SUPPORTS_BFLOAT16"] is expected


@pytest.mark.parametrize(
    "archs,kwargs",
    [
        ([], {}),
        (["gfx1032"], {"count_raises": True}),
        (["gfx1032"], {"props_raises_on": (0,)}),
    ],
)
def test_an_unreadable_probe_leaves_torchs_answer_alone(monkeypatch, archs, kwargs):
    """Fail-open on purpose: guessing False would drop bf16 on any CDNA host whose probe hiccups."""
    namespace = _run_chain(monkeypatch, _fake_torch(archs, **kwargs), "hip")
    assert namespace["SUPPORTS_BFLOAT16"] is True


def test_one_wedged_device_does_not_discard_the_gfx10_beside_it(monkeypatch):
    namespace = _run_chain(
        monkeypatch, _fake_torch(["gfx1032", "gfx1100"], props_raises_on = (1,)), "hip"
    )
    assert namespace["SUPPORTS_BFLOAT16"] is False


def test_torch_saying_no_is_still_respected(monkeypatch):
    namespace = _run_chain(monkeypatch, _fake_torch(["gfx1100"], base_bf16 = False), "hip")
    assert namespace["SUPPORTS_BFLOAT16"] is False


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_the_gate_does_not_leak_off_hip(monkeypatch, device_type):
    fake = _fake_torch(["gfx1032"])
    namespace = _run_chain(monkeypatch, fake, device_type)
    assert namespace["SUPPORTS_BFLOAT16"] is True
    if device_type == "cuda":
        assert fake.cuda.is_bf16_supported(including_emulation = False) is True


def test_importing_unsloth_twice_is_stable(monkeypatch):
    """The second pass captures the already-patched probe, which must not recurse."""
    fake = _fake_torch(["gfx1032"])
    _run_chain(monkeypatch, fake, "hip")
    namespace = _run_chain(monkeypatch, fake, "hip")
    assert namespace["SUPPORTS_BFLOAT16"] is False
    assert fake.cuda.is_bf16_supported() is False
