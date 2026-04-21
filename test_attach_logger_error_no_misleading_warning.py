import ast
import warnings as _warnings
from pathlib import Path
import torch


def _find_vision():
    for p in [
        Path(__file__).resolve().parent / "unsloth" / "models" / "vision.py",
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("vision.py not found")


class _RaisingLogger:
    def info(self, *a, **kw):
        raise RuntimeError("simulated broken logging handler")


def _load_fns(logger):
    tree = ast.parse(_find_vision().read_text())
    ns = {"torch": torch, "warnings": _warnings, "logger": logger}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_infer_device_map_from_loaded_model",
            "_attach_bnb_multidevice_hooks",
        }:
            exec(
                compile(
                    ast.Module(body = [node], type_ignores = []),
                    str(_find_vision()),
                    "exec",
                ),
                ns,
            )
    return ns["_infer_device_map_from_loaded_model"], ns[
        "_attach_bnb_multidevice_hooks"
    ]


class _P:
    def __init__(self, dev):
        self.device = torch.device(dev) if isinstance(dev, str) else dev


class _FakeMod:
    def __init__(self, params = None):
        self._p = list(params or [])
        self.hf_device_map = None

    def named_parameters(self, recurse = True, remove_duplicate = False):
        for n, d in self._p:
            yield n, _P(d)

    def parameters(self, recurse = True):
        for _, p in self.named_parameters(recurse = recurse):
            yield p

    def named_buffers(self, recurse = True):
        return iter([])

    def named_children(self):
        return iter([])


def test_successful_dispatch_does_not_emit_misleading_warning_when_logger_raises(monkeypatch):
    """When dispatch_model succeeds but the user's logger.info raises
    (broken handler / strict test harness), the helper must NOT emit the
    'Could not attach multi-device dispatch hooks automatically' warning,
    because the hooks *are* installed."""
    import accelerate

    dispatch_called = {"n": 0}
    monkeypatch.setattr(
        accelerate,
        "dispatch_model",
        lambda *a, **kw: dispatch_called.__setitem__("n", dispatch_called["n"] + 1),
    )
    _, attach = _load_fns(_RaisingLogger())
    m = _FakeMod(params = [("w", "cuda:1")])

    with _warnings.catch_warnings(record = True) as caught:
        _warnings.simplefilter("always")
        try:
            attach(
                m,
                load_in_4bit = True,
                load_in_8bit = False,
                offload_embedding = False,
                fast_inference = False,
            )
        except RuntimeError:
            # Post-fix: logger error may propagate (hooks are installed;
            # surfacing a real logging misconfiguration is acceptable).
            pass

    assert dispatch_called["n"] == 1, "dispatch_model must have been called"
    misleading = [
        w
        for w in caught
        if "Could not attach multi-device dispatch hooks" in str(w.message)
    ]
    assert not misleading, (
        f"Must not emit the 'Could not attach' warning when dispatch "
        f"actually succeeded (got {[str(w.message) for w in misleading]})"
    )
