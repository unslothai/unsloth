import ast
from pathlib import Path
import logging
import warnings as _warnings
import torch


def _find_vision():
    for p in [
        Path(__file__).resolve().parent / "unsloth" / "models" / "vision.py",
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py",
        Path(
            "/mnt/disks/unslothai/ubuntu/workspace_25/github_review/unsloth-pr-5053-staging-3/unsloth/models/vision.py"
        ),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("vision.py not found")


def _load_fns():
    tree = ast.parse(_find_vision().read_text())
    ns = {"torch": torch, "warnings": _warnings, "logger": logging.getLogger("test")}
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


class _ObservableMod:
    """Tracks whether any mutation happened on its parameters."""

    def __init__(self, params):
        self._p = [(n, _P(d)) for n, d in params]
        self.hf_device_map = None

    def named_parameters(self, recurse = True, remove_duplicate = False):
        for n, p in self._p:
            yield n, p

    def parameters(self, recurse = True):
        for _, p in self._p:
            yield p

    def named_buffers(self, recurse = True):
        return iter([])

    def named_children(self):
        return iter([])


def test_early_exit_does_not_strip_params(monkeypatch):
    """When a guard triggers (fast_inference=True), the helper must return
    before touching any parameter attributes (strip loop is never entered)."""
    import accelerate

    monkeypatch.setattr(accelerate, "dispatch_model", lambda *a, **kw: None)
    _, attach = _load_fns()
    m = _ObservableMod([("w", "cuda:1")])
    p = m._p[0][1]
    p._is_hf_initialized = "original"
    p._other_attr = "keep"
    attach(
        m,
        load_in_4bit = True,
        load_in_8bit = False,
        offload_embedding = False,
        fast_inference = True,
    )
    assert p._is_hf_initialized == "original"
    assert p._other_attr == "keep"
