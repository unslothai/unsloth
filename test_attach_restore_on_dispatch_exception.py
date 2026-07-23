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


class _TrackMod:
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


def test_attach_restores_is_hf_initialized_after_dispatch_raises(monkeypatch):
    """If dispatch_model raises, the inner finally must still restore the
    stripped _is_hf_initialized attribute on every param."""
    import accelerate

    def boom(*a, **kw):
        raise RuntimeError("dispatch blew up")

    monkeypatch.setattr(accelerate, "dispatch_model", boom)
    _, attach = _load_fns()
    m = _TrackMod([("w", "cuda:1")])
    p = next(iter(m._p))[1]
    p._is_hf_initialized = True
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        attach(
            m,
            load_in_4bit = True,
            load_in_8bit = False,
            offload_embedding = False,
            fast_inference = False,
        )
    assert p.__dict__.get("_is_hf_initialized") is True
