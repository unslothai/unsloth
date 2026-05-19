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
        self.device = dev if isinstance(dev, torch.device) else torch.device(dev)


class _FakeMod:
    def __init__(self, params = None, buffers = None, children = None, hf_device_map = None):
        self._p = list(params or [])
        self._b = list(buffers or [])
        self._c = list(children or [])
        self.hf_device_map = hf_device_map

    def named_parameters(self, recurse = True, remove_duplicate = False):
        for n, d in self._p:
            yield n, _P(d)
        if recurse:
            for cn, cm in self._c:
                for pn, pp in cm.named_parameters(
                    recurse = True, remove_duplicate = remove_duplicate
                ):
                    yield f"{cn}.{pn}", pp

    def parameters(self, recurse = True):
        for _, p in self.named_parameters(recurse = recurse):
            yield p

    def named_buffers(self, recurse = True):
        for n, d in self._b:
            yield n, _P(d)
        if recurse:
            for cn, cm in self._c:
                for bn, bb in cm.named_buffers(recurse = True):
                    yield f"{cn}.{bn}", bb

    def named_children(self):
        yield from self._c


def test_infer_handles_xpu_device_recursion():
    """XPU / custom-device types must recurse and be assigned correctly. The
    infer function is device-type agnostic and should treat different xpu
    indices as distinct devices for map-building purposes."""
    infer, _ = _load_fns()
    # xpu:0 and xpu:1 — avoids any cuda-specific codepath in infer
    a = _FakeMod(params = [("w", torch.device("xpu", 0))])
    b = _FakeMod(params = [("w", torch.device("xpu", 1))])
    root = _FakeMod(children = [("a", a), ("b", b)])
    dm = infer(root)
    assert dm.get("a") == torch.device("xpu", 0)
    assert dm.get("b") == torch.device("xpu", 1)
    assert "" not in dm
