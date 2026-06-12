import ast
from pathlib import Path
import logging
import warnings as _warnings
import torch


def _find_vision():
    for p in [
        Path(__file__).resolve().parent / "unsloth" / "models" / "vision.py",
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py",
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


class _BufMod:
    def __init__(self, buffers = None, children = None):
        self._b = list(buffers or [])
        self._c = list(children or [])
        self.hf_device_map = None

    def named_parameters(self, recurse = True, remove_duplicate = False):
        if False:
            yield

    def parameters(self, recurse = True):
        if False:
            yield

    def named_buffers(self, recurse = True):
        for n, d in self._b:
            yield n, _P(d)
        if recurse:
            for cn, cm in self._c:
                for bn, bb in cm.named_buffers(recurse = True):
                    yield f"{cn}.{bn}", bb

    def named_children(self):
        yield from self._c


def test_buffer_only_subtree_with_multi_device_recurses():
    """Param-less subtree whose buffers span multiple devices must NOT collapse
    to the first buffer's device. It must recurse into children so each child
    gets its own map entry keyed by its own prefix."""
    infer, _ = _load_fns()
    a = _BufMod(buffers = [("cache", "cuda:0")])
    b = _BufMod(buffers = [("cache", "cuda:1")])
    root = _BufMod(children = [("a", a), ("b", b)])
    dm = infer(root)
    assert dm.get("a") == torch.device("cuda", 0), dm
    assert dm.get("b") == torch.device("cuda", 1), dm
    assert "" not in dm, dm
