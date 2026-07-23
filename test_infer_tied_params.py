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


class _TiedMod:
    """Emits the same parameter object under two different names to simulate
    tied weights (lm_head.weight == embed.weight). With remove_duplicate=False
    we yield both names; devices unioned must still be a single device."""

    def __init__(self, dev):
        self._shared = _P(dev)
        self.hf_device_map = None

    def named_parameters(self, recurse = True, remove_duplicate = False):
        yield "embed.weight", self._shared
        if not remove_duplicate:
            yield "lm_head.weight", self._shared

    def parameters(self, recurse = True):
        yield self._shared

    def named_buffers(self, recurse = True):
        return iter([])

    def named_children(self):
        return iter([])


def test_infer_tied_params_single_entry():
    """Tied-weight models (same Parameter yielded twice under different names)
    must still collapse to a single-device map entry."""
    infer, _ = _load_fns()
    m = _TiedMod("cuda:1")
    dm = infer(m)
    assert dm == {"": torch.device("cuda", 1)}
