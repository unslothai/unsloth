"""Tests _resolve_offload_embedding in vision.py. No GPU needed.

`offload_embedding = True` on a model with tied word embeddings used to raise
NotImplementedError and abort the load. It is a VRAM optimisation, not a
correctness switch, so it should turn itself off instead, as the fast_inference
case a few lines earlier already does. Two shipped notebooks (NeMo-Gym-Sudoku,
NeMo-Gym-Multi-Environment) died this way on unsloth/Qwen2.5-1.5B-Instruct.

Every platform branch is driven explicitly, so the assertions hold on Linux,
macOS, Windows and WSL alike: the host's own os.name never decides.
"""

import ast, os
from contextlib import contextmanager

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")

_SRC = open(VISION, encoding = "utf-8").read()


def _load(*names):
    mod = ast.parse(_SRC)
    ns = {"torch": torch, "os": os}
    wanted = set(names)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(_SRC, node), ns)
            wanted.discard(node.name)
    if wanted:
        raise AssertionError(f"not found in vision.py: {sorted(wanted)}")
    return ns


_NS = _load(
    "_embeddings_are_tied",
    "_offload_embedding_unsupported_platform",
    "_resolve_offload_embedding",
)
resolve = _NS["_resolve_offload_embedding"]
unsupported_platform = _NS["_offload_embedding_unsupported_platform"]

_WSL_VARS = ("WSL_DISTRO_NAME", "WSL_INTEROP")


@contextmanager
def _as_platform(os_name, wsl = False):
    """Drive the platform inputs directly instead of trusting the host's."""
    saved_env = {v: os.environ.get(v) for v in _WSL_VARS}
    saved_name = os.name
    for v in _WSL_VARS:
        os.environ.pop(v, None)
    if wsl:
        os.environ["WSL_DISTRO_NAME"] = "Ubuntu"
    # os.name is read by pathlib, so keep the window as small as possible.
    os.name = os_name
    try:
        yield
    finally:
        os.name = saved_name
        for v, old in saved_env.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


class _Model:
    def __init__(self, emb, out):
        self._emb, self._out = emb, out

    def get_input_embeddings(self):
        return self._emb

    def get_output_embeddings(self):
        return self._out


class _Opaque:
    """Some architectures refuse to expose embeddings."""

    def get_input_embeddings(self):
        raise NotImplementedError("no embeddings here")

    def get_output_embeddings(self):
        return None


def _tied_model():
    emb = nn.Embedding(32, 8)
    lm = nn.Linear(8, 32, bias = False)
    lm.weight = emb.weight
    return _Model(emb, lm)


def _untied_model():
    return _Model(nn.Embedding(32, 8), nn.Linear(8, 32, bias = False))


def test_disabled_stays_disabled():
    for os_name, wsl in (("posix", False), ("nt", False), ("posix", True)):
        with _as_platform(os_name, wsl = wsl):
            assert resolve(_untied_model(), False) is False
            assert resolve(_tied_model(), False) is False


def test_untied_model_keeps_offload():
    with _as_platform("posix"):
        assert resolve(_untied_model(), True) is True


def test_tied_model_disables_offload_instead_of_raising():
    with _as_platform("posix"):
        assert resolve(_tied_model(), True) is False


def test_opaque_model_leaves_request_alone():
    # Cannot inspect it, so do not guess, and do not crash.
    with _as_platform("posix"):
        assert resolve(_Opaque(), True) is True


def test_wsl_and_windows_disable_offload():
    # Neither can offload, and the flag also gates the multi-device hook attach,
    # so it has to read False rather than pass through.
    for var in _WSL_VARS:
        with _as_platform("posix"):
            os.environ[var] = "1"
            assert unsupported_platform() == "WSL"
            assert resolve(_tied_model(), True) is False
            assert resolve(_untied_model(), True) is False
            assert resolve(_Opaque(), True) is False

    with _as_platform("nt"):
        assert unsupported_platform() == "Windows"
        assert resolve(_tied_model(), True) is False
        assert resolve(_untied_model(), True) is False
        assert resolve(_Opaque(), True) is False

    with _as_platform("posix"):
        assert unsupported_platform() is None


def test_platform_gate_lives_in_one_place():
    # The offload block used to re-test os.name itself; the copies drifted apart
    # and only Windows noticed. _resolve_offload_embedding owns it now.
    helper = _SRC[_SRC.index("def _offload_embedding_unsupported_platform(") :]
    helper = helper[: helper.index("\n\n\ndef ")]
    for probe in ('os.name == "nt"', "WSL_DISTRO_NAME", "WSL_INTEROP"):
        assert _SRC.count(probe) == 1, f"{probe} must be tested in exactly one place"
        assert probe in helper, f"{probe} belongs in _offload_embedding_unsupported_platform"


def test_resolved_before_multidevice_hooks():
    # Hook attach returns early while offload_embedding is still True.
    call = _SRC.index("offload_embedding = _resolve_offload_embedding(")
    # Anchor on the indented CALL, not the module-level `def`.
    hooks = _SRC.index("\n                _attach_bnb_multidevice_hooks(")
    assert call < hooks, "offload_embedding must be resolved before hook attach"


def test_no_tied_embedding_raise_remains():
    assert "is not supported for models with tied word" not in _SRC


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all offload tied auto-disable tests passed")
