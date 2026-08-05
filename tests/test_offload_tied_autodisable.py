"""Tests _resolve_offload_embedding in vision.py.

`offload_embedding = True` on a model with tied word embeddings used to raise
NotImplementedError and abort the load. It is a VRAM optimisation, not a
correctness switch, so a model it cannot help should turn it off and carry on
-- which is already what the fast_inference case does a few lines earlier.

Two shipped notebooks (NeMo-Gym-Sudoku, NeMo-Gym-Multi-Environment) pass
offload_embedding = True against unsloth/Qwen2.5-1.5B-Instruct, which ties its
embeddings, and died at model load because of the raise.

No GPU needed.
"""

import ast, os
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


_NS = _load("_embeddings_are_tied", "_resolve_offload_embedding")
resolve = _NS["_resolve_offload_embedding"]


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
    assert resolve(_untied_model(), False) is False
    assert resolve(_tied_model(), False) is False


def test_untied_model_keeps_offload():
    assert resolve(_untied_model(), True) is True


def test_tied_model_disables_offload_instead_of_raising():
    assert resolve(_tied_model(), True) is False


def test_opaque_model_leaves_request_alone():
    # Cannot inspect it, so do not guess -- and above all do not crash.
    assert resolve(_Opaque(), True) is True


def test_wsl_and_windows_are_untouched():
    # These platforms skip the offload block entirely; probing embeddings there
    # would be a code path that never used to run.
    for var in ("WSL_DISTRO_NAME", "WSL_INTEROP"):
        old = os.environ.get(var)
        os.environ[var] = "1"
        try:
            assert resolve(_tied_model(), True) is True
            assert resolve(_Opaque(), True) is True
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old

    old_name = os.name
    try:
        os.name = "nt"
        assert resolve(_tied_model(), True) is True
    finally:
        os.name = old_name


def test_resolved_before_multidevice_hooks():
    # _attach_bnb_multidevice_hooks returns early while offload_embedding is
    # still True, so resolving after it would silently skip hook attachment.
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
