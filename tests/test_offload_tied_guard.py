"""Tests _embeddings_are_tied in vision.py: offload_embedding must detect a shared
embed_tokens/lm_head weight so the loader can refuse to offload tied embeddings
(offloading would strand the output projection on CPU). No GPU needed."""

import ast, os
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_fn():
    src = open(VISION).read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_embeddings_are_tied":
            ns = {"torch": torch}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_embeddings_are_tied"]
    raise AssertionError("_embeddings_are_tied not found in vision.py")


tied = _load_fn()


def test_untied_separate_weights():
    emb = nn.Embedding(32, 8)
    lm = nn.Linear(8, 32, bias = False)
    assert tied(emb, lm) is False


def test_tied_shared_parameter():
    emb = nn.Embedding(32, 8)
    lm = nn.Linear(8, 32, bias = False)
    lm.weight = emb.weight  # transformers-style weight tying
    assert tied(emb, lm) is True


def test_tied_by_storage_even_if_distinct_parameter():
    emb = nn.Embedding(32, 8)
    lm = nn.Linear(8, 32, bias = False)
    lm.weight = nn.Parameter(emb.weight.detach())  # distinct Parameter, shared storage
    assert tied(emb, lm) is True


def test_none_output_is_untied():
    emb = nn.Embedding(32, 8)
    assert tied(emb, None) is False
    assert tied(None, nn.Linear(8, 32)) is False


if __name__ == "__main__":
    test_untied_separate_weights()
    print("[PASS] untied separate weights -> False")
    test_tied_shared_parameter()
    print("[PASS] tied shared parameter -> True")
    test_tied_by_storage_even_if_distinct_parameter()
    print("[PASS] tied by storage -> True")
    test_none_output_is_untied()
    print("[PASS] missing lm_head -> untied (safe to offload)")
    print("OK: tied embeddings are detected so offload_embedding can refuse them")
