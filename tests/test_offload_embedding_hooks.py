"""Tests _install_offload_embedding_hooks in vision.py: the offloaded lookup must work
whether the weight is on CPU or (bf16) pulled back on GPU. CUDA cases skip without a GPU."""

import ast, os
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_installer():
    src = open(VISION).read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_install_offload_embedding_hooks":
            ns = {"torch": torch}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_install_offload_embedding_hooks"]
    raise AssertionError("_install_offload_embedding_hooks not found in vision.py")


install = _load_installer()


def _fresh_emb():
    return nn.Embedding(32, 8)


def test_install_and_idempotent():
    emb = _fresh_emb()
    assert install(emb) is True
    assert emb._unsloth_offload_hooks_installed is True
    n_pre = len(emb._forward_pre_hooks)
    n_post = len(emb._forward_hooks)
    assert install(emb) is True
    assert len(emb._forward_pre_hooks) == n_pre and len(emb._forward_hooks) == n_post
    assert install(None) is False


def test_cpu_noop_forward():
    # cpu weight + cpu input -> pre-hook no-op.
    emb = _fresh_emb()
    install(emb)
    x = torch.randint(0, 32, (2, 5))
    out = emb(x)
    assert out.shape == (2, 5, 8)
    assert out.device.type == "cpu"


def test_cuda_offloaded_weight_roundtrip():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # BUG 1: CPU weight, CUDA input -> must not raise; output back on CUDA.
    emb = _fresh_emb().to("cpu")
    install(emb)
    x = torch.randint(0, 32, (2, 5), device = "cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device


def test_cuda_weight_pulled_back_to_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # BUG 2: CUDA weight (bf16 pulled back) + CUDA input -> no-op (hard-coded .to('cpu') would crash).
    emb = _fresh_emb().to("cuda")
    install(emb)
    x = torch.randint(0, 32, (2, 5), device = "cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device


if __name__ == "__main__":
    test_install_and_idempotent()
    print("[PASS] install + idempotent")
    test_cpu_noop_forward()
    print("[PASS] cpu no-op forward")
    test_cuda_offloaded_weight_roundtrip()
    print("[PASS] cuda offloaded-weight roundtrip (bug 1)")
    test_cuda_weight_pulled_back_to_gpu()
    print("[PASS] cuda weight-on-gpu no-op (bug 2)")
    print("OK: offload embedding hooks are device-safe both directions")
