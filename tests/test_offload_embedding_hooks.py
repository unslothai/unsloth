"""Tests _install_offload_embedding_hooks in vision.py: the offloaded lookup must work and
its output must land on the decoder device (return_device), whatever device the input is on.
CUDA cases skip without a GPU."""

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
CPU = torch.device("cpu")


def _fresh_emb():
    return nn.Embedding(32, 8)


def test_install_and_idempotent():
    emb = _fresh_emb()
    assert install(emb, CPU) is True
    assert emb._unsloth_offload_hooks_installed is True
    n_pre = len(emb._forward_pre_hooks)
    n_post = len(emb._forward_hooks)
    assert install(emb, CPU) is True
    assert len(emb._forward_pre_hooks) == n_pre and len(emb._forward_hooks) == n_post
    assert install(None, CPU) is False


def test_cpu_noop_forward():
    # cpu weight + cpu input + cpu return -> pre-hook no-op, output stays cpu.
    emb = _fresh_emb()
    install(emb, CPU)
    x = torch.randint(0, 32, (2, 5))
    out = emb(x)
    assert out.shape == (2, 5, 8)
    assert out.device.type == "cpu"


def test_cuda_input_roundtrip():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # CPU weight, CUDA input -> lookup on cpu, output back on the cuda decoder.
    emb = _fresh_emb().to("cpu")
    install(emb, torch.device("cuda"))
    x = torch.randint(0, 32, (2, 5), device = "cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device


def test_cpu_input_still_returns_to_decoder():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # P1: offload makes model.device cpu, so the input arrives on cpu too. The output must
    # still reach the cuda decoder, not stay on cpu.
    emb = _fresh_emb().to("cpu")
    install(emb, torch.device("cuda"))
    x = torch.randint(0, 32, (2, 5), device = "cpu")
    out = emb(x)
    assert out.device.type == "cuda", out.device


def test_cuda_weight_pulled_back_to_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # bf16 weight later pulled back to gpu + cuda input -> no-op, stays on cuda.
    emb = _fresh_emb().to("cuda")
    install(emb, torch.device("cuda"))
    x = torch.randint(0, 32, (2, 5), device = "cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device


if __name__ == "__main__":
    test_install_and_idempotent()
    print("[PASS] install + idempotent")
    test_cpu_noop_forward()
    print("[PASS] cpu no-op forward")
    test_cuda_input_roundtrip()
    print("[PASS] cuda input roundtrip")
    test_cpu_input_still_returns_to_decoder()
    print("[PASS] cpu input still returns to cuda decoder (P1)")
    test_cuda_weight_pulled_back_to_gpu()
    print("[PASS] cuda weight-on-gpu no-op")
    print("OK: offloaded embedding output always lands on the decoder device")
