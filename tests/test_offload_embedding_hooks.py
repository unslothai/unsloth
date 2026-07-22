"""Tests _install_offload_embedding_hooks in vision.py: the offloaded lookup must work and
its output must land on the decoder device, read live from the output embeddings (lm_head)
so it tracks model.to() moves. CUDA cases skip without a GPU."""

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


def _emb():
    return nn.Embedding(32, 8)


def _lm_head(device):
    # Stand-in decoder reference (untied lm_head) whose weight device is the target.
    return nn.Linear(8, 32, bias = False).to(device)


def test_install_and_idempotent():
    emb = _emb()
    lm = _lm_head(CPU)
    assert install(emb, lm, CPU) is True
    assert emb._unsloth_offload_hooks_installed is True
    n_pre = len(emb._forward_pre_hooks)
    n_post = len(emb._forward_hooks)
    assert install(emb, lm, CPU) is True
    assert len(emb._forward_pre_hooks) == n_pre and len(emb._forward_hooks) == n_post
    assert install(None, lm, CPU) is False


def test_cpu_noop_forward():
    # cpu weight + cpu decoder + cpu input -> output stays cpu.
    emb = _emb()
    install(emb, _lm_head(CPU), CPU)
    out = emb(torch.randint(0, 32, (2, 5)))
    assert out.shape == (2, 5, 8)
    assert out.device.type == "cpu"


def test_cuda_input_roundtrip():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # CPU weight, CUDA decoder + input -> lookup on cpu, output back on cuda.
    emb = _emb().to("cpu")
    install(emb, _lm_head("cuda"), torch.device("cuda"))
    out = emb(torch.randint(0, 32, (2, 5), device = "cuda"))
    assert out.device.type == "cuda", out.device


def test_cpu_input_still_returns_to_decoder():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # P1: offload makes the input arrive on cpu; the output must still reach the cuda decoder.
    emb = _emb().to("cpu")
    install(emb, _lm_head("cuda"), torch.device("cuda"))
    out = emb(torch.randint(0, 32, (2, 5), device = "cpu"))
    assert out.device.type == "cuda", out.device


def test_live_decoder_over_stale_fallback():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # P2: fallback captured as cpu (model loaded on cpu), but the decoder later lives on cuda.
    # The output must follow the live lm_head device, not the stale cpu fallback.
    emb = _emb().to("cpu")
    install(emb, _lm_head("cuda"), CPU)
    out = emb(torch.randint(0, 32, (2, 5), device = "cuda"))
    assert out.device.type == "cuda", out.device


def test_meta_lm_head_falls_back():
    # A disk-offloaded (meta) lm_head must not be used as the return device: moving hidden
    # states to meta is unrecoverable, so fall back to the captured device. No GPU needed.
    emb = _emb().to("cpu")
    lm = _lm_head(CPU)
    lm.weight = nn.Parameter(lm.weight.to("meta"))
    install(emb, lm, CPU)
    out = emb(torch.randint(0, 32, (2, 5)))
    assert out.device.type == "cpu", out.device


def test_cuda_weight_pulled_back_to_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    # bf16 weight later pulled back to gpu + cuda input -> no-op, stays on cuda.
    emb = _emb().to("cuda")
    install(emb, _lm_head("cuda"), torch.device("cuda"))
    out = emb(torch.randint(0, 32, (2, 5), device = "cuda"))
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
    test_live_decoder_over_stale_fallback()
    print("[PASS] live decoder device beats stale fallback (P2)")
    test_meta_lm_head_falls_back()
    print("[PASS] meta lm_head falls back to captured device (P2)")
    test_cuda_weight_pulled_back_to_gpu()
    print("[PASS] cuda weight-on-gpu no-op")
    print("OK: offloaded embedding output always lands on the live decoder device")
