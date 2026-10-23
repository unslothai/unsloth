"""CPU (gloo) CP correctness tests from tests/test_context_parallel.py: 2 ranks vs 1 process.

usage: CUDA_VISIBLE_DEVICES= python run_cpu_tests.py _tiny_kimi_k3 _tiny_deepseek_v32 ...
       (no names: run every builder in DEFAULT_BUILDERS)
Ranks are forked rather than spawned so torch / transformers pages are shared (fits a 2 GiB container).
Children are forked before any torch compute runs. UNSLOTH_CP_ATTN_CHUNK=4 exercises the chunked paths.
"""
import os, pathlib, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
import torch
import torch.multiprocessing as mp
import test_context_parallel as T

DEFAULT_BUILDERS = [
    "_tiny_deepseek_v2",
    "_tiny_deepseek_v32",
    "_tiny_deepseek_v4",
    "_tiny_minimax",
    "_tiny_minimax_m2",
    "_tiny_gemma4",
    "_tiny_gemma4_moe_k_eq_v",
    "_tiny_kimi_linear",
    "_tiny_kimi_k3",
]

if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT_BUILDERS:
        builder = getattr(T, name)
        seq = 16
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "cp.pt")
            ctx = mp.start_processes(
                T._generic_worker,
                args = (2, T._free_port(), seq, out, builder),
                nprocs = 2, join = False, start_method = "fork",
            )
            model = builder()
            ids, labels = T._batch(seq)
            ref = T.P.sft_loss(model, ids, labels, None)
            ref.backward()
            ref_grads = T._grads(model)
            while not ctx.join():
                pass
            got = torch.load(out)
        torch.testing.assert_close(got["loss"], ref.detach(), rtol = 1e-4, atol = 1e-5)
        assert got["grads"].keys() == ref_grads.keys(), (set(got["grads"]) ^ set(ref_grads))
        worst = 0.0
        for k, g in ref_grads.items():
            torch.testing.assert_close(got["grads"][k], g, rtol = 1e-4, atol = 1e-5, msg = k)
            worst = max(worst, (got["grads"][k] - g).abs().max().item())
        print(f"{name} PASS  loss={ref.item():.6f}  params={len(ref_grads)}  max|dgrad|={worst:.2e}", flush=True)
