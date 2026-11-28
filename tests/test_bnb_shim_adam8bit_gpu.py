# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The bitsandbytes stand-in's 8-bit Adam / AdamW on a GPU: the Triton kernels against the torch
path and a float64 oracle, against an installed bitsandbytes bit for bit (optimizer checkpoints
handed over in both directions), one compiled variant per weight dtype, CUDA graph capture,
torch.compile, and bfloat16 stochastic rounding on top. Fresh interpreters, installer by path."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA or ROCm GPU")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_INIT = os.path.join(REPO, "unsloth", "bnb_shim", "__init__.py")
NVIDIA = torch.version.hip is None


def run(
    body,
    shim = True,
    argv = (),
    extra_env = None,
):
    env = dict(os.environ)
    env["UNSLOTH_BNB_SHIM"] = "1" if shim else "0"
    env["UNSLOTH_BNB_SHIM_QUIET"] = "1"
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    prelude = ""
    if shim:
        prelude = textwrap.dedent(
            f"""
            import importlib.util
            spec = importlib.util.spec_from_file_location("_unsloth_bnb_shim", {SHIM_INIT!r})
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            assert shim.install()
            """
        )
    code = prelude + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
        cwd = REPO,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    lines = [l for l in proc.stdout.splitlines() if l.startswith("OUT ")]
    assert lines, proc.stdout + proc.stderr
    return json.loads(lines[-1][4:])


# One training run: parameters on every path (8-bit with block tails, 32-bit under min_8bit_size,
# an optim_bits=32 override, three weight dtypes, a group without weight decay), synthetic
# gradients that depend on the weights, an lr schedule; optional checkpoint save / resume.
TRAIN = """
import json, sys, torch
import bitsandbytes as bnb
cls_name, out_path, save_path, resume_path, steps, save_at = sys.argv[1:7]
steps, save_at = int(steps), int(save_at)
dev = "cuda"
SPECS = [("emb", (300, 64), torch.float32), ("w1", (300, 257), torch.bfloat16),
         ("w2", (4097,), torch.float16), ("w3", (64, 64), torch.float32), ("b1", (300,), torch.bfloat16)]
gen = torch.Generator(device = dev).manual_seed(1234)
params = {n: torch.nn.Parameter((torch.randn(s, generator = gen, device = dev) * 0.1).to(d)) for n, s, d in SPECS}
emb = torch.nn.Embedding(300, 64, device = dev)
emb.weight = params["emb"]
bnb.optim.GlobalOptimManager.get_instance().register_module_override(emb, "weight", {"optim_bits": 32})
groups = [{"params": [params[n] for n in ("emb", "w1", "w2", "w3")], "weight_decay": 0.01},
          {"params": [params["b1"]], "weight_decay": 0.0}]
if cls_name == "hf":  # transformers' optim="adamw_8bit"
    opt = bnb.optim.AdamW(groups, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, optim_bits = 8, is_paged = False)
else:
    opt = getattr(bnb.optim, cls_name)(groups, lr = 1e-3, betas = (0.9, 0.995), eps = 1e-8)
start = 1
if resume_path != "-":
    ck = torch.load(resume_path, weights_only = False)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(ck["params"][n])
    opt.load_state_dict(ck["opt"])
    start = save_at + 1
for t in range(start, steps + 1):
    for gi, group in enumerate(opt.param_groups):
        group["lr"] = 1e-3 * min(1.0, t / 5) * (1.0 if gi == 0 else 2.0) * (1 - t / (steps + 1))
    g = torch.Generator(device = dev).manual_seed(10_000 + t)
    for n, s, d in SPECS:
        p = params[n]
        p.grad = (torch.sin(3.0 * p.detach().float()) + 0.3 * torch.randn(s, generator = g, device = dev)).to(d)
    opt.step()
    if save_path != "-" and t == save_at:
        torch.save({"params": {n: p.detach().cpu() for n, p in params.items()}, "opt": opt.state_dict()}, save_path)
state = {n: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()} for n, p in params.items()}
torch.save({"params": {n: p.detach().cpu() for n, p in params.items()}, "state": state}, out_path)
print("OUT " + json.dumps({
    "shim": bool(getattr(bnb, "__unsloth_shim__", False)),
    "kinds": {n: str(state[n]["state1"].dtype) for n in state},
}))
"""


def _bnb_usable():
    probe = (
        "import bitsandbytes, torch;"
        "p = torch.nn.Parameter(torch.zeros(4096, device='cuda')); p.grad = torch.ones_like(p);"
        "bitsandbytes.optim.AdamW8bit([p]).step(); torch.cuda.synchronize(); print('OK')"
    )
    env = dict(os.environ, UNSLOTH_BNB_SHIM = "0")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output = True, text = True, env = env)
    return proc.returncode == 0 and "OK" in proc.stdout


def _bits(x):
    if x.dtype.is_floating_point:
        return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])
    return x


def _assert_same(a, b):
    for n in a["params"]:
        assert torch.equal(_bits(a["params"][n]), _bits(b["params"][n])), n
        for k, v in a["state"][n].items():
            w = b["state"][n][k]
            if torch.is_tensor(v):
                assert torch.equal(_bits(v), _bits(w)), (n, k)
            else:
                assert int(v) == int(w), (n, k)


def test_stand_in_states_and_paths(tmp_path):
    out = run(TRAIN, argv = ("AdamW8bit", str(tmp_path / "a.pt"), "-", "-", "12", "0"))
    assert out["shim"] is True
    # >= 4096 elements: 8-bit; below it, and the optim_bits=32 embedding override: 32-bit.
    assert out["kinds"] == {
        "emb": "torch.float32",
        "w1": "torch.uint8",
        "w2": "torch.uint8",
        "w3": "torch.uint8",
        "b1": "torch.float32",
    }


@pytest.mark.skipif(not NVIDIA, reason = "bit-exactness is the NVIDIA PTX path")
def test_bit_identical_to_installed_bitsandbytes_and_checkpoints_cross(tmp_path):
    if not _bnb_usable():
        pytest.skip("no working bitsandbytes installed")
    load = lambda p: torch.load(p, weights_only = False)
    for cls in ("AdamW8bit", "hf"):  # the class, and transformers' AdamW(optim_bits=8) route
        real, mine = tmp_path / f"real_{cls}.pt", tmp_path / f"shim_{cls}.pt"
        real_ck, mine_ck = tmp_path / f"real_{cls}_ck.pt", tmp_path / f"shim_{cls}_ck.pt"
        assert not run(TRAIN, shim = False, argv = (cls, str(real), str(real_ck), "-", "30", "15"))[
            "shim"
        ]
        assert run(TRAIN, argv = (cls, str(mine), str(mine_ck), "-", "30", "15"))["shim"]
        _assert_same(load(real), load(mine))
    # stand-in checkpoint resumed by bitsandbytes, and the other way round
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    run(
        TRAIN,
        shim = False,
        argv = ("AdamW8bit", str(a), "-", str(tmp_path / "shim_AdamW8bit_ck.pt"), "30", "15"),
    )
    run(TRAIN, argv = ("AdamW8bit", str(b), "-", str(tmp_path / "real_AdamW8bit_ck.pt"), "30", "15"))
    _assert_same(load(tmp_path / "real_AdamW8bit.pt"), load(a))
    _assert_same(load(tmp_path / "real_AdamW8bit.pt"), load(b))


KERNEL_VS_TORCH = """
import json, torch
import bitsandbytes as bnb
from bitsandbytes.optim._adam_torch import adam_8bit_blockwise_torch, adam_32bit_torch
from unsloth.kernels import adam8bit_blockwise as K
dev = "cuda"
out = {}
def ulps(a, b, prev):
    # |a - b| beyond 5e-4 of the step, in units of b's spacing in the weight dtype. The step itself
    # differs that much early on: bitsandbytes' fast-math powf (ex2.approx / lg2.approx) in
    # 1 - beta2**t loses about 1e-4 relative to the cancellation, which the kernels reproduce.
    ity = torch.int32 if a.dtype == torch.float32 else torch.int16
    mag = b.abs()
    up = (mag.view(ity) + 1).view(a.dtype)
    spacing = (up.float() - mag.float()).clamp_min(torch.finfo(a.dtype).tiny)
    excess = ((a.float() - b.float()).abs() - 5e-4 * (b.float() - prev.float()).abs()).clamp_min(0)
    return (excess / spacing).max().item()
code1 = bnb.functional.create_dynamic_map(signed = True).to(dev)
code2 = bnb.functional.create_dynamic_map(signed = False).to(dev)
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    for n in (1, 255, 257, 4097, 65549):
        gen = torch.Generator(device = dev).manual_seed(n)
        p0 = torch.randn(n, generator = gen, device = dev).to(dtype)
        blocks = -(n // -256)
        tri = dict(p = p0.clone(), s1 = torch.zeros(n, dtype = torch.uint8, device = dev),
                   s2 = torch.zeros(n, dtype = torch.uint8, device = dev),
                   a1 = torch.zeros(blocks, device = dev), a2 = torch.zeros(blocks, device = dev))
        ref = {k: v.clone() for k, v in tri.items()}
        f64 = {k: v.clone() for k, v in tri.items()}
        worst = dict(p_ulps = 0.0, code = 0, absmax = 0.0, oracle_ulps = 0.0)
        for t in range(1, 9):
            g = (torch.randn(n, generator = gen, device = dev) * 0.01).to(dtype)
            if n > 8 and t == 4:
                g[3] = float("nan")
            for st, fn, cd in ((tri, K.adam_8bit_blockwise_step, None),
                               (ref, adam_8bit_blockwise_torch, torch.float32),
                               (f64, adam_8bit_blockwise_torch, torch.float64)):
                # the oracle steps from the Triton state, so each step is checked on its own
                if cd is torch.float64:
                    for k in st:
                        st[k].copy_(tri_prev[k])
                kw = {} if cd is None else {"compute_dtype": cd}
                if cd is None:
                    tri_prev = {k: v.clone() for k, v in st.items()}
                fn(st["p"], g, st["s1"], st["s2"], code1, code2, st["a1"], st["a2"], t, 1e-3, 0.9, 0.999, 1e-8, 0.01, **kw)
            worst["p_ulps"] = max(worst["p_ulps"], ulps(tri["p"], ref["p"], tri_prev["p"]))
            worst["oracle_ulps"] = max(worst["oracle_ulps"], ulps(tri["p"], f64["p"], tri_prev["p"]))
            worst["code"] = max(worst["code"], (tri["s1"].int() - ref["s1"].int()).abs().max().item(),
                                (tri["s2"].int() - ref["s2"].int()).abs().max().item())
            worst["absmax"] = max(worst["absmax"], ((tri["a1"] - ref["a1"]).abs() / ref["a1"].clamp_min(1e-30)).max().item())
            # the torch path follows the Triton one (identical inputs each step)
            for k in ref:
                ref[k].copy_(tri[k])
        out[f"{dtype}-{n}"] = worst
print("OUT " + json.dumps(out))
"""


def test_triton_matches_torch_path_and_float64_oracle():
    out = run(KERNEL_VS_TORCH)
    for case, w in out.items():
        # fast-math vs IEEE float32: the step agrees to 5e-4 of itself and the weight, rounded twice
        # (step, then decay), is at most two representable values further; codes one step
        assert w["p_ulps"] <= 2 and w["oracle_ulps"] <= 2, (case, w)
        assert w["code"] <= 1 and w["absmax"] <= 1e-5, (case, w)


VARIANTS_GRAPHS_COMPILE = """
import json, os, torch
os.environ["UNSLOTH_BNB_SHIM_MULTI_TENSOR"] = "0"  # the per-tensor kernel (multi-tensor: own test)
import bitsandbytes as bnb
from unsloth.kernels import adam8bit_blockwise as K
dev = "cuda"
out = {}
def build(n, dtype, capturable):
    p = torch.nn.Parameter(torch.randn(n, device = dev).to(dtype))
    return p, bnb.optim.AdamW8bit([p], lr = 1e-3, capturable = capturable)
for dtype in (torch.bfloat16, torch.float32):
    for n in (4097, 65549, 300001, 1 << 20):
        p, opt = build(n, dtype, False)
        for t in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
out["variants"] = sum(len(c[0]) for c in K._adam8bit_blockwise_kernel.device_caches.values())
# CUDA graph: capture one capturable step, replay; equals the same steps run eagerly.
torch.manual_seed(0)
grads = [torch.randn(70001, device = dev) for _ in range(6)]
pe, oe = build(70001, torch.float32, True)
pg, og = build(70001, torch.float32, True)
with torch.no_grad():
    pg.copy_(pe)
for t in range(6):
    pe.grad = grads[t].clone()
    oe.step()
static = grads[0].clone()
pg.grad = static
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    og.step()  # warm-up (step 1) outside the graph
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    og.step()
for t in range(1, 6):
    static.copy_(grads[t])
    graph.replay()
torch.cuda.synchronize()
se, sg = oe.state[pe], og.state[pg]
out["graph_equal"] = bool(torch.equal(pe, pg) and torch.equal(se["state1"], sg["state1"])
                          and torch.equal(se["absmax2"], sg["absmax2"]))
out["graph_step"] = float(sg["step"]) == float(se["step"]) == 6.0
out["state_dict_step"] = og.state_dict()["state"][0]["step"]
# torch.compile: the step runs as one opaque op, no graph break, same bits as eager.
n = 4097
def fresh():
    g = torch.Generator(device = dev).manual_seed(7)
    p = torch.randn(n, generator = g, device = dev)
    return dict(p = p, g = torch.randn(n, generator = g, device = dev),
                s1 = torch.zeros(n, dtype = torch.uint8, device = dev), s2 = torch.zeros(n, dtype = torch.uint8, device = dev),
                a1 = torch.zeros(17, device = dev), a2 = torch.zeros(17, device = dev))
code1 = bnb.functional.create_dynamic_map(signed = True).to(dev)
code2 = bnb.functional.create_dynamic_map(signed = False).to(dev)
def stepper(d, t):
    K.compiled_safe_8bit_step(d["p"], d["g"], d["s1"], d["s2"], code1, code2, d["a1"], d["a2"], t, 1e-3, 0.9, 0.999, 1e-8, 0.0)
    return d["p"].sum()
eager, comp = fresh(), fresh()
torch._dynamo.reset()
compiled = torch.compile(stepper, fullgraph = True)
for t in range(1, 4):
    stepper(eager, t)
    compiled(comp, t)
out["compiled_equal"] = all(torch.equal(eager[k], comp[k]) for k in eager)
print("OUT " + json.dumps(out))
"""


def test_one_variant_cuda_graphs_and_compile():
    out = run(VARIANTS_GRAPHS_COMPILE)
    # one compiled kernel per weight dtype across sizes (bf16, fp32 here)
    assert out["variants"] == 2, out
    assert out["graph_equal"] and out["graph_step"], out
    assert out["state_dict_step"] == 6
    assert out["compiled_equal"], out


SR = """
import json, torch
import bitsandbytes as bnb
from unsloth.optimizers.bf16_stochastic_rounding import enable_bf16_stochastic_rounding
p = torch.nn.Parameter(torch.randn(8192, device = "cuda").to(torch.bfloat16))
opt = bnb.optim.AdamW8bit([p], lr = 1e-3)
covered = enable_bf16_stochastic_rounding(opt, verbose = False)
for _ in range(3):
    p.grad = torch.randn_like(p)
    opt.step()
print("OUT " + json.dumps({"covered": covered, "fused": getattr(opt, "_unsloth_sr_bnb_fused_steps", 0),
                           "codes": str(opt.state[p]["state1"].dtype)}))
"""


def test_bf16_stochastic_rounding_drives_the_stand_in():
    out = run(SR)
    # The rounding wrapper sees a bitsandbytes Adam with 8-bit state and steps it in one kernel.
    assert out["covered"] == 1 and out["codes"] == "torch.uint8"
    assert out["fused"] == 3, out


MULTI_TENSOR = """
import json, os, sys, torch
import bitsandbytes as bnb
from unsloth.kernels import adam8bit_blockwise as K
dev = "cuda"
SPECS = [((300, 257), torch.bfloat16), ((4097,), torch.bfloat16), ((64, 64), torch.float32), ((300,), torch.float32),
         ((5000,), torch.float16), ((1,), torch.float32), ((70001,), torch.float32)]


def run(multi):
    os.environ["UNSLOTH_BNB_SHIM_MULTI_TENSOR"] = "1" if multi else "0"
    g0 = torch.Generator(device = dev).manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(s, generator = g0, device = dev).to(d)) for s, d in SPECS]
    opt = bnb.optim.AdamW8bit([{"params": ps[:4], "weight_decay": 0.1}, {"params": ps[4:], "lr": 3e-3, "weight_decay": 0.0}],
                              lr = 1e-3)
    for t in range(1, 13):
        for i, p in enumerate(ps):
            # some weights miss a gradient on some steps; the bf16 launch starts with one weight and
            # then grows (an int argument equal to 1 must not stay specialized)
            skip = (i == 1 and t % 4 == 0) or (i == 0 and t == 1)
            p.grad = None if skip else torch.randn(p.shape, generator = g0, device = dev).to(p.dtype)
        if t == 7:  # resume mid-run from its own checkpoint
            opt.load_state_dict(opt.state_dict())
        opt.param_groups[0]["lr"] = 1e-3 * (1 - t / 13)
        opt.step()
    torch.cuda.synchronize()
    return ps, opt


pa, oa = run(True)
pb, ob = run(False)
same = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(pa, pb))
for a, b in zip(pa, pb):
    for k, v in oa.state[a].items():
        w = ob.state[b][k]
        same = same and (torch.equal(v, w) if torch.is_tensor(v) else v == w)
# a weight listed twice, and two weights sharing memory: the per-tensor route (sequential) steps them
os.environ["UNSLOTH_BNB_SHIM_MULTI_TENSOR"] = "1"
fallback = []
buf = torch.zeros(10000, device = dev)
for params in ([torch.nn.Parameter(torch.zeros(5000, device = dev))] * 2,
               [torch.nn.Parameter(buf[:6000]), torch.nn.Parameter(buf[4000:])]):
    o = bnb.optim.AdamW8bit([{"params": params}])
    for q in params:
        q.grad = torch.ones_like(q)
    o.step()
    fallback.append(o.__dict__.get("_unsloth_multi_tensor_steps", 0))
variants = sum(len(c[0]) for c in K._adam8bit_multi_kernel.device_caches.values())
variants32 = sum(len(c[0]) for c in K._adam32bit_multi_kernel.device_caches.values())
print("OUT " + json.dumps({"same": same, "multi_steps": oa.__dict__.get("_unsloth_multi_tensor_steps", 0),
                           "single_steps": ob.__dict__.get("_unsloth_multi_tensor_steps", 0),
                           "variants": variants, "variants32": variants32, "fallback": fallback}))
"""


def test_multi_tensor_step_matches_per_tensor_step():
    out = run(MULTI_TENSOR)
    # One launch per (dtype, state kind) per step, bit-identical to one launch per weight, across
    # param groups, missing gradients and a state_dict reload.
    assert out["same"], out
    assert out["multi_steps"] == 12 and out["single_steps"] == 0, out
    assert out["fallback"] == [0, 0], out
    # 8-bit: bf16 + fp16 + fp32 weights; 32-bit: small fp32 weights; at most 2 variants per dtype
    assert out["variants"] <= 6 and out["variants32"] <= 2, out
