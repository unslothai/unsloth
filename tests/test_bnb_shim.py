# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Unsloth's bitsandbytes stand-in (unsloth/bnb_shim), CPU only. Each case runs in a fresh
interpreter that loads the installer by path (no `import unsloth`), so the real bitsandbytes, if
installed, never leaks between cases."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_INIT = os.path.join(REPO, "unsloth", "bnb_shim", "__init__.py")

PRELUDE = f"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("_unsloth_bnb_shim", {SHIM_INIT!r})
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
"""


def run(
    body,
    mode = "1",
    extra_env = None,
):
    env = dict(os.environ)
    env["UNSLOTH_BNB_SHIM"] = mode
    env["UNSLOTH_BNB_SHIM_QUIET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    code = PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 300
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.startswith("OUT ")]
    assert lines, proc.stdout + proc.stderr
    return json.loads(lines[-1][4:])


def test_forced_install_serves_module_and_metadata():
    out = run(
        """
        assert shim.install() is True
        import importlib.metadata as md
        import bitsandbytes
        mapping = md.packages_distributions()
        print("OUT " + json.dumps({
            "shim": bitsandbytes.__unsloth_shim__,
            "file": bitsandbytes.__file__,
            "version": md.version("bitsandbytes"),
            "dunder": bitsandbytes.__version__,
            "spec": importlib.util.find_spec("bitsandbytes") is not None,
            "mapped": "bitsandbytes" in mapping.get("bitsandbytes", []),
            "finders": [type(f).__name__ for f in sys.meta_path],
            "devices": sorted(bitsandbytes.supported_torch_devices),
        }))
        """
    )
    assert out["shim"] is True
    assert out["file"].startswith(os.path.join(REPO, "unsloth", "bnb_shim", "bitsandbytes"))
    assert out["version"] == out["dunder"] == "0.49.2"
    assert out["spec"] and out["mapped"]
    # Ahead of a real install (forced over it), else after the path finders.
    assert "_ShimFinder" in out["finders"]
    assert "cuda" in out["devices"] and "cpu" in out["devices"]


def test_off_and_working_install_are_left_alone():
    out = run(
        """
        off = shim.install()
        os.environ["UNSLOTH_BNB_SHIM"] = "auto"
        shim._has_gpu = lambda: True
        shim._real_installed = lambda: True
        shim._real_unusable = lambda: False
        auto_working = shim.install()
        finders = [type(f).__name__ for f in sys.meta_path]
        print("OUT " + json.dumps({"off": off, "auto_working": auto_working, "finders": finders}))
        """,
        mode = "0",
    )
    assert out["off"] is False and out["auto_working"] is False
    assert "_ShimFinder" not in out["finders"]


def test_auto_registers_when_missing_or_unusable():
    out = run(
        """
        shim._has_gpu = lambda: True
        shim._real_installed = lambda: False
        missing = shim.install()
        appended = type(sys.meta_path[-1]).__name__
        reason_missing = shim._STATE["reason"]
        sys.meta_path.pop()
        shim._STATE["active"] = False
        shim._real_installed = lambda: True
        shim._real_unusable = lambda: True
        broken = shim.install()
        print("OUT " + json.dumps({
            "missing": missing, "appended": appended, "reason_missing": reason_missing,
            "broken": broken, "first": type(sys.meta_path[0]).__name__, "reason": shim._STATE["reason"],
        }))
        """,
        mode = "auto",
    )
    assert out["missing"] is True and out["appended"] == "_ShimFinder"
    assert "not installed" in out["reason_missing"]
    assert out["broken"] is True and out["first"] == "_ShimFinder"
    assert "native library" in out["reason"]


def test_transformers_and_peft_see_the_stand_in():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    out = run(
        """
        shim.install()
        from transformers.utils import import_utils
        from peft.import_utils import is_bnb_4bit_available, is_bnb_available
        print("OUT " + json.dumps({
            "tf": bool(import_utils.is_bitsandbytes_available()),
            "peft": bool(is_bnb_available() and is_bnb_4bit_available()),
        }))
        """
    )
    assert out == {"tf": True, "peft": True}


def test_quant_state_packed_layout_matches_bitsandbytes():
    # The packed quant_state JSON and key names are what bnb-4bit checkpoints store.
    out = run(
        """
        shim.install()
        import torch
        from bitsandbytes import functional as F
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype = torch.bfloat16)
        q, st = F.quantize_4bit(x, blocksize = 64, quant_type = "nf4", compress_statistics = True)
        packed = st.as_dict(packed = True)
        blob = json.loads(bytes(packed["quant_state.bitsandbytes__nf4"].numpy()).decode())
        back = F.QuantState.from_dict({"weight." + k: v for k, v in packed.items()}, device = "cpu")
        print("OUT " + json.dumps({
            "keys": sorted(packed), "blob_keys": list(blob), "blob": blob,
            "q_shape": list(q.shape), "q_dtype": str(q.dtype),
            "roundtrip": bool(back == st), "nested": back.nested,
            "raw_json": bytes(packed["quant_state.bitsandbytes__nf4"].numpy()).decode(),
        }))
        """
    )
    assert out["keys"] == [
        "absmax",
        "nested_absmax",
        "nested_quant_map",
        "quant_map",
        "quant_state.bitsandbytes__nf4",
    ]
    assert out["blob_keys"] == [
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]
    assert out["blob"]["dtype"] == "bfloat16" and out["blob"]["nested_dtype"] == "float32"
    assert out["blob"]["shape"] == [4, 64] and out["blob"]["nested_blocksize"] == 256
    assert out["raw_json"].startswith('{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16"')
    assert out["q_shape"] == [128, 1] and out["q_dtype"] == "torch.uint8"
    assert out["roundtrip"] and out["nested"]


def test_quantize_codes_nibbles_and_odd_sizes():
    out = run(
        """
        shim.install()
        import torch
        from bitsandbytes import functional as F
        code = F.get_4bit_type("nf4", device = "cpu")
        # Every NF4 level scaled by an absmax of 2 must code to its own index, high nibble first.
        x = (code * 2).repeat(4)
        q, st = F.quantize_4bit(x.view(1, -1), blocksize = 64, quant_type = "nf4", compress_statistics = False)
        nibbles = torch.stack([q.view(-1) >> 4, q.view(-1) & 15], -1).reshape(-1).tolist()
        dq = F.dequantize_4bit(q, st)
        # Odd element count: the padded last nibble codes the zero bitsandbytes pads with (index 7).
        y = torch.ones(65)
        qy, sy = F.quantize_4bit(y, blocksize = 64, quant_type = "nf4")
        # An all zero block: absmax 0, every element codes 0 (NaN through the comparison tree).
        z = torch.zeros(64)
        qz, sz = F.quantize_4bit(z, blocksize = 64, quant_type = "nf4")
        r = torch.randn(3, 333, dtype = torch.float16)
        qr, sr = F.quantize_4bit(r, blocksize = 128, quant_type = "fp4", compress_statistics = True)
        err = (F.dequantize_4bit(qr, sr).float() - r.float()).abs().max().item()
        print("OUT " + json.dumps({
            "nibbles": nibbles[:16], "exact": bool(torch.equal(dq.view(-1), x)),
            "odd_last": int(qy.view(-1)[-1]), "odd_len": qy.numel(),
            "zero": qz.view(-1).tolist()[:4], "zero_absmax": float(sz.absmax[0]),
            "fp4_err": err, "fp4_scale": float(r.abs().max()),
        }))
        """
    )
    assert out["nibbles"] == list(range(16))
    assert out["exact"]
    assert out["odd_len"] == 33 and out["odd_last"] == (15 << 4) | 7
    assert out["zero"] == [0, 0, 0, 0] and out["zero_absmax"] == 0.0
    assert out["fp4_err"] < 0.3 * out["fp4_scale"]


def test_linear4bit_quantizes_on_move_and_saves_bnb_keys():
    out = run(
        """
        shim.install()
        import torch
        import bitsandbytes as bnb
        from bitsandbytes import functional as F
        torch.manual_seed(0)
        dense = torch.nn.Linear(128, 64, bias = True).to(torch.bfloat16)
        layer = bnb.nn.Linear4bit(128, 64, bias = True, compute_dtype = torch.bfloat16,
                                  compress_statistics = True, quant_type = "nf4")
        layer.load_state_dict(dense.state_dict())
        layer = layer.to("cpu")
        w = layer.weight
        module_qs = layer.quant_state is w.quant_state
        x = torch.randn(2, 5, 128, dtype = torch.bfloat16, requires_grad = True)
        y = layer(x)
        W = F.dequantize_4bit(w.data, w.quant_state).to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, W, layer.bias.to(torch.bfloat16))
        y.sum().backward()
        sd = layer.state_dict()
        # PEFT merge / accelerate rebuild a Params4bit from the live attribute dict.
        kwargs = dict(w.__dict__)
        kwargs["bnb_quantized"] = False
        kwargs.pop("data", None)
        rebuilt = bnb.nn.Params4bit(F.dequantize_4bit(w.data, w.quant_state), **kwargs).to("cpu")
        print("OUT " + json.dumps({
            "cls": type(w).__name__, "quantized": w.bnb_quantized, "module_qs": module_qs,
            "match": bool(torch.equal(y, ref)), "grad": x.grad is not None and bool(torch.isfinite(x.grad).all()),
            "keys": sorted(sd), "rebuilt_same": bool(torch.equal(rebuilt.data, w.data)),
            "is_linear": isinstance(layer, torch.nn.Linear),
        }))
        """
    )
    assert out["cls"] == "Params4bit" and out["quantized"] and out["module_qs"]
    assert out["match"] and out["grad"] and out["is_linear"]
    assert out["keys"] == [
        "bias",
        "weight",
        "weight.absmax",
        "weight.nested_absmax",
        "weight.nested_quant_map",
        "weight.quant_map",
        "weight.quant_state.bitsandbytes__nf4",
    ]
    assert out["rebuilt_same"]


def test_optimizers_keep_8bit_state_and_int8_refuses():
    out = run(
        """
        shim.install()
        import contextlib, io, pickle
        import torch
        import bitsandbytes as bnb
        from bitsandbytes.optim.optimizer import Optimizer2State, Optimizer8bit
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(5000))
        small = torch.nn.Parameter(torch.randn(10))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            # transformers' optim="paged_adamw_8bit": AdamW(optim_bits=8, is_paged=True)
            opt = bnb.optim.AdamW([p, small], lr = 0.1, betas = (0.9, 0.999), eps = 1e-8,
                                  weight_decay = 0.0, optim_bits = 8, is_paged = True)
            bnb.optim.AdamW8bit([p], lr = 0.1)
            bnb.optim.SGD8bit([p], lr = 0.1)
        before = p.detach().clone()
        p.grad = torch.ones(5000)
        small.grad = torch.ones(10)
        opt.step()
        st, st_small = opt.state[p], opt.state[small]
        sd = opt.state_dict()
        pickle.loads(pickle.dumps(sd))
        wrapped = sd["state"][0]["__bnb_optimizer_quant_state__"]
        # Param groups given as generators (Trainer builds lists, others may not) keep their params.
        q = torch.nn.Parameter(torch.ones(3))
        grouped = bnb.optim.AdamW8bit(({"params": (t for t in [q]), "lr": 0.1},))
        group_sizes = [len(g["params"]) for g in grouped.param_groups]
        pickle.loads(pickle.dumps(bnb.optim.PagedAdamW8bit))
        h = torch.nn.Parameter(torch.ones(4096, dtype = torch.float16))
        h.grad = torch.full_like(h, 0.5)
        bnb.optim.AdamW8bit([h], lr = 1e-2).step()
        # a bitsandbytes checkpoint resumes, and lands on the same state
        twin = torch.nn.Parameter(before.clone())
        resumed = bnb.optim.AdamW([twin, torch.nn.Parameter(small.detach().clone())], lr = 0.1,
                                  weight_decay = 0.0, optim_bits = 8)
        resumed.load_state_dict(sd)
        rs = resumed.state[twin]
        same = all(torch.equal(rs[k], st[k]) for k in ("state1", "state2", "absmax1", "absmax2"))
        # a GlobalOptimManager override pins one weight to 32-bit state
        e = torch.nn.Parameter(torch.randn(8192))
        mng = bnb.optim.GlobalOptimManager.get_instance()
        mng.override_config(e, "optim_bits", 32)
        o2 = bnb.optim.Adam8bit([e])
        e.grad = torch.ones_like(e)
        o2.step()
        errors = {}
        for label, make in (
            ("amsgrad", lambda: bnb.optim.AdamW8bit([p], amsgrad = True)),
            ("clipping", lambda: bnb.optim.AdamW([p], percentile_clipping = 50)),
            ("lion", lambda: bnb.optim.Lion([p])),
            ("int8", lambda: bnb.nn.Linear8bitLt(4, 4)),
        ):
            try:
                make()
                errors[label] = "built"
            except (RuntimeError, ValueError) as err:
                errors[label] = str(err)
        bnb.optim.AdamW([p], percentile_clipping = 100, block_wise = True)  # older defaults accepted
        # a transposed (non-contiguous) weight steps with contiguous 8-bit state
        tp = torch.nn.Parameter(torch.randn(64, 64).t())
        tp.grad = torch.randn(64, 64).t()
        o3 = bnb.optim.AdamW8bit([tp])
        o3.step()
        transposed = [str(o3.state[tp]["state1"].dtype), o3.state[tp]["state1"].is_contiguous()]
        # a checkpoint of the previous stand-in (torch AdamW moments) resumes as 32-bit state
        w = torch.nn.Parameter(torch.randn(5000))
        w.grad = torch.randn(5000)
        old = torch.optim.AdamW([w], lr = 1e-3)
        old.step()
        new = bnb.optim.AdamW8bit([w], lr = 1e-3)
        new.load_state_dict(old.state_dict())
        new.step()
        migrated = [str(new.state[w]["state1"].dtype), new.state[w]["step"], "exp_avg" in new.state[w]]
        sgd = bnb.optim.SGD8bit([torch.nn.Parameter(torch.ones(4))], lr = 0.1)
        try:
            sgd.load_state_dict({"state": {0: {"state1": torch.zeros(4, dtype = torch.uint8)}},
                                 "param_groups": sgd.state_dict()["param_groups"]})
            sgd_resume = "loaded"
        except RuntimeError as err:
            sgd_resume = str(err)
        print("OUT " + json.dumps({
            "bnb_class": isinstance(opt, Optimizer2State) and isinstance(opt, Optimizer8bit),
            "module": type(opt).__module__, "name": type(opt).__name__, "factory_module": bnb.optim.AdamW.__module__,
            # transformers >= 5 treats anything that is not a torch.optim.Optimizer subclass as a factory.
            "optimizer_subclass": all(
                isinstance(c, type) and issubclass(c, torch.optim.Optimizer)
                for c in (bnb.optim.AdamW, bnb.optim.AdamW8bit, bnb.optim.PagedAdamW8bit, bnb.optim.Adam8bit)
            ),
            "kinds": [str(st["state1"].dtype), str(st_small["state1"].dtype), str(o2.state[e]["state1"].dtype)],
            "keys": sorted(k for k in st), "wrapped": sorted(wrapped), "step": sd["state"][0]["step"],
            "absmax_blocks": st["absmax1"].numel(), "paged": opt.is_paged,
            "moved": float((p.detach() - before).abs().max()),
            "notices": printed.getvalue().count("runs torch.optim"), "sgd_notice": printed.getvalue().count("SGD8bit"),
            "fp16_moved": float((h.detach().float() - 1).abs().max()), "resumed_same": same,
            "errors": errors, "group_sizes": group_sizes, "sgd_resume": sgd_resume,
            "transposed": transposed, "migrated": migrated,
            "manager": type(bnb.optim.GlobalOptimManager.get_instance()).__name__,
        }))
        """
    )
    assert (
        out["bnb_class"] and out["module"] == "bitsandbytes.optim.adamw" and out["name"] == "AdamW"
    )
    assert out["factory_module"] == "bitsandbytes.optim.adamw" and out["optimizer_subclass"]
    # >= min_8bit_size elements: 8-bit codes; under it, or overridden to 32 bits: float32 state.
    assert out["kinds"] == ["torch.uint8", "torch.float32", "torch.float32"]
    assert out["keys"] == ["absmax1", "absmax2", "qmap1", "qmap2", "state1", "state2", "step"]
    assert out["wrapped"] == ["absmax1", "absmax2", "qmap1", "qmap2", "state1", "state2"]
    assert out["step"] == 1 and out["absmax_blocks"] == 20 and out["paged"] is True
    # Adam's first step moves every weight by lr (sign of the gradient); no fallback notice for Adam.
    assert abs(out["moved"] - 0.1) < 1e-3 and out["fp16_moved"] > 0
    assert out["notices"] == 1 and out["sgd_notice"] == 1
    assert out["resumed_same"] and out["group_sizes"] == [1]
    assert "amsgrad" in out["errors"]["amsgrad"] and "bitsandbytes" in out["errors"]["clipping"]
    assert "load_in_8bit" in out["errors"]["int8"] and "bitsandbytes" in out["errors"]["lion"]
    assert "8-bit state" in out["sgd_resume"]
    assert out["transposed"] == ["torch.uint8", True]
    assert out["migrated"] == ["torch.float32", 2, False]
    assert out["manager"] == "GlobalOptimManager"


def test_torch_adam_path_matches_float64_oracle():
    # CPU tensors take the plain torch update: float32 rounding away from the float64 oracle.
    out = run(
        """
        shim.install()
        import torch
        import bitsandbytes as bnb
        from bitsandbytes.optim._adam_torch import adam_8bit_blockwise_torch
        torch.manual_seed(0)
        code1 = bnb.functional.create_dynamic_map(signed = True)
        code2 = bnb.functional.create_dynamic_map(signed = False)
        worst = 0.0
        for n in (257, 5000):
            p = torch.randn(n)
            st = [torch.zeros(n, dtype = torch.uint8), torch.zeros(n, dtype = torch.uint8),
                  torch.zeros(-(n // -256)), torch.zeros(-(n // -256))]
            for t in range(1, 6):
                g = torch.randn(n)
                ref = [x.clone() for x in [p] + st]
                adam_8bit_blockwise_torch(p, g, *st[:2], code1, code2, *st[2:], t, 1e-2, 0.9, 0.999, 1e-8, 0.01)
                adam_8bit_blockwise_torch(ref[0], g, ref[1], ref[2], code1, code2, ref[3], ref[4], t, 1e-2,
                                          0.9, 0.999, 1e-8, 0.01, compute_dtype = torch.float64)
                # relative to the weight, or to the step (lr) for weights near zero
                rel = ((p - ref[0]).abs() / ref[0].abs().clamp_min(1e-2)).max().item()
                worst = max(worst, rel)
        print("OUT " + json.dumps({"worst_rel": worst}))
        """
    )
    assert out["worst_rel"] < 1e-5, out


def test_auto_without_gpu_does_nothing():
    out = run(
        """
        shim._real_installed = lambda: False
        print("OUT " + json.dumps({"installed": shim.install(),
                                   "finders": [type(f).__name__ for f in sys.meta_path]}))
        """,
        mode = "auto",
    )
    assert out["installed"] is False and "_ShimFinder" not in out["finders"]


def test_vllm_gate_and_int8_refusal():
    out = run(
        """
        shim.install()
        g = shim.vllm_blocked
        try:
            shim.refuse_int8(True)
            int8 = "allowed"
        except RuntimeError:
            int8 = "refused"
        print("OUT " + json.dumps({
            "4bit": g("unsloth/Qwen3-0.6B", True, False, False, False, None),
            "prequantized_name": g("unsloth/Qwen3-0.6B-unsloth-bnb-4bit", False, False, False, False, None),
            "bnb_config": g("org/model", False, False, False, False, {"load_in_4bit": True}),
            "16bit": g("unsloth/Qwen3-0.6B", True, False, True, False, None),
            "full_ft": g("unsloth/Qwen3-0.6B", True, False, False, True, None),
            "int8": int8,
        }))
        """
    )
    assert out == {
        "4bit": True,
        "prequantized_name": True,
        "bnb_config": True,
        "16bit": False,
        "full_ft": False,
        "int8": "refused",
    }


def test_native_library_reads_as_missing():
    # unsloth.bnb_availability must report the stand-in as having no native kernels.
    out = run(
        f"""
        shim.install()
        import bitsandbytes
        spec2 = importlib.util.spec_from_file_location("_bnb_av", {os.path.join(REPO, "unsloth", "bnb_availability.py")!r})
        av = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(av)
        handle = bitsandbytes.functional.lib.cdequantize_blockwise_fp32
        try:
            handle()
            called = "ran"
        except RuntimeError:
            called = "raised"
        print("OUT " + json.dumps({{"ready": av.native_kernels_ready(bitsandbytes, "cuda"), "called": called}}))
        """
    )
    assert out == {"ready": False, "called": "raised"}


def test_unusable_probe_pins_windows_rocm_version_first():
    # A Windows ROCm wheel only loads once BNB_ROCM_VERSION is pinned: the probe must pin it before
    # importing bitsandbytes, or a working install reads as broken and gets replaced.
    out = run(
        f"""
        import types, torch
        calls = []
        fake_fixes = types.ModuleType("unsloth.import_fixes")
        fake_fixes.maybe_set_windows_rocm_bnb_version = lambda: calls.append("pin")
        sys.modules.setdefault("unsloth", types.ModuleType("unsloth"))
        sys.modules["unsloth.import_fixes"] = fake_fixes
        class Probe:
            def find_spec(self, name, path = None, target = None):
                if name == "bitsandbytes":
                    calls.append("import")
                    raise ImportError("probe stop")
                return None
        sys.meta_path.insert(0, Probe())
        torch.cuda.is_available = lambda: True
        unusable = shim._real_unusable()
        print("OUT " + json.dumps({{"calls": calls, "unusable": unusable}}))
        """,
        mode = "auto",
    )
    assert out["calls"][:2] == ["pin", "import"], out


def test_matmul_4bit_bias_grad_matches_dense():
    # Unbatched (1-D) and batched inputs: the bias grad has the bias shape and equals the dense one.
    out = run(
        """
        shim.install()
        import torch
        import bitsandbytes as bnb
        from bitsandbytes import functional as F
        torch.manual_seed(0)
        w = torch.randn(32, 64)
        q, st = F.quantize_4bit(w, blocksize = 64, quant_type = "nf4")
        W = F.dequantize_4bit(q, st)
        res = {}
        for name, shape in (("unbatched", (64,)), ("batched", (2, 3, 64))):
            x = torch.randn(shape, requires_grad = True)
            b = torch.randn(32, requires_grad = True)
            bnb.matmul_4bit(x, q, st, bias = b).pow(2).sum().backward()
            x2 = x.detach().clone().requires_grad_(True)
            b2 = b.detach().clone().requires_grad_(True)
            torch.nn.functional.linear(x2, W, b2).pow(2).sum().backward()
            res[name] = [list(b.grad.shape), bool(torch.allclose(b.grad, b2.grad)),
                         bool(torch.allclose(x.grad, x2.grad))]
        print("OUT " + json.dumps(res))
        """
    )
    assert out["unbatched"] == [[32], True, True], out
    assert out["batched"] == [[32], True, True], out
