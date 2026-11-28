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
            "first_finder": type(sys.meta_path[0]).__name__,
            "devices": sorted(bitsandbytes.supported_torch_devices),
        }))
        """
    )
    assert out["shim"] is True
    assert out["file"].startswith(os.path.join(REPO, "unsloth", "bnb_shim", "bitsandbytes"))
    assert out["version"] == out["dunder"] == "0.49.2"
    assert out["spec"] and out["mapped"]
    assert out["first_finder"] == "_ShimFinder"
    assert "cuda" in out["devices"] and "cpu" in out["devices"]


def test_off_and_working_install_are_left_alone():
    out = run(
        """
        off = shim.install()
        os.environ["UNSLOTH_BNB_SHIM"] = "auto"
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


def test_optimizers_fall_back_and_int8_refuses():
    out = run(
        """
        shim.install()
        import contextlib, io
        import torch
        import bitsandbytes as bnb
        p = torch.nn.Parameter(torch.ones(4))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            opt = bnb.optim.AdamW([p], lr = 0.1, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0,
                                  optim_bits = 8, is_paged = True)
            bnb.optim.AdamW([p], lr = 0.1, optim_bits = 8)
        p.grad = torch.ones(4)
        opt.step()
        try:
            bnb.nn.Linear8bitLt(4, 4)
            int8 = "built"
        except RuntimeError as e:
            int8 = str(e)
        try:
            bnb.optim.Lion([p])
            lion = "built"
        except RuntimeError as e:
            lion = str(e)
        print("OUT " + json.dumps({
            "torch_adamw": isinstance(opt, torch.optim.AdamW), "module": type(opt).__module__,
            "stepped": float(p[0]), "notices": printed.getvalue().count("runs torch.optim.AdamW"),
            "int8": int8, "lion": lion,
            "manager": type(bnb.optim.GlobalOptimManager.get_instance()).__name__,
        }))
        """
    )
    assert out["torch_adamw"] and out["module"].startswith("bitsandbytes.optim")
    assert out["stepped"] < 1.0 and out["notices"] == 1
    assert "load_in_8bit" in out["int8"] and "bitsandbytes" in out["lion"]
    assert out["manager"] == "GlobalOptimManager"


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
