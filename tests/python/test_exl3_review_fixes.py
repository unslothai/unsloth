"""GPU-free regression tests for the EXL3 PR review fixes.

Covers the converter-argument namespace, the conversion cache key, the 1-bpw
request, the unsupported ``skip_modules`` option, offline calibration, the
shared dense-expert cache budget and device_map-aware expert placement.
"""

import argparse
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

from unsloth.exllama import loader as EL
from unsloth.exllama import moe as M
from unsloth.exllama import quantize as Q
from unsloth.exllama.config import Exl3Config


def _fake_converter(parser):
    pkg = types.ModuleType("exllamav3")
    conv = types.ModuleType("exllamav3.conversion")
    cm = types.ModuleType("exllamav3.conversion.convert_model")
    cm.parser = parser
    conv.convert_model = cm
    pkg.conversion = conv
    return {
        "exllamav3": pkg,
        "exllamav3.conversion": conv,
        "exllamav3.conversion.convert_model": cm,
    }


def _build_args(cfg = None):
    return Q._build_conversion_args(
        in_dir = "in",
        out_dir = "out",
        work_dir = "work",
        cfg = cfg or Exl3Config(),
        devices = "0",
        resume = False,
        verbose = False,
    )


class TestConversionArgs(unittest.TestCase):
    def test_newer_converter_options_come_from_its_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--recipe", default = None)
        parser.add_argument("--vision_bits", type = int, default = None)
        parser.add_argument("--codebook", default = "mcg")
        parser.add_argument("--bits", type = float)
        with patch.dict(sys.modules, _fake_converter(parser)):
            args = _build_args(Exl3Config(bits = 3, codebook = "mul1"))
        self.assertIn("recipe", args)
        self.assertIsNone(args.vision_bits)
        # Our values win over the parser defaults.
        self.assertEqual(args.bits, 3.0)
        self.assertEqual(args.codebook, "mul1")

    def test_parser_that_exits_falls_back(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required = True)
        with patch.dict(sys.modules, _fake_converter(parser)):
            with patch.object(sys, "stderr", open(os.devnull, "w")):
                args = _build_args()
        self.assertEqual(args.bits, 4.0)


class TestCacheKey(unittest.TestCase):
    def test_default_config_keeps_its_existing_directory(self):
        path = Q.resolve_exl3_cache_dir("m", Exl3Config(), "/c")
        self.assertTrue(path.endswith(os.sep + Exl3Config().label()))

    def test_every_conversion_option_changes_the_directory(self):
        base = Q.resolve_exl3_cache_dir("m", Exl3Config(), "/c")
        variants = [
            Exl3Config(codebook = "mul1"),
            Exl3Config(mtp_bits = 6),
            Exl3Config(calibrate = False),
            Exl3Config(cal_rows = 100),
            Exl3Config(cal_cols = 512),
        ]
        paths = {Q.resolve_exl3_cache_dir("m", v, "/c") for v in variants}
        self.assertEqual(len(paths), len(variants))
        self.assertNotIn(base, paths)

    def test_cache_key_is_stable(self):
        a = Q.resolve_exl3_cache_dir("m", Exl3Config(cal_rows = 100), "/c")
        b = Q.resolve_exl3_cache_dir("m", Exl3Config(cal_rows = 100), "/c")
        self.assertEqual(a, b)


class TestBitrateRequest(unittest.TestCase):
    def test_one_bpw_is_not_mistaken_for_true(self):
        self.assertEqual(EL.resolve_exl3_config(1, None).bits, 1.0)
        self.assertEqual(EL.resolve_exl3_config(1.0, None).bits, 1.0)

    def test_booleans_still_mean_default(self):
        self.assertEqual(EL.resolve_exl3_config(True, None).bits, Exl3Config().bits)


class TestSkipModules(unittest.TestCase):
    def test_non_empty_skip_modules_is_rejected(self):
        with self.assertRaises(ValueError):
            Exl3Config(skip_modules = ["lm_head"])

    def test_empty_skip_modules_round_trips(self):
        cfg = Exl3Config.from_dict(Exl3Config().to_dict())
        self.assertEqual(cfg.skip_modules, [])


class TestOfflineCalibration(unittest.TestCase):
    def test_offline_env_disables_network(self):
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UNSLOTH_EXL3_NO_NETWORK"):
            with patch.dict(os.environ, {var: "1"}):
                self.assertTrue(Q._network_disabled())

    def test_data_free_conversion_does_not_fetch_calibration_data(self):
        import tempfile
        for calibrate, expected in ((False, 0), (True, 1)):
            calls = []
            fake = _fake_converter(argparse.ArgumentParser())
            cm = fake["exllamav3.conversion.convert_model"]
            cm.prepare = lambda args: ({}, {}, True, None)
            cm.main = lambda in_args, job_state: None
            with (
                tempfile.TemporaryDirectory() as root,
                patch.dict(sys.modules, fake),
                patch.object(Q, "ensure_calibration_data", lambda: calls.append(1) or True),
                patch.object(Q, "require_exllama", lambda: None),
                patch("unsloth.exllama.patcher.patch_transformers_exl3", lambda: True),
                patch.object(Q, "is_exl3_model_dir", lambda path: False),
                patch.object(Q, "_preflight_check", lambda *a, **k: None),
                patch.object(Q, "ensure_vlm_preprocessor", lambda *a, **k: None),
                patch.object(Q, "is_moe_model", lambda *a, **k: False),
                patch.object(Q, "_conversion_complete", side_effect = [False, True]),
                patch.object(Q, "_maybe_enable_uncalibrated", lambda cfg: None),
            ):
                src = os.path.join(root, "src")
                os.makedirs(src)
                Q.quantize_to_exl3(src, Exl3Config(calibrate = calibrate), cache_root = root)
            self.assertEqual(len(calls), expected, f"calibrate={calibrate}")


class _MockLinear:
    def __init__(self, in_features, out_features):
        self._w = torch.randn(in_features, out_features, dtype = torch.float16)

    def get_weight_tensor(self):
        return self._w


def _experts(
    num_experts = 4,
    hidden = 16,
    inter = 8,
):
    exp = M.Exl3QuantizedExperts(num_experts, hidden, inter, torch.nn.SiLU())
    for e in range(num_experts):
        gate_up = M._FusedGateUp(_MockLinear(hidden, inter), _MockLinear(hidden, inter))
        exp.set_expert(e, gate_up, _MockLinear(inter, hidden))
    return exp


class TestSharedExpertBudget(unittest.TestCase):
    def setUp(self):
        M._DENSE_EXPERT_LRU.clear()
        M._DENSE_EXPERT_BYTES = 0

    def test_budget_bounds_all_layers_together(self):
        hidden, inter = 16, 8
        # One dense expert in fp32: gate_up [2*inter, hidden] + down [hidden, inter].
        per_expert = (2 * inter * hidden + hidden * inter) * 4
        env = {
            "UNSLOTH_EXL3_EXPERT_CACHE": "64",
            "UNSLOTH_EXL3_EXPERT_CACHE_MB": str(3 * per_expert / (1 << 20)),
        }
        with patch.dict(os.environ, env):
            layers = [_experts(hidden = hidden, inter = inter) for _ in range(4)]
            for layer in layers:
                for e in range(layer.num_experts):
                    layer._get_dense(e, torch.float32, torch.device("cpu"))
        self.assertLessEqual(M._dense_expert_cache_bytes(), 3 * per_expert)
        held = sum(len(layer._gu_cache) for layer in layers)
        self.assertEqual(held, 3)

    def test_cache_is_keyed_by_dtype(self):
        with patch.dict(os.environ, {"UNSLOTH_EXL3_EXPERT_CACHE": "8"}):
            layer = _experts()
            a, _ = layer._get_dense(0, torch.float32, torch.device("cpu"))
            b, _ = layer._get_dense(0, torch.float16, torch.device("cpu"))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float16)

    def test_collected_layer_releases_its_bytes(self):
        import gc

        with patch.dict(os.environ, {"UNSLOTH_EXL3_EXPERT_CACHE": "8"}):
            layer = _experts()
            layer._get_dense(0, torch.float32, torch.device("cpu"))
        self.assertGreater(M._dense_expert_cache_bytes(), 0)
        del layer
        gc.collect()
        self.assertEqual(M._dense_expert_cache_bytes(), 0)


class TestExpertDevicePlacement(unittest.TestCase):
    def test_longest_prefix_wins(self):
        default = torch.device("cuda:0")
        device_map = {"model.layers.0": 0, "model.layers.1": 1, "model.layers.1.mlp": "cuda:2"}
        self.assertEqual(
            M._device_for_module("model.layers.1.mlp.experts", device_map, default),
            torch.device("cuda:2"),
        )
        self.assertEqual(
            M._device_for_module("model.layers.1.self_attn", device_map, default),
            torch.device("cuda", 1),
        )

    def test_cpu_and_unmapped_modules_keep_default(self):
        default = torch.device("cuda:0")
        device_map = {"model.layers.0": "cpu"}
        self.assertEqual(
            M._device_for_module("model.layers.0.mlp.experts", device_map, default), default
        )
        self.assertEqual(M._device_for_module("lm_head", device_map, default), default)
        self.assertEqual(M._device_for_module("x", "auto", default), default)


if __name__ == "__main__":
    unittest.main()
