"""GPU-free unit tests for EXL3 checkpoint detection and cache-dir resolution."""

import json
import os
import tempfile
import unittest

from unsloth.exllama.patcher import is_exl3_model_dir, read_exl3_bitrate
from unsloth.exllama.quantize import resolve_exl3_cache_dir, is_moe_model, _read_model_config
from unsloth.exllama.config import Exl3Config


class TestCheckpointDetection(unittest.TestCase):
    def _mkdir_with(self, files: dict) -> str:
        d = tempfile.mkdtemp(prefix = "exl3_detect_")
        for name, content in files.items():
            with open(os.path.join(d, name), "w", encoding = "utf-8") as f:
                if isinstance(content, (dict, list)):
                    json.dump(content, f)
                else:
                    f.write(content)
        return d

    def test_not_a_dir(self):
        self.assertFalse(is_exl3_model_dir("/definitely/not/here"))
        self.assertFalse(is_exl3_model_dir(None))

    def test_detects_via_config_json(self):
        d = self._mkdir_with(
            {"config.json": {"quantization_config": {"quant_method": "exl3", "bits": 3}}}
        )
        self.assertTrue(is_exl3_model_dir(d))

    def test_detects_via_quantization_config_json(self):
        d = self._mkdir_with({"quantization_config.json": {"quant_method": "exl3", "bits": 4}})
        self.assertTrue(is_exl3_model_dir(d))

    def test_detects_via_trellis_in_index(self):
        d = self._mkdir_with(
            {
                "model.safetensors.index.json": {
                    "weight_map": {"model.layers.0.self_attn.q_proj.trellis": "model.safetensors"}
                }
            }
        )
        self.assertTrue(is_exl3_model_dir(d))

    def test_not_exl3_for_bnb_config(self):
        d = self._mkdir_with(
            {
                "config.json": {
                    "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}
                }
            }
        )
        self.assertFalse(is_exl3_model_dir(d))

    def test_read_bitrate(self):
        d = self._mkdir_with({"quantization_config.json": {"quant_method": "exl3", "bits": 2.5}})
        self.assertEqual(read_exl3_bitrate(d), 2.5)

    def test_read_bitrate_none_when_absent(self):
        d = self._mkdir_with({"config.json": {"hidden_size": 128}})
        self.assertIsNone(read_exl3_bitrate(d))


class TestCacheDir(unittest.TestCase):
    def test_deterministic_and_labelled(self):
        cfg = Exl3Config(bits = 3, head_bits = 6)
        p1 = resolve_exl3_cache_dir("org/Model-Name", cfg, cache_root = "/tmp/xcache")
        p2 = resolve_exl3_cache_dir("org/Model-Name", cfg, cache_root = "/tmp/xcache")
        self.assertEqual(p1, p2)
        self.assertTrue(p1.endswith(os.path.join("3bpw_H6")))
        self.assertIn("/tmp/xcache", p1)

    def test_different_bits_differ(self):
        a = resolve_exl3_cache_dir("m", Exl3Config(bits = 3), cache_root = "/tmp/x")
        b = resolve_exl3_cache_dir("m", Exl3Config(bits = 4), cache_root = "/tmp/x")
        self.assertNotEqual(a, b)

    def test_different_models_differ(self):
        a = resolve_exl3_cache_dir("org/A", Exl3Config(bits = 3), cache_root = "/tmp/x")
        b = resolve_exl3_cache_dir("org/B", Exl3Config(bits = 3), cache_root = "/tmp/x")
        self.assertNotEqual(a, b)


class TestMoEDetection(unittest.TestCase):
    def _write_cfg(self, cfg: dict) -> str:
        d = tempfile.mkdtemp(prefix = "exl3_moe_")
        with open(os.path.join(d, "config.json"), "w", encoding = "utf-8") as f:
            json.dump(cfg, f)
        return d

    def test_mixtral_detected(self):
        d = self._write_cfg({"hidden_size": 512, "num_local_experts": 8})
        self.assertTrue(is_moe_model(d))

    def test_qwen3_moe_detected(self):
        d = self._write_cfg({"hidden_size": 512, "num_experts": 128})
        self.assertTrue(is_moe_model(d))

    def test_dense_not_detected(self):
        d = self._write_cfg({"hidden_size": 512, "intermediate_size": 2048})
        self.assertFalse(is_moe_model(d))

    def test_nested_text_config_merged(self):
        d = self._write_cfg({"text_config": {"num_local_experts": 4, "hidden_size": 256}})
        self.assertTrue(is_moe_model(d))
        merged = _read_model_config(d)
        self.assertEqual(merged.get("num_local_experts"), 4)


if __name__ == "__main__":
    unittest.main()
