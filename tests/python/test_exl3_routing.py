"""GPU-free unit tests for EXL3 backend-selection routing (unsloth.exllama.loader)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from unsloth.exllama import loader as EL
from unsloth.exllama.config import Exl3Config
import json
import tempfile


class TestExl3DefaultBackend(unittest.TestCase):
    """exl3_is_default_backend decision matrix (EXL3 replaces bnb as default)."""

    def _call(self, **kw):
        # Force exllamav3 + CUDA "available" so we test the decision logic itself.
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
        ):
            with patch.dict(os.environ, {}, clear = False):
                os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
                return EL.exl3_is_default_backend(**kw)

    def test_plain_4bit_defaults_to_exl3(self):
        self.assertTrue(self._call(load_in_4bit = True))

    def test_pure_8bit_stays_on_bnb(self):
        self.assertFalse(self._call(load_in_8bit = True))

    def test_no_quant_is_not_exl3(self):
        self.assertFalse(self._call())

    def test_explicit_bnb_config_opts_out(self):
        bnb_like = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        self.assertFalse(self._call(load_in_4bit = True, quantization_config = bnb_like))

    def test_bnb_flags_without_method_opt_out(self):
        self.assertFalse(self._call(load_in_4bit = True, quantization_config = {"load_in_4bit": True}))

    def test_exl3_config_does_not_count_as_bnb(self):
        self.assertTrue(self._call(load_in_4bit = True, quantization_config = Exl3Config(bits = 3)))

    def test_env_override_forces_bnb(self):
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch.dict(os.environ, {"UNSLOTH_QUANT_BACKEND": "bnb"}),
        ):
            self.assertFalse(EL.exl3_is_default_backend(load_in_4bit = True))

    def test_real_bitsandbytes_config_opts_out(self):
        # A genuine transformers BitsAndBytesConfig reports quant_method as an
        # enum whose str() is 'QuantizationMethod.BITS_AND_BYTES' but whose
        # .value is 'bitsandbytes'. It must route to bnb, not EXL3.
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            self.skipTest("transformers not available")
        bnb = BitsAndBytesConfig(load_in_4bit = True)
        self.assertFalse(self._call(load_in_4bit = True, quantization_config = bnb))

    def test_no_exllama_falls_back(self):
        with (
            patch.object(EL, "is_exllama_available", lambda: False),
            patch.object(EL, "_cuda_available", lambda: True),
        ):
            self.assertFalse(EL.exl3_is_default_backend(load_in_4bit = True))

    def test_no_cuda_falls_back(self):
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: False),
        ):
            self.assertFalse(EL.exl3_is_default_backend(load_in_4bit = True))

    def test_calibrate_override_flows_to_config(self):
        # The loader's exl3_calibrate= override must reach the resolved config,
        # unless the user passed their own Exl3Config (which wins).
        from unsloth.exllama.config import Exl3Config

        self.assertFalse(EL.resolve_exl3_config(True, None, calibrate = False).calibrate)
        self.assertTrue(EL.resolve_exl3_config(True, None, calibrate = True).calibrate)
        # explicit user config is authoritative
        user = Exl3Config(bits = 3, calibrate = True)
        self.assertTrue(EL.resolve_exl3_config(True, user, calibrate = False).calibrate)


class TestArchSupportGating(unittest.TestCase):
    """Default routing must fall back to bnb for architectures EXL3 lacks."""

    def _tmp_model(self, arch):
        import json, tempfile, os

        d = tempfile.mkdtemp(prefix = "exl3_arch_")
        with open(os.path.join(d, "config.json"), "w", encoding = "utf-8") as f:
            json.dump({"architectures": [arch]}, f)
        return d

    def test_unsupported_arch_local_dir_falls_back(self):
        d = self._tmp_model("OlmoeForCausalLM")
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch("unsloth.exllama.patcher.exllama_supports_arch", lambda p: False),
            patch.dict(os.environ, {}, clear = False),
        ):
            os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
            # A quantized default load of an unsupported arch -> not EXL3.
            self.assertFalse(EL.should_use_exl3(d, load_in_4bit = True))

    def test_supported_arch_local_dir_routes_to_exl3(self):
        d = self._tmp_model("LlamaForCausalLM")
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch("unsloth.exllama.patcher.exllama_supports_arch", lambda p: True),
            patch.dict(os.environ, {}, clear = False),
        ):
            os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
            self.assertTrue(EL.should_use_exl3(d, load_in_4bit = True))

    def test_explicit_request_still_routes_even_if_unsupported(self):
        # Explicit load_in_exl3=True must still route to EXL3 (so the loader can
        # raise its clear "unsupported architecture" error), not silently skip.
        d = self._tmp_model("OlmoeForCausalLM")
        self.assertTrue(EL.should_use_exl3(d, load_in_exl3 = True))


class TestArchDetectionLogic(unittest.TestCase):
    """Exercise the *real* exllama_supports_arch against a mocked registry.

    (The tests above stub exllama_supports_arch itself, so they do not cover the
    function's own logic - in particular the fail-open behaviour when the
    registry cannot be read.)
    """

    def _tmp_model(self, cfg):
        import json, tempfile, os

        d = tempfile.mkdtemp(prefix = "exl3_detect_")
        with open(os.path.join(d, "config.json"), "w", encoding = "utf-8") as f:
            json.dump(cfg, f)
        return d

    def setUp(self):
        from unsloth.exllama import patcher as PT
        self.PT = PT
        self.SUPPORTED = {"LlamaForCausalLM", "Qwen2ForCausalLM"}

    def test_supported_and_unsupported_local_dir(self):
        with patch.object(self.PT, "exllama_supported_architectures", lambda: self.SUPPORTED):
            self.assertTrue(
                self.PT.exllama_supports_arch(
                    self._tmp_model({"architectures": ["LlamaForCausalLM"]})
                )
            )
            self.assertFalse(
                self.PT.exllama_supports_arch(
                    self._tmp_model({"architectures": ["OlmoeForCausalLM"]})
                )
            )

    def test_nested_text_config_architectures(self):
        with patch.object(self.PT, "exllama_supported_architectures", lambda: self.SUPPORTED):
            d = self._tmp_model(
                {
                    "architectures": ["SomeVLForConditionalGeneration"],
                    "text_config": {"architectures": ["Qwen2ForCausalLM"]},
                }
            )
            self.assertTrue(self.PT.exllama_supports_arch(d))

    def test_fail_open_when_registry_unreadable(self):
        # If the registry cannot be read (empty set) we must NOT claim the model
        # is unsupported, or every supported model would silently downgrade to
        # bnb (default route) or be spuriously rejected (explicit route).
        with patch.object(self.PT, "exllama_supported_architectures", lambda: set()):
            d = self._tmp_model({"architectures": ["OlmoeForCausalLM"]})
            self.assertTrue(self.PT.exllama_supports_arch(d))
            self.assertTrue(self.PT.exllama_supports_arch("AnythingForCausalLM"))

    def test_defers_when_config_unreadable(self):
        import tempfile
        with patch.object(self.PT, "exllama_supported_architectures", lambda: self.SUPPORTED):
            # Directory with no config.json - cannot prove unsupported.
            empty = tempfile.mkdtemp(prefix = "exl3_noconfig_")
            self.assertTrue(self.PT.exllama_supports_arch(empty))


class TestFullFinetuningPrecedence(unittest.TestCase):
    """Full finetuning must never be routed through EXL3.

    The loader guards the EXL3 branch with ``not full_finetuning``; this test
    documents/locks that contract at the decision-function level: EXL3 default
    routing is about *quantized* loads, and a full-finetuning request is not one
    (its base weights are trained in full precision).
    """

    def test_default_backend_is_quant_only(self):
        # exl3_is_default_backend only says yes for a quantized (4-bit) load;
        # the loader additionally gates on `not full_finetuning`.
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch.dict(os.environ, {}, clear = False),
        ):
            os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
            # A 16-bit (non-quantized) load - as full finetuning effectively is
            # once the loader disables 4bit - is never EXL3.
            self.assertFalse(EL.exl3_is_default_backend(load_in_4bit = False))


class TestShouldUseExl3(unittest.TestCase):
    def test_explicit_flag(self):
        self.assertTrue(EL.should_use_exl3("m", load_in_exl3 = True))
        self.assertTrue(EL.should_use_exl3("m", load_in_exl3 = "3bit"))

    def test_exl3_config(self):
        self.assertTrue(EL.should_use_exl3("m", quantization_config = Exl3Config(bits = 3)))
        self.assertTrue(
            EL.should_use_exl3("m", quantization_config = {"quant_method": "exl3", "bits": 3})
        )

    def test_plain_non_quant_is_false(self):
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
        ):
            self.assertFalse(EL.should_use_exl3("m"))

    def test_default_route_when_available(self):
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch.dict(os.environ, {}, clear = False),
        ):
            os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
            self.assertTrue(EL.should_use_exl3("m", load_in_4bit = True))

    def test_on_disk_checkpoint_detected(self):
        with patch.object(EL, "is_exl3_model_dir", lambda p: True):
            self.assertTrue(EL.should_use_exl3("/some/exl3/dir"))


class TestNonExl3QuantOptOut(unittest.TestCase):
    """Existing non-EXL3 quant configs / checkpoints / adapter repos opt out."""

    def _mkdir(self, files):
        d = tempfile.mkdtemp(prefix = "exl3_optout_")
        for name, content in files.items():
            with open(os.path.join(d, name), "w", encoding = "utf-8") as f:
                json.dump(content, f)
        return d

    def _route(self, model_name, **kw):
        with (
            patch.object(EL, "is_exllama_available", lambda: True),
            patch.object(EL, "_cuda_available", lambda: True),
            patch.dict(os.environ, {}, clear = False),
        ):
            os.environ.pop("UNSLOTH_QUANT_BACKEND", None)
            return EL.should_use_exl3(model_name, load_in_4bit = True, **kw)

    def test_gptq_awq_hqq_configs_opt_out(self):
        for method in ("gptq", "awq", "hqq", "bitsandbytes"):
            self.assertFalse(self._route("m", quantization_config = {"quant_method": method}))

    def test_existing_bnb_checkpoint_not_hijacked(self):
        d = self._mkdir({"config.json": {"quantization_config": {"quant_method": "bitsandbytes"}}})
        self.assertFalse(self._route(d))

    def test_existing_gptq_checkpoint_not_hijacked(self):
        d = self._mkdir({"config.json": {"quantization_config": {"quant_method": "gptq"}}})
        self.assertFalse(self._route(d))

    def test_peft_adapter_repo_not_quantized(self):
        d = self._mkdir({"adapter_config.json": {"base_model_name_or_path": "x"}})
        self.assertFalse(self._route(d))

    def test_plain_unquantized_dir_still_routes(self):
        d = self._mkdir({"config.json": {"architectures": ["LlamaForCausalLM"]}})
        with patch("unsloth.exllama.patcher.exllama_supports_arch", lambda p: True):
            self.assertTrue(self._route(d))


class TestResolvedCheckpointSafety(unittest.TestCase):
    def _mkdir(self, config):
        d = tempfile.mkdtemp(prefix = "exl3_resolved_")
        with open(os.path.join(d, "config.json"), "w", encoding = "utf-8") as f:
            json.dump(config, f)
        return d

    def test_remote_snapshot_of_native_quant_checkpoint_skips_converter(self):
        d = self._mkdir(
            {
                "architectures": ["LlamaForCausalLM"],
                "quantization_config": {"quant_method": "bitsandbytes"},
            }
        )
        with (
            patch.object(EL, "require_exllama"),
            patch.object(EL, "_resolve_local_dir", return_value = d),
            patch.object(EL, "patch_transformers_exl3", side_effect = AssertionError("EXL3 patch must be skipped")),
            patch.object(EL, "quantize_to_exl3", side_effect = AssertionError("native weights must not be re-quantized")),
        ):
            plan = EL.prepare_exl3_checkpoint(
                "unsloth/example-bnb-4bit",
                load_in_exl3 = True,
                is_explicit = False,
            )
        self.assertIsNotNone(plan)
        self.assertFalse(plan.is_exl3)
        self.assertEqual(plan.checkpoint_dir, d)
        self.assertEqual(plan.source_model, "unsloth/example-bnb-4bit")

    def test_explicit_exl3_request_rejects_native_quant_checkpoint(self):
        d = self._mkdir(
            {"quantization_config": {"quant_method": "gptq"}}
        )
        with (
            patch.object(EL, "require_exllama"),
            patch.object(EL, "_resolve_local_dir", return_value = d),
            self.assertRaisesRegex(ValueError, "cannot quantize an existing non-EXL3 quantized checkpoint"),
        ):
            EL.prepare_exl3_checkpoint(
                "org/example-gptq",
                load_in_exl3 = True,
                is_explicit = True,
            )

    def test_source_reference_is_used_by_peft(self):
        model = SimpleNamespace(
            name_or_path = "/home/user/.cache/unsloth/exl3/model",
            config = SimpleNamespace(_name_or_path = "/home/user/.cache/unsloth/exl3/model"),
        )
        result = EL.set_exl3_source_model_reference(model, "org/model")
        self.assertIs(result, model)
        self.assertEqual(model.name_or_path, "org/model")
        self.assertEqual(model.config._name_or_path, "org/model")
        self.assertEqual(model._unsloth_exl3_source_model, "org/model")


if __name__ == "__main__":
    unittest.main()
