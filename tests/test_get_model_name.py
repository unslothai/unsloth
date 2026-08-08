import json
import os
import tempfile
import unittest
from unittest.mock import patch
from unsloth.models.loader_utils import get_model_name
from unsloth.models import loader_utils
from unsloth.models.mapper import FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit


def _no_remote_mapper():
    # int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row
    return {}, {}, {}, {}, {}


def _write_cached_model(
    repo_cache,
    commit,
    *,
    revision = "main",
    sharded = False,
    complete = True,
    tokenizer = True,
    tokenizer_format = "tokenizer.json",
    vision = False,
    processor = True,
):
    snapshot = os.path.join(repo_cache, "snapshots", commit)
    os.makedirs(snapshot)
    ref = os.path.join(repo_cache, "refs", revision)
    os.makedirs(os.path.dirname(ref), exist_ok = True)
    with open(ref, "w", encoding = "utf-8") as ref_file:
        ref_file.write(commit)
    config = {"vision_config": {}} if vision else {}
    with open(os.path.join(snapshot, "config.json"), "w", encoding = "utf-8") as config_file:
        json.dump(config, config_file)
    if vision and processor:
        open(os.path.join(snapshot, "preprocessor_config.json"), "w").close()
    if tokenizer:
        open(os.path.join(snapshot, tokenizer_format), "w").close()
    if not sharded:
        open(os.path.join(snapshot, "model.safetensors"), "w").close()
        return snapshot
    shard_name = "model-00001-of-00001.safetensors"
    with open(os.path.join(snapshot, "model.safetensors.index.json"), "w") as index_file:
        json.dump({"weight_map": {"model.embed_tokens.weight": shard_name}}, index_file)
    if complete:
        open(os.path.join(snapshot, shard_name), "w").close()
    return snapshot


class TestGetModelName(unittest.TestCase):
    def _assert_mapping(self, model_name, load_in_4bit, expected, should_change):
        mapped = get_model_name(model_name, load_in_4bit = load_in_4bit)
        self.assertEqual(mapped.lower(), expected.lower())
        if should_change:
            self.assertNotEqual(mapped.lower(), model_name.lower())
        else:
            self.assertEqual(mapped.lower(), model_name.lower())

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_resolution_matrix(self):
        cases = [
            # Core mappings
            ("meta-llama/Llama-2-7b-hf", True, "unsloth/llama-2-7b-bnb-4bit", True),
            ("meta-llama/Llama-2-7b-hf", False, "unsloth/llama-2-7b", True),
            (
                "mistralai/Ministral-8B-Instruct-2410",
                True,
                "mistralai/Ministral-8B-Instruct-2410",
                False,
            ),
            (
                "meta-llama/Llama-3.2-1B-Instruct",
                False,
                "unsloth/Llama-3.2-1B-Instruct",
                True,
            ),
            (
                "meta-llama/Llama-2-7b-chat-hf",
                True,
                "unsloth/llama-2-7b-chat-bnb-4bit",
                True,
            ),
            (
                "meta-llama/Llama-3.3-70B-Instruct",
                True,
                "unsloth/llama-3.3-70b-instruct-unsloth-bnb-4bit",
                True,
            ),
            ("Qwen/Qwen3-8B", True, "unsloth/Qwen3-8B-unsloth-bnb-4bit", True),
            ("Qwen/Qwen3-8B", False, "unsloth/Qwen3-8B", True),
            ("Qwen/Qwen3-8B-FP8", False, "unsloth/Qwen3-8B-FP8", True),
            ("Qwen/Qwen3-8B-FP8", True, "unsloth/Qwen3-8B-unsloth-bnb-4bit", True),
            (
                "mistralai/Ministral-3-3B-Instruct-2512",
                True,
                "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit",
                True,
            ),
            (
                "mistralai/Ministral-3-3B-Instruct-2512",
                False,
                "unsloth/Ministral-3-3B-Instruct-2512",
                True,
            ),
            (
                "allenai/Olmo-3-7B-Instruct",
                True,
                "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit",
                True,
            ),
            (
                "allenai/Olmo-3-7B-Instruct",
                False,
                "unsloth/Olmo-3-7B-Instruct",
                True,
            ),
            (
                "allenai/Olmo-3-7B-Think",
                True,
                "unsloth/Olmo-3-7B-Think-unsloth-bnb-4bit",
                True,
            ),
            (
                "allenai/Olmo-3-7B-Think",
                False,
                "unsloth/Olmo-3-7B-Think",
                True,
            ),
            (
                "allenai/Olmo-3-32B-Think",
                True,
                "unsloth/Olmo-3-32B-Think-unsloth-bnb-4bit",
                True,
            ),
            (
                "allenai/Olmo-3-32B-Think",
                False,
                "unsloth/Olmo-3-32B-Think",
                True,
            ),
            ("unsloth/Kimi-K2-Instruct", True, "unsloth/Kimi-K2-Instruct-BF16", True),
            ("unsloth/Kimi-K2-Instruct", False, "unsloth/Kimi-K2-Instruct", False),
            # DeepScaleR-1.5B must resolve to its own 16bit repo, not another model
            (
                "agentica-org/DeepScaleR-1.5B-Preview",
                False,
                "unsloth/DeepScaleR-1.5B-Preview",
                True,
            ),
            (
                "agentica-org/DeepScaleR-1.5B-Preview",
                True,
                "unsloth/DeepScaleR-1.5B-Preview-unsloth-bnb-4bit",
                True,
            ),
            # Fallback-to-original behavior
            "nonexistent-user/nonexistent-model-123",
            "google/gemma-3-random-prototype-123",
            "imdatta0/nanoqwen-fp8",
            "imdatta0/nanoqwen-bf16",
            # Backward compatibility for legacy 4bit names
            ("unsloth/llama-2-7b-bnb-4bit", True, "unsloth/llama-2-7b-bnb-4bit", False),
            ("unsloth/llama-2-7b-bnb-4bit", False, "unsloth/llama-2-7b", True),
            ("google/gemma-2-9b", True, "unsloth/gemma-2-9b-bnb-4bit", True),
            # GPT-OSS behavior
            ("openai/gpt-oss-20b", False, "unsloth/gpt-oss-20b", True),
            ("openai/gpt-oss-20b", True, "unsloth/gpt-oss-20b-unsloth-bnb-4bit", True),
            ("unsloth/gpt-oss-20b", True, "unsloth/gpt-oss-20b-unsloth-bnb-4bit", True),
            ("unsloth/gpt-oss-20b-bf16", True, "unsloth/gpt-oss-20b-bf16", False),
            (
                "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                False,
                "unsloth/gpt-oss-20b",
                True,
            ),
            (
                "unsloth/gpt-oss-20b-bnb-4bit",
                True,
                "unsloth/gpt-oss-20b-bnb-4bit",
                False,
            ),
        ]
        for case in cases:
            if isinstance(case, str):
                model_name = case
                with self.subTest(model_name = model_name, load_in_4bit = True):
                    self._assert_mapping(model_name, True, model_name, False)
            else:
                model_name, load_in_4bit, expected, should_change = case
                with self.subTest(model_name = model_name, load_in_4bit = load_in_4bit):
                    self._assert_mapping(model_name, load_in_4bit, expected, should_change)

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_artifactory_report_preserves_repo_id_case(self):
        resolved = get_model_name(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            load_in_4bit = True,
        )
        self.assertEqual(
            resolved,
            "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_offline_load_reuses_legacy_lowercase_cache(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            canonical_cache = os.path.join(cache_dir, "models--" + canonical.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-commit")
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                ),
                legacy,
            )
            # An empty repo shell is not a usable cache entry.
            os.makedirs(canonical_cache)
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                ),
                legacy,
            )

            # Cached weights without tokenizer artifacts cannot satisfy the downstream load.
            _write_cached_model(canonical_cache, "canonical-weights-only", tokenizer = False)
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                ),
                legacy,
            )
            # A config-only sharded snapshot is also incomplete until every indexed shard exists.
            canonical_snapshot = _write_cached_model(
                canonical_cache, "canonical-commit", sharded = True, complete = False
            )
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                ),
                legacy,
            )
            open(os.path.join(canonical_snapshot, "model-00001-of-00001.safetensors"), "w").close()
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                ),
                canonical,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_offline_legacy_cache_matches_requested_revision(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            canonical_cache = os.path.join(cache_dir, "models--" + canonical.replace("/", "--"))
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(canonical_cache, "canonical-main", revision = "main")
            _write_cached_model(legacy_cache, "legacy-release", revision = "release/v1")

            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                    revision = "release/v1",
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_cross_repo_remap_probes_default_revision(self):
        canonical = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-main", revision = "main")

            self.assertEqual(
                get_model_name(
                    "Qwen/Qwen3-8B",
                    load_in_4bit = True,
                    cache_dir = cache_dir,
                    local_files_only = True,
                    revision = "source-only-ref",
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_vlm_cache_requires_processor_artifacts(self):
        canonical = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            canonical_cache = os.path.join(cache_dir, "models--" + canonical.replace("/", "--"))
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-main", vision = True)
            canonical_snapshot = _write_cached_model(
                canonical_cache,
                "canonical-main",
                vision = True,
                processor = False,
            )
            kwargs = dict(load_in_4bit = True, cache_dir = cache_dir, local_files_only = True)

            self.assertEqual(get_model_name("Qwen/Qwen2.5-VL-3B-Instruct", **kwargs), legacy)
            open(os.path.join(canonical_snapshot, "preprocessor_config.json"), "w").close()
            self.assertEqual(get_model_name("Qwen/Qwen2.5-VL-3B-Instruct", **kwargs), canonical)

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_external_tokenizer_allows_weight_only_model_cache(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-main", tokenizer = False)
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    cache_dir = cache_dir,
                    local_files_only = True,
                    require_tokenizer = False,
                    require_processor = False,
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_legacy_cache_accepts_all_supported_tokenizer_formats(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        for tokenizer_format in ("vocab.txt", "spiece.model"):
            with (
                self.subTest(tokenizer_format = tokenizer_format),
                tempfile.TemporaryDirectory() as cache_dir,
            ):
                legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
                _write_cached_model(
                    legacy_cache,
                    "legacy-main",
                    tokenizer_format = tokenizer_format,
                )
                self.assertEqual(
                    get_model_name(
                        "unsloth/Meta-Llama-3.1-8B-Instruct",
                        cache_dir = cache_dir,
                        local_files_only = True,
                    ),
                    legacy,
                )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_text_only_vlm_cache_does_not_require_processor(self):
        canonical = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-main", vision = True, processor = False)
            self.assertEqual(
                get_model_name(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    cache_dir = cache_dir,
                    local_files_only = True,
                    require_processor = False,
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_legacy_cache_honors_weight_subfolder_and_variant(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            snapshot = _write_cached_model(legacy_cache, "legacy-main")
            os.remove(os.path.join(snapshot, "model.safetensors"))
            weights = os.path.join(snapshot, "weights")
            os.makedirs(weights)
            open(os.path.join(weights, "model.fp16.safetensors"), "w").close()

            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    cache_dir = cache_dir,
                    local_files_only = True,
                    subfolder = "weights",
                    variant = "fp16",
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_legacy_cache_honors_requested_weight_format(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            canonical_cache = os.path.join(cache_dir, "models--" + canonical.replace("/", "--"))
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(canonical_cache, "canonical-main")
            legacy_snapshot = _write_cached_model(legacy_cache, "legacy-main")
            os.rename(
                os.path.join(legacy_snapshot, "model.safetensors"),
                os.path.join(legacy_snapshot, "pytorch_model.bin"),
            )
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    cache_dir = cache_dir,
                    local_files_only = True,
                    use_safetensors = False,
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_remote_code_cache_requires_referenced_module(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            canonical_cache = os.path.join(cache_dir, "models--" + canonical.replace("/", "--"))
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            canonical_snapshot = _write_cached_model(canonical_cache, "canonical-main")
            legacy_snapshot = _write_cached_model(legacy_cache, "legacy-main")
            config = {"auto_map": {"AutoModelForCausalLM": "modeling_custom.CustomModel"}}
            for snapshot in (canonical_snapshot, legacy_snapshot):
                with open(os.path.join(snapshot, "config.json"), "w", encoding = "utf-8") as file:
                    json.dump(config, file)
            open(os.path.join(legacy_snapshot, "modeling_custom.py"), "w").close()
            self.assertEqual(
                get_model_name(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    cache_dir = cache_dir,
                    local_files_only = True,
                    trust_remote_code = True,
                ),
                legacy,
            )

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_offline_legacy_cache_uses_transformers_default(self):
        canonical = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
        legacy = canonical.lower()
        with tempfile.TemporaryDirectory() as cache_dir:
            legacy_cache = os.path.join(cache_dir, "models--" + legacy.replace("/", "--"))
            _write_cached_model(legacy_cache, "legacy-commit")
            with patch("transformers.utils.hub.TRANSFORMERS_CACHE", cache_dir, create = True):
                self.assertEqual(
                    get_model_name(
                        "unsloth/Meta-Llama-3.1-8B-Instruct",
                        load_in_4bit = True,
                        local_files_only = True,
                    ),
                    legacy,
                )

    def test_static_mapper_contract(self):
        # A lowercased key is how __get_model_name always looks up; the value it
        # gets back is used verbatim as a repo id, so it must carry the casing
        # the repo actually has. Each expectation below was checked against the
        # Hub's canonical id. MAP_TO_UNSLOTH_16bit below already worked this way.
        contracts = [
            ("qwen/qwen3-8b", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
            ("qwen/qwen3-8b-fp8", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
            (
                "mistralai/ministral-3-3b-instruct-2512",
                "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit",
            ),
            (
                "allenai/olmo-3-7b-instruct",
                "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit",
            ),
            ("unsloth/kimi-k2-instruct", "unsloth/Kimi-K2-Instruct-BF16"),
        ]
        for src, expected in contracts:
            with self.subTest(src = src):
                self.assertEqual(FLOAT_TO_INT_MAPPER[src], expected)
        self.assertEqual(MAP_TO_UNSLOTH_16bit["qwen/qwen3-8b-fp8"], "unsloth/Qwen3-8B-FP8")
        self.assertEqual(
            MAP_TO_UNSLOTH_16bit["agentica-org/deepscaler-1.5b-preview"],
            "unsloth/DeepScaleR-1.5B-Preview",
        )


if __name__ == "__main__":
    unittest.main()
