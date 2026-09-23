# SPDX-License-Identifier: AGPL-3.0-only
"""An untrusted load of a VLM whose repo auto_map names only AutoModelForCausalLM.

stepfun-ai/Step-3.7-Flash ships remote code registered under AutoModelForCausalLM, while
transformers itself implements the same checkpoint (model_type step3p7) as an image-text class.
FastModel took the auto_map entry whatever the trust decision, so trust_remote_code = False (the
default) asked AutoModelForCausalLM to build the repo's class and transformers refused the load
("contains custom code which must be executed"). Untrusted, only auto_map entries transformers can
build natively are considered now; a trusted load still takes the repo's class.
"""

import json

import pytest
import torch
from real_accelerator import has_real_accelerator


step3p7 = pytest.importorskip("transformers.models.step3p7.modeling_step3p7")


def _write_tokenizer(path, vocab_size):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tok, unk_token = "t0", pad_token = "t1", eos_token = "t2"
    ).save_pretrained(path)


def _tiny_step3p7(path, auto_map):
    from transformers.models.step3p7.configuration_step3p7 import Step3p7Config

    vocab = 64
    config = Step3p7Config(
        text_config = dict(
            vocab_size = vocab,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 32,
            n_routed_experts = 4,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            share_expert_dim = 32,
            layer_types = ["full_attention", "full_attention"],
            mlp_layer_types = ["dense", "sparse"],
            max_position_embeddings = 256,
        ),
        vision_config = dict(width = 32, layers = 1, heads = 2, image_size = 28, patch_size = 14),
    )
    torch.manual_seed(0)
    model = step3p7.Step3p7ForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(path)
    cfg = json.loads((path / "config.json").read_text(encoding = "utf-8"))
    # What the hub repo declares: its own class under AutoModelForCausalLM. The module is never
    # shipped here, so any attempt to build it fails loudly.
    cfg["auto_map"] = {
        name: "modeling_step3p7.Step3p7ForConditionalGeneration" for name in auto_map
    }
    (path / "config.json").write_text(json.dumps(cfg), encoding = "utf-8")
    _write_tokenizer(path, vocab)
    return path


def test_native_class_lookup():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.step3p7.configuration_step3p7 import Step3p7Config

    loader = pytest.importorskip("unsloth.models.loader")
    _config_has_native_class = loader._config_has_native_class

    assert _config_has_native_class(AutoModelForImageTextToText, Step3p7Config())
    assert not _config_has_native_class(AutoModelForCausalLM, Step3p7Config())
    assert _config_has_native_class(AutoModelForCausalLM, LlamaConfig())
    assert not _config_has_native_class(None, LlamaConfig())


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_accelerator(), reason = "import unsloth needs an accelerator")
# Kimi-K2.5 also names its class under AutoModel; native AutoModel is the headless backbone.
@pytest.mark.parametrize(
    "auto_map",
    [("AutoModelForCausalLM",), ("AutoModel", "AutoModelForCausalLM")],
    ids = ["causal_lm_only", "auto_model_and_causal_lm"],
)
def test_untrusted_load_builds_the_native_image_text_class(tmp_path, auto_map):
    import unsloth  # noqa: F401
    from unsloth import FastModel

    path = _tiny_step3p7(tmp_path / "tiny_step3p7", auto_map)
    model, _ = FastModel.from_pretrained(
        str(path),
        max_seq_length = 64,
        dtype = torch.bfloat16,
        load_in_4bit = False,
        load_in_16bit = True,
        trust_remote_code = False,
        text_only = True,
    )
    assert type(model).__module__ == step3p7.__name__
    assert type(model).__name__ == "Step3p7ForConditionalGeneration"
