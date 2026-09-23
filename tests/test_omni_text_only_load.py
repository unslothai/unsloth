# SPDX-License-Identifier: AGPL-3.0-only
"""Qwen3-Omni loads with text_only = True, and a kept wrapper still takes a text forward.

Qwen3OmniMoeConfig has no top-level vision_config (it lives under thinker_config) and no
causal-LM class, so text_only = True used to send it to AutoModelForCausalLM, which raised
"Unrecognized configuration class". A default load kept the wrapper, whose missing forward
made model(input_ids = ...) reach nn.Module.forward.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
omni_config = pytest.importorskip(
    "transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe",
    reason = "this transformers has no qwen3_omni_moe",
)
omni_modeling = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")

VOCAB = 64
# Unsloth's patched kernels are Triton on a GPU host, so the tiny model runs where they do.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_config():
    text = dict(
        vocab_size = VOCAB,
        hidden_size = 16,
        intermediate_size = 32,
        moe_intermediate_size = 8,
        shared_expert_intermediate_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
        num_experts = 2,
        num_experts_per_tok = 1,
        max_position_embeddings = 64,
        rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2], "interleaved": True},
    )
    config = omni_config.Qwen3OmniMoeConfig(
        enable_audio_output = True,
        thinker_config = dict(
            text_config = text,
            # Special ids above anything the tests feed in, so a text prompt stays text.
            audio_start_token_id = 56,
            audio_end_token_id = 57,
            audio_token_id = 58,
            image_token_id = 59,
            video_token_id = 60,
            vision_start_token_id = 61,
            vision_end_token_id = 62,
            audio_config = dict(
                num_mel_bins = 8,
                encoder_layers = 1,
                encoder_attention_heads = 2,
                encoder_ffn_dim = 16,
                d_model = 16,
                output_dim = 16,
                n_window = 2,
                n_window_infer = 4,
                downsample_hidden_size = 8,
                conv_chunksize = 10,
                max_source_positions = 16,
            ),
            vision_config = dict(
                depth = 1,
                hidden_size = 16,
                intermediate_size = 16,
                num_heads = 2,
                out_hidden_size = 16,
                patch_size = 2,
                spatial_merge_size = 1,
                temporal_patch_size = 1,
                deepstack_visual_indexes = [0],
            ),
        ),
        talker_config = dict(
            text_config = text,
            code_predictor_config = dict(
                vocab_size = 16,
                hidden_size = 16,
                intermediate_size = 16,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                num_key_value_heads = 1,
                head_dim = 8,
                num_code_groups = 2,
            ),
            thinker_hidden_size = 16,
            num_code_groups = 2,
            spatial_merge_size = 1,
        ),
        code2wav_config = dict(
            hidden_size = 16,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            intermediate_size = 16,
            codebook_size = 16,
            decoder_dim = 16,
            num_quantizers = 2,
            upsample_rates = (2,),
            upsampling_ratios = (2,),
            head_dim = 8,
            sliding_window = 8,
        ),
    )
    config.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    # transformers 5.4's top-level config does not declare the field its _init_weights reads.
    if getattr(config, "initializer_range", None) is None:
        config.initializer_range = 0.02
    return config


def _tiny_omni():
    torch.manual_seed(0)
    try:
        return omni_modeling.Qwen3OmniMoeForConditionalGeneration(_tiny_config()).to(DEVICE).eval()
    except Exception as error:  # a transformers whose Omni config takes other names
        pytest.skip(f"cannot build a tiny Qwen3-Omni here: {error}")


def _capture_fast_base_kwargs(monkeypatch, tmp_path, text_only):
    from unsloth import FastModel
    from unsloth.models.vision import FastBaseModel

    _tiny_config().save_pretrained(tmp_path)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(FastBaseModel, "from_pretrained", staticmethod(capture))
    try:
        FastModel.from_pretrained(str(tmp_path), text_only = text_only, load_in_4bit = False)
    except RuntimeError as error:
        if str(error) != "captured":
            if not torch.cuda.is_available():
                pytest.skip(f"FastModel.from_pretrained needs a GPU here: {error}")
            raise
    else:
        pytest.fail("FastModel.from_pretrained did not reach FastBaseModel.from_pretrained")
    return seen


def test_text_only_loads_qwen3_omni_through_its_own_auto_class(monkeypatch, tmp_path):
    from transformers import AutoModelForCausalLM
    from unsloth.models._utils import resolve_model_class

    seen = _capture_fast_base_kwargs(monkeypatch, tmp_path, text_only = True)
    auto_model = seen["auto_model"]
    assert auto_model is not AutoModelForCausalLM
    assert resolve_model_class(auto_model, _tiny_config()) is not None
    # The full composition loads; the caller's text intent hands it to the thinker.
    assert seen["text_only"] is False
    assert seen["text_intent"] is True
    assert seen["text_only_decoder"] is False


def test_the_default_load_resolves_the_same_class(monkeypatch, tmp_path):
    default = _capture_fast_base_kwargs(monkeypatch, tmp_path, text_only = False)
    text = _capture_fast_base_kwargs(monkeypatch, tmp_path, text_only = True)
    assert default["auto_model"] is text["auto_model"]
    assert default["text_intent"] is False


def test_text_intent_hands_qwen3_omni_to_its_thinker(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    from unsloth.models.vision import _text_trainable_core

    model = _tiny_omni()
    core = _text_trainable_core(model, text_intent = True)
    assert type(core).__name__ == "Qwen3OmniMoeThinkerForConditionalGeneration"
    # generate and save read the architecture off the core; a sub-config names none.
    assert core.config.architectures == ["Qwen3OmniMoeThinkerForConditionalGeneration"]
    ids = torch.tensor([[1, 2, 3, 4]], device = DEVICE)
    out = core(input_ids = ids, labels = ids)
    assert out.logits.shape == (1, 4, VOCAB)
    assert torch.isfinite(out.loss)
    generated = core.generate(input_ids = ids, max_new_tokens = 2, do_sample = False)
    assert generated.shape == (1, 6)


def test_a_kept_qwen3_omni_wrapper_forwards_through_its_thinker(capsys, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    from unsloth.models.vision import _text_trainable_core

    model = _tiny_omni()
    kept = _text_trainable_core(model, text_intent = False)
    assert kept is model
    # Nothing dropped: the talker still generates audio.
    assert hasattr(model, "talker") and hasattr(model, "code2wav")
    assert "text_only = True" in capsys.readouterr().out
    ids = torch.tensor([[1, 2, 3, 4]], device = DEVICE)
    with torch.no_grad():
        out = model(input_ids = ids, labels = ids)
        reference = model.thinker(input_ids = ids, labels = ids)
    assert torch.equal(out.logits, reference.logits)
    assert torch.isfinite(out.loss)
    assert model.get_output_embeddings() is model.thinker.lm_head
    assert model.get_input_embeddings() is model.thinker.get_input_embeddings()
    text = model.generate(
        input_ids = ids, return_audio = False, thinker_max_new_tokens = 2, do_sample = False
    )
    if isinstance(text, tuple):  # transformers 4.x returns (sequences, None) without audio
        text = text[0]
    assert text.shape == (1, 6)
    # deepcopy (and so a reference model for KD or DPO) follows the copy's own thinker.
    import copy

    clone = copy.deepcopy(model)
    assert clone.get_output_embeddings() is clone.thinker.lm_head
    assert clone.forward.__self__ is clone.thinker
    # Only this instance changed; the class still has no forward.
    assert type(model).forward is torch.nn.Module.forward
    fresh = _tiny_omni()
    with pytest.raises(TypeError, match = "input_ids"):
        fresh(input_ids = ids)


def test_a_kept_wrapper_trains_through_peft():
    peft = pytest.importorskip("peft")
    from unsloth.models.vision import _text_trainable_core

    model = _text_trainable_core(_tiny_omni(), text_intent = False)
    # Reentrant checkpointing with frozen embeddings trains LoRA only when the embedding
    # output requires grad, which goes through the wrapper's get_input_embeddings.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    config = peft.LoraConfig(
        r = 2, target_modules = r"thinker\.model\.layers\.\d+\.self_attn\.q_proj", lora_alpha = 4
    )
    peft_model = peft.get_peft_model(model, config)
    peft_model.train()
    ids = torch.tensor([[1, 2, 3, 4]], device = DEVICE)
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std = 0.1)
    loss = peft_model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = [p.grad for n, p in peft_model.named_parameters() if "lora_A" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
    assert peft_model.get_output_embeddings() is model.thinker.lm_head


def test_the_off_switch_leaves_the_wrapper_without_a_forward(monkeypatch):
    from unsloth.models.vision import _text_trainable_core

    monkeypatch.setenv("UNSLOTH_KEEP_COMPOSED_WRAPPER", "1")
    model = _tiny_omni()
    assert _text_trainable_core(model, text_intent = False) is model
    assert "forward" not in vars(model)


def test_a_kept_wrapper_scopes_lora_to_the_thinker_decoder():
    """get_peft_model's regex must reach thinker.model.layers, not only the vision tower."""
    import inspect
    import re
    from unsloth.models.vision import _text_core_decoder_prefix, _text_trainable_core
    from unsloth_zoo.peft_utils import get_peft_regex

    if "language_tags" not in inspect.signature(get_peft_regex).parameters:
        pytest.skip("this unsloth_zoo's get_peft_regex takes no language_tags")
    model = _text_trainable_core(_tiny_omni(), text_intent = False)
    assert _text_core_decoder_prefix(model) == "thinker.model"
    default_tags = inspect.signature(get_peft_regex).parameters["language_tags"].default
    regex = get_peft_regex(model, language_tags = list(default_tags) + [re.escape("thinker.model")])
    targets = [name for name, _ in model.named_modules() if re.fullmatch(regex, name)]
    assert any(name.startswith("thinker.model.layers.") for name in targets)
    assert not any(
        name.startswith(("talker.", "code2wav.", "thinker.audio_tower.")) for name in targets
    )
    # Without the core's prefix only the vision tower matched, so a text batch trained nothing.
    before = get_peft_regex(model)
    assert not any(
        name.startswith("thinker.model.layers.")
        for name, _ in model.named_modules()
        if re.fullmatch(before, name)
    )
    assert _text_core_decoder_prefix(_tiny_omni()) is None


def test_get_peft_model_passes_the_thinker_decoder_as_a_language_tag(monkeypatch):
    import inspect
    from unsloth.models import vision

    if "language_tags" not in inspect.signature(vision.get_peft_regex).parameters:
        pytest.skip("this unsloth_zoo's get_peft_regex takes no language_tags")
    import functools

    seen = {}

    # functools.wraps keeps the real signature, which get_peft_model inspects.
    @functools.wraps(vision.get_peft_regex)
    def capture(model, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(vision, "get_peft_regex", capture)
    model = vision._text_trainable_core(_tiny_omni(), text_intent = False)
    with pytest.raises(RuntimeError, match = "captured"):
        vision.FastBaseModel.get_peft_model(model, r = 2)
    assert "thinker\\.model" in seen["language_tags"]
    seen.clear()
    thinker = vision._text_trainable_core(_tiny_omni(), text_intent = True)
    with pytest.raises(RuntimeError, match = "captured"):
        vision.FastBaseModel.get_peft_model(thinker, r = 2)
    assert "language_tags" not in seen


def test_a_kept_wrapper_with_an_accelerate_hook_forwards_through_the_hook(monkeypatch):
    """A device-mapped load wraps forward on the instance; the hook must reach the thinker."""
    hooks = pytest.importorskip("accelerate.hooks")
    from unsloth.models.vision import _text_trainable_core

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    calls = []

    class Counting(hooks.ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            calls.append(type(module).__name__)
            return args, kwargs

    model = _tiny_omni()
    hooks.add_hook_to_module(model, Counting())
    model = _text_trainable_core(model, text_intent = False)
    ids = torch.tensor([[1, 2, 3, 4]], device = DEVICE)
    with torch.no_grad():
        out = model(input_ids = ids, labels = ids)
    assert torch.isfinite(out.loss)
    assert calls == ["Qwen3OmniMoeForConditionalGeneration"]


def _capture_adapter_reload(monkeypatch, tmp_path, target_modules):
    peft = pytest.importorskip("peft")
    from unsloth import FastModel
    from unsloth.models.vision import FastBaseModel

    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _tiny_config().save_pretrained(base)
    peft.LoraConfig(
        r = 2, lora_alpha = 2, target_modules = target_modules, base_model_name_or_path = str(base)
    ).save_pretrained(adapter)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(FastBaseModel, "from_pretrained", staticmethod(capture))
    try:
        FastModel.from_pretrained(str(adapter), load_in_4bit = False)
    except RuntimeError as error:
        if str(error) != "captured":
            if not torch.cuda.is_available():
                pytest.skip(f"FastModel.from_pretrained needs a GPU here: {error}")
            raise
    return seen


def test_an_adapter_trained_on_the_thinker_reloads_onto_the_thinker(monkeypatch, tmp_path):
    # A text_only adapter's regex is rooted at model.layers; the composition names them
    # thinker.model.layers, so a default reload could not find its targets.
    seen = _capture_adapter_reload(
        monkeypatch, tmp_path, r"(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn)\.(?:q_proj))"
    )
    assert seen["text_intent"] is True


def test_an_adapter_trained_on_the_kept_wrapper_keeps_the_wrapper(monkeypatch, tmp_path):
    seen = _capture_adapter_reload(
        monkeypatch, tmp_path, r"(?:.*?(?:thinker\.model).*?(?:self_attn).*?(?:q_proj))"
    )
    assert seen["text_intent"] is False


def _tiny_omni_checkpoint(path):
    tokenizers = pytest.importorskip("tokenizers")
    _tiny_omni().save_pretrained(path)
    vocab = {f"w{i}": i for i in range(48)}
    vocab.update({"<unk>": 48, "<pad>": 49, "<eos>": 50})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object = backend, unk_token = "<unk>", pad_token = "<pad>", eos_token = "<eos>"
    ).save_pretrained(path)


@pytest.mark.parametrize("text_only", [True, False], ids = ["thinker", "kept_wrapper"])
@pytest.mark.parametrize(
    "target_modules", [None, ["q_proj", "v_proj"]], ids = ["unsloth_regex", "leaf_list"]
)
def test_an_omni_adapter_reloads_onto_the_model_it_was_trained_on(
    tmp_path, text_only, target_modules
):
    # A leaf-name list reloads as a set that matches either layout, so the saved weight
    # keys (model.layers vs thinker.model.layers) are what decide.
    if not torch.cuda.is_available():
        pytest.skip("FastModel needs a GPU")
    from unsloth import FastModel

    base = tmp_path / "base"
    _tiny_omni_checkpoint(base)
    model, _ = FastModel.from_pretrained(str(base), text_only = text_only, load_in_4bit = False)
    kwargs = {} if target_modules is None else {"target_modules": target_modules}
    model = FastModel.get_peft_model(model, r = 2, lora_alpha = 2, random_state = 0, **kwargs)
    trained_on = type(model.get_base_model()).__name__
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.1)
    saved = {
        n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if "lora_" in n
    }
    assert saved
    model.save_pretrained(tmp_path / "adapter")
    del model
    reloaded, _ = FastModel.from_pretrained(str(tmp_path / "adapter"), load_in_4bit = False)
    assert type(reloaded.get_base_model()).__name__ == trained_on
    loaded = {n: p.detach().float().cpu() for n, p in reloaded.named_parameters() if "lora_" in n}
    assert loaded.keys() == saved.keys()
    for name, value in saved.items():
        assert torch.equal(loaded[name], value), name


def test_saved_weight_keys_decide_over_the_target_regex():
    from unsloth.models.loader import _adapter_targets_text_core

    config = type("Config", (), {"target_modules": {"q_proj", "v_proj"}})()
    thinker_keys = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    wrapper_keys = ["base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert _adapter_targets_text_core(config, thinker_keys) is True
    assert _adapter_targets_text_core(config, wrapper_keys) is False
    # Without keys a leaf list cannot tell the layouts apart, so the composition is kept.
    assert _adapter_targets_text_core(config, None) is False
    regex = type("Config", (), {"target_modules": r".*\.q_proj"})()
    assert _adapter_targets_text_core(regex, wrapper_keys) is False


def test_hub_adapter_keys_come_from_adapter_model_safetensors(monkeypatch):
    # get_safetensors_metadata only looks for model.safetensors, which an adapter repo lacks.
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from unsloth.models.loader import _adapter_weight_keys

    asked = {}

    def fake(self, repo_id, filename, **kwargs):
        asked.update(repo_id = repo_id, filename = filename, **kwargs)
        return type("Metadata", (), {"tensors": {"base_model.model.model.layers.0.q_proj.lora_A.weight": None}})()

    monkeypatch.setattr(huggingface_hub.HfApi, "parse_safetensors_file_metadata", fake)
    keys = _adapter_weight_keys("someone/omni-adapter", token = "t", revision = "r")
    assert asked == {"repo_id": "someone/omni-adapter", "filename": "adapter_model.safetensors", "revision": "r", "token": "t"}
    assert keys == ["base_model.model.model.layers.0.q_proj.lora_A.weight"]
    assert _adapter_weight_keys("someone/omni-adapter", local_files_only = True) is None


def test_local_bin_adapter_keys_are_read_without_weights(tmp_path):
    from unsloth.models.loader import _adapter_weight_keys

    key = "base_model.model.thinker.model.layers.0.q_proj.lora_A.weight"
    torch.save({key: torch.ones(2, 2)}, tmp_path / "adapter_model.bin")
    assert _adapter_weight_keys(str(tmp_path)) == [key]
    assert _adapter_weight_keys(str(tmp_path / "missing")) is None
