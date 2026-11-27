# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Turn padding-free off only where the forward signature shows it cannot accept packed_seq_lengths."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

try:
    from unsloth.trainer import _forward_accepts_packing_kwargs  # noqa: E402
except ImportError:
    # On MLX, unsloth.trainer is a shim without these helpers, so skip rather than fail.
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)


@pytest.fixture(autouse = True)
def _return_logits_unset(monkeypatch):
    """UNSLOTH_RETURN_LOGITS=1 blocks packing and padding-free outright, so with it set the
    gate never runs and these tests see neither the refusal nor the metadata hook. Product
    paths set it process-wide (the eval prediction_step, zoo's compiler), and
    an earlier test in the same xdist worker leaving it at "1" failed a different subset of
    this file on each run. Tests that want it set still say so with monkeypatch.setenv."""
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)


class _NoKwargs(nn.Module):
    """microsoft/Phi-4-reasoning-vision-15B's shape: no **kwargs, no packing."""

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        labels = None,
    ):
        return None


class _TakesKwargs(nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return None


class _NamesPacking(nn.Module):
    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
    ):
        return None


class _FakePeft(nn.Module):
    """PEFT forwards **kwargs through, so the wrapper says yes for a checkpoint that does not."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_base_model(self):
        return self.inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class _InnerDecoderTakesKwargs(nn.Module):
    """`PreTrainedModel.base_model` reaches the inner decoder, which usually does take
    **kwargs, so following it answers for the wrong module."""

    def __init__(self):
        super().__init__()
        self.model = _TakesKwargs()

    @property
    def base_model(self):
        return self.model

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
    ):
        return None


class _SelfReturningUnwrap(nn.Module):
    def get_base_model(self):
        return self

    def forward(self, input_ids = None):
        return None


class _RaisingUnwrap(nn.Module):
    def get_base_model(self):
        raise RuntimeError("adapter not ready")

    def forward(self, input_ids = None):
        return None


@pytest.mark.parametrize(
    "model, expected, why",
    [
        (_NoKwargs(), False, "the real defect"),
        (_TakesKwargs(), True, "**kwargs absorbs it"),
        (_NamesPacking(), True, "names it explicitly"),
        (_FakePeft(_NoKwargs()), False, "unwrap past the adapter"),
        (_FakePeft(_TakesKwargs()), True, "unwrapped model accepts it"),
        (_InnerDecoderTakesKwargs(), False, "base_model must NOT be followed"),
        (_SelfReturningUnwrap(), False, "self-returning unwrap terminates"),
        (_RaisingUnwrap(), False, "a raising unwrap falls back to the wrapper"),
        (None, True, "unknown fails open"),
        ("meta-llama/Llama-3.1-8B", True, "a name, not a model: fails open"),
        (object(), True, "no forward at all: fails open"),
    ],
)
def test_gate(model, expected, why):
    assert _forward_accepts_packing_kwargs(model) is expected, why


def test_the_defect_is_real():
    """Negative control: the rejected shape really does raise."""
    with pytest.raises(TypeError, match = "packed_seq_lengths"):
        _NoKwargs()(input_ids = torch.zeros(1, 4).long(), packed_seq_lengths = [4])


def test_the_accepted_shape_really_accepts_it():
    """And the shape the gate allows really does tolerate the argument."""
    _TakesKwargs()(input_ids = torch.zeros(1, 4).long(), packed_seq_lengths = [4])


def test_the_blocker_names_itself_in_the_warning(monkeypatch, caplog):
    """Run with UNSLOTH_RETURN_LOGITS also set, so the warning must name the real blocker, not the flag."""
    import logging
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = model
            self.args = args

    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: None)
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = True, padding_free = None, max_length = 512)
    with caplog.at_level(logging.WARNING, logger = trainer_module.logger.name):
        module.SFTTrainer(model = _NoKwargs(), args = config)

    assert "packing=True ignored" in caplog.text, "the reason chain never emitted"
    assert (
        "_NoKwargs.forward()" in caplog.text
    ), "the model's forward is what blocks packing, so the warning has to say so"
    assert (
        "UNSLOTH_RETURN_LOGITS" not in caplog.text
    ), "naming the env var blames a flag the user did not set for a model-shape blocker"


def test_the_signature_probe_runs_once_per_call():
    """The signature probe can reach remote-code resolution, so it must run once per call."""
    import inspect as _inspect

    from unsloth import trainer as trainer_module

    source = _inspect.getsource(trainer_module._patch_sft_trainer_auto_packing)
    assert source.count("_forward_accepts_packing_kwargs(model)") == 1


@pytest.mark.parametrize("packing", [False, True])
def test_a_string_model_is_rechecked_once_trl_has_built_it(monkeypatch, packing):
    """Unresolved string models are rechecked after init and refused, since flags alone lose boundaries."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    built = _NoKwargs()
    inits = []

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = built if isinstance(model, str) else model
            self.args = args
            inits.append(model)

    injected = []
    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: injected.append(model))
    monkeypatch.setattr(trainer_module, "_resolve_string_model_config", lambda *a, **k: None)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = packing, padding_free = True, max_length = 512)
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        module.SFTTrainer(model = "microsoft/Phi-4-reasoning-vision-15B", args = config)

    assert injected == [], "nothing may wrap the collator for this forward"
    assert len(inits) == 1, "the checkpoint must not be materialized a second time"


def test_a_string_model_that_can_take_the_metadata_is_left_alone(monkeypatch):
    """The refusal must fire only on the signature, never on the string itself."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _TakesKwargs() if isinstance(model, str) else model
            self.args = args

    injected = []
    monkeypatch.setattr(
        trainer_module,
        "enable_padding_free_metadata",
        lambda model, trainer: injected.append(model),
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_config", lambda *a, **k: None)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = False, padding_free = True, max_length = 512)
    instance = module.SFTTrainer(model = "meta-llama/Llama-3.1-8B", args = config)

    assert len(injected) == 1
    assert instance.args.padding_free is True


def _fake_config():
    """A config just real enough for the model-type lookup the wrapper does first."""
    from types import SimpleNamespace
    return SimpleNamespace(
        model_type = "llama",
        architectures = ["LlamaForCausalLM"],
        auto_map = None,
        is_encoder_decoder = False,
        to_dict = lambda: {"model_type": "llama"},
    )


def test_the_class_behind_a_string_is_resolved_without_downloading_weights():
    """`from_pretrained` is never called, so a config built in memory is enough."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from unsloth.trainer import _resolve_string_model_class

    config = LlamaConfig(architectures = ["LlamaForCausalLM"])
    assert _resolve_string_model_class("any/name", config, None) is LlamaForCausalLM

    bare = LlamaConfig()
    bare.architectures = None
    assert _resolve_string_model_class("any/name", bare, None) is LlamaForCausalLM


def test_a_native_architecture_is_answered_without_touching_remote_code():
    """Native config.architectures wins over auto_map, as TRL resolves it, so the loader is never called."""
    from types import SimpleNamespace

    import transformers
    from transformers import LlamaForCausalLM

    from unsloth.trainer import _resolve_string_model_class

    called = []

    class _Config:
        model_type = "llama"
        architectures = ["LlamaForCausalLM"]
        auto_map = {"AutoModelForCausalLM": "payload.Model"}

    import transformers.dynamic_module_utils as dmu

    original = dmu.get_class_from_dynamic_module
    dmu.get_class_from_dynamic_module = lambda *a, **k: called.append(a) or _NoKwargs
    try:
        resolved = _resolve_string_model_class(
            "evil/repo", _Config(), SimpleNamespace(trust_remote_code = True)
        )
    finally:
        dmu.get_class_from_dynamic_module = original

    assert resolved is LlamaForCausalLM
    assert called == [], "the remote module must not be imported when a native name answers"


@pytest.mark.parametrize(
    "init_kwargs, top_level, may_execute",
    [
        ({}, True, True),
        ({}, False, False),
        ({}, None, False),
        # An explicit None is not a grant and must match _resolve_string_model_config's reading.
        ({"trust_remote_code": None}, True, False),
        ({"trust_remote_code": False}, True, False),
        ({"trust_remote_code": True}, False, True),
    ],
)
def test_the_remote_code_grant_is_read_by_membership(init_kwargs, top_level, may_execute):
    from types import SimpleNamespace

    from unsloth.trainer import _resolve_string_model_class

    called = []

    class _RemoteOnlyConfig:
        model_type = "not-a-real-model-type"
        architectures = ["NoSuchClassInTransformers"]
        auto_map = {"AutoModelForCausalLM": "payload.Model"}

    import transformers.dynamic_module_utils as dmu

    original = dmu.get_class_from_dynamic_module
    dmu.get_class_from_dynamic_module = lambda *a, **k: called.append(a) or _NoKwargs
    try:
        _resolve_string_model_class(
            "some/repo",
            _RemoteOnlyConfig(),
            SimpleNamespace(model_init_kwargs = init_kwargs, trust_remote_code = top_level),
        )
    finally:
        dmu.get_class_from_dynamic_module = original

    assert bool(called) is may_execute


def test_the_same_auth_keys_reach_both_fetches():
    """Config and class fetches must send the same auth keys, incl. the deprecated use_auth_token alias."""
    import inspect as _inspect
    from types import SimpleNamespace

    from unsloth import trainer as trainer_module
    from unsloth.trainer import _resolve_string_model_class

    sibling = _inspect.getsource(trainer_module._resolve_string_model_config)
    resolver = _inspect.getsource(_resolve_string_model_class)
    for key in ("revision", "subfolder", "token", "use_auth_token", "cache_dir", "code_revision"):
        assert f'"{key}"' in sibling, f"{key} is not forwarded by the config fetch"
        assert f'"{key}"' in resolver, f"{key} is not forwarded by the class fetch"

    captured = {}

    class _RemoteOnlyConfig:
        model_type = "not-a-real-model-type"
        architectures = ["NoSuchClassInTransformers"]
        auto_map = {"AutoModelForCausalLM": "payload.Model"}

    import transformers.dynamic_module_utils as dmu

    original = dmu.get_class_from_dynamic_module

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return _NoKwargs

    dmu.get_class_from_dynamic_module = _capture
    try:
        _resolve_string_model_class(
            "private/repo",
            _RemoteOnlyConfig(),
            SimpleNamespace(
                model_init_kwargs = {
                    "trust_remote_code": True,
                    "use_auth_token": "hf_legacy",
                    "revision": "abc123",
                }
            ),
        )
    finally:
        dmu.get_class_from_dynamic_module = original

    assert captured.get("use_auth_token") == "hf_legacy"
    assert captured.get("revision") == "abc123"
    assert "trust_remote_code" not in captured


def test_an_unresolvable_string_returns_none_rather_than_guessing():
    """None leaves the post-init backstop in charge, which is the safe answer."""
    from transformers import LlamaConfig

    from unsloth.trainer import _resolve_string_model_class

    assert _resolve_string_model_class("any/name", None, None) is None
    assert _resolve_string_model_class(_NoKwargs(), LlamaConfig(), None) is None

    class _UnknownConfig:
        architectures = ["NoSuchModelForCausalLM"]
        auto_map = None

    assert _resolve_string_model_class("any/name", _UnknownConfig(), None) is None


def test_a_resolvable_string_is_blocked_silently_instead_of_raising(monkeypatch):
    """No exception, just padding-free turned off: the user configured nothing."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _NoKwargs() if isinstance(model, str) else model
            self.args = args

    injected = []
    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: injected.append(model))
    monkeypatch.setattr(
        trainer_module, "_resolve_string_model_config", lambda *a, **k: _fake_config()
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_class", lambda *a, **k: _NoKwargs)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = False, padding_free = None, max_length = 512)
    instance = module.SFTTrainer(model = "microsoft/Phi-4-reasoning-vision-15B", args = config)

    assert instance.args.padding_free is False, "padding-free must be off, not fatal"
    assert injected == [], "nothing may wrap the collator for this forward"


def test_a_resolvable_string_that_accepts_the_metadata_keeps_padding_free(monkeypatch):
    """The control: resolving early must not cost a normal checkpoint its padding-free."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _TakesKwargs() if isinstance(model, str) else model
            self.args = args

    injected = []
    monkeypatch.setattr(
        trainer_module,
        "enable_padding_free_metadata",
        lambda model, trainer: injected.append(model),
    )
    monkeypatch.setattr(
        trainer_module, "_resolve_string_model_config", lambda *a, **k: _fake_config()
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_class", lambda *a, **k: _TakesKwargs)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = False, padding_free = True, max_length = 512)
    instance = module.SFTTrainer(model = "meta-llama/Llama-3.1-8B", args = config)

    assert instance.args.padding_free is True
    assert len(injected) == 1


def test_the_warning_names_the_resolved_class_not_str(monkeypatch, caplog):
    """`str.forward()` would name the spelling rather than the checkpoint."""
    import logging
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _NoKwargs() if isinstance(model, str) else model
            self.args = args

    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: None)
    monkeypatch.setattr(
        trainer_module, "_resolve_string_model_config", lambda *a, **k: _fake_config()
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_class", lambda *a, **k: _NoKwargs)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = True, padding_free = None, max_length = 512)
    with caplog.at_level(logging.WARNING, logger = trainer_module.logger.name):
        module.SFTTrainer(model = "microsoft/Phi-4-reasoning-vision-15B", args = config)

    assert "_NoKwargs.forward()" in caplog.text
    assert "str.forward()" not in caplog.text


def test_a_class_that_disagrees_with_the_built_model_is_still_caught(monkeypatch):
    """A resolved class is not proof about the built instance, so the post-init check stays armed."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _NoKwargs() if isinstance(model, str) else model
            self.args = args

    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: None)
    monkeypatch.setattr(
        trainer_module, "_resolve_string_model_config", lambda *a, **k: _fake_config()
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_class", lambda *a, **k: _TakesKwargs)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = False, padding_free = True, max_length = 512)
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        module.SFTTrainer(model = "some/multi-headed-checkpoint", args = config)


def test_a_resolver_that_explodes_falls_back_to_the_backstop(monkeypatch):
    """Resolution is best-effort; a failure must not become the user's problem."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model = _NoKwargs() if isinstance(model, str) else model
            self.args = args

    def _explode(*a, **k):
        raise RuntimeError("hub is down")

    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: None)
    monkeypatch.setattr(
        trainer_module, "_resolve_string_model_config", lambda *a, **k: _fake_config()
    )
    monkeypatch.setattr(trainer_module, "_resolve_string_model_class", _explode)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = False, padding_free = True, max_length = 512)
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        module.SFTTrainer(model = "microsoft/Phi-4-reasoning-vision-15B", args = config)


def test_a_mixed_adapter_wrapper_is_unwrapped_to_the_checkpoint():
    """PeftMixedModel has no get_base_model, so the gate must unwrap it to the checkpoint."""
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from unsloth.trainer import _forward_accepts_packing_kwargs as gate

    def _config():
        return transformers.LlamaConfig(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            vocab_size = 64,
        )

    class _NarrowLlama(transformers.LlamaForCausalLM):
        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            labels = None,
        ):
            return super().forward(
                input_ids = input_ids, attention_mask = attention_mask, labels = labels
            )

    lora = peft.LoraConfig(target_modules = ["q_proj"])
    mixed_narrow = peft.get_peft_model(_NarrowLlama(_config()), lora, mixed = True)
    mixed_stock = peft.get_peft_model(transformers.LlamaForCausalLM(_config()), lora, mixed = True)

    assert type(mixed_narrow).__name__ == "PeftMixedModel"
    assert not hasattr(mixed_narrow, "get_base_model")

    assert gate(mixed_narrow) is False, "must see through to the narrow checkpoint"
    assert gate(mixed_stock) is True, "and must not cost a stock checkpoint its padding-free"


def test_every_delegating_wrapper_is_unwrapped():
    """Driven over the whole registered family, so a wrapper added without an unwrap fails here."""
    from unsloth.trainer import _DELEGATING_MODULE_WRAPPERS
    from unsloth.trainer import _forward_accepts_packing_kwargs as gate

    assert _DELEGATING_MODULE_WRAPPERS, "nothing resolved: the gate would read the wrapper"
    for wrapper, attribute in _DELEGATING_MODULE_WRAPPERS:
        for inner, expected in ((_NoKwargs(), False), (_TakesKwargs(), True)):
            # Subclass: FSDP's `module` is read-only. Uninitialised: its __setattr__ rejects stubs.
            stub = type("_Stub", (wrapper,), {attribute: inner})
            wrapped = stub.__new__(stub)

            assert gate(inner) is expected, f"{wrapper.__name__}: control"
            assert gate(wrapped) is expected, wrapper.__name__
