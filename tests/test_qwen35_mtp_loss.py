import inspect
import types

import torch

from unsloth.models import mtp


def _fast_cross_entropy_loss(logits, labels, **_kwargs):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index = -100,
    )


def _load_mtp_loss_helpers():
    namespace = {name: getattr(mtp, name) for name in dir(mtp) if not name.startswith("_")}
    namespace["fast_cross_entropy_loss"] = _fast_cross_entropy_loss
    return namespace


class _MTPHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias = False)

    def forward(self, hidden_states, **_kwargs):
        return self.proj(hidden_states)


class _DictMTPHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias = False)

    def forward(self, hidden_states):
        return {"logits": self.proj(hidden_states)}


class _ListMTPHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj_2 = torch.nn.Linear(hidden_size, vocab_size, bias = False)
        self.proj_3 = torch.nn.Linear(hidden_size, vocab_size, bias = False)

    def forward(self, hidden_states, **_kwargs):
        return [self.proj_2(hidden_states), self.proj_3(hidden_states)]


class _RequiredInputsMTPHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias = False)
        self.seen = None

    def forward(self, hidden_states, input_ids, position_ids, embed_fn):
        self.seen = (input_ids, position_ids, embed_fn)
        embeddings = embed_fn(input_ids).to(hidden_states.dtype)
        hidden_states = hidden_states + 0.0 * embeddings
        return self.proj(hidden_states)


class _AttentionMaskMTPHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias = False)
        self.seen_attention_mask = None

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        **_kwargs,
    ):
        self.seen_attention_mask = attention_mask
        return self.proj(hidden_states)


class _ModelOutput:
    def __init__(self, loss, hidden_states):
        self.loss = loss
        self.hidden_states = hidden_states

    def to_tuple(self):
        output = (self.loss,)
        if self.hidden_states is not None:
            output = output + (self.hidden_states,)
        return output


def test_qwen35_mtp_loss_auto_enables_and_backprops():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(
        model_type = "qwen3_5_moe_text",
        vocab_size = vocab_size,
        mtp_loss_weight = 0.5,
    )
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = torch.nn.ModuleList(
        [_MTPHead(hidden_size, vocab_size), _MTPHead(hidden_size, vocab_size)]
    )
    hidden_states = torch.randn(2, 5, hidden_size, requires_grad = True)
    labels = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ],
        dtype = torch.long,
    )

    loss = helpers["compute_mtp_loss"](
        model,
        hidden_states,
        labels,
        loss_fn = helpers["fast_cross_entropy_loss"],
    )

    assert loss is not None
    loss.backward()
    assert model.mtp[0].proj.weight.grad is not None
    assert model.mtp[1].proj.weight.grad is not None


def test_mtp_loss_does_not_enable_for_other_models_by_default():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "llama", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = torch.nn.ModuleList([_MTPHead(hidden_size, vocab_size)])

    loss = helpers["compute_mtp_loss"](
        model,
        torch.randn(1, 4, hidden_size),
        torch.tensor([[0, 1, 2, 3]], dtype = torch.long),
        loss_fn = helpers["fast_cross_entropy_loss"],
    )

    assert loss is None


def test_mtp_shift_labels_mask_packed_boundaries_for_offset():
    helpers = _load_mtp_loss_helpers()
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype = torch.long)

    shifted = helpers["make_mtp_shift_labels"](
        labels,
        2,
        packed_seq_lengths = [3, 3],
    )

    assert shifted.tolist() == [[3, -100, -100, 6, -100, -100]]


def test_mtp_shift_labels_are_contiguous_with_non_contiguous_labels():
    helpers = _load_mtp_loss_helpers()
    labels = torch.arange(12, dtype = torch.long).reshape(2, 6)[:, ::2]

    shifted = helpers["make_mtp_shift_labels"](labels, 1)

    assert not labels.is_contiguous()
    assert shifted.is_contiguous()


def test_mtp_loss_accepts_dict_outputs_and_filters_kwargs():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "qwen3_5_moe_text", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = torch.nn.ModuleList([_DictMTPHead(hidden_size, vocab_size)])

    loss = helpers["compute_mtp_loss"](
        model,
        torch.randn(1, 4, hidden_size),
        torch.tensor([[0, 1, 2, 3]], dtype = torch.long),
        loss_fn = helpers["fast_cross_entropy_loss"],
        unknown_kwarg = object(),
    )

    assert loss is not None


def test_mtp_loss_preserves_all_depths_from_list_output():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "qwen3_5_moe_text", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = _ListMTPHead(hidden_size, vocab_size)
    hidden_states = torch.randn(1, 5, hidden_size, requires_grad = True)
    labels = torch.tensor([[0, 1, 2, 3, 4]], dtype = torch.long)

    loss = helpers["compute_mtp_loss"](
        model,
        hidden_states,
        labels,
        loss_fn = helpers["fast_cross_entropy_loss"],
    )

    assert loss is not None
    loss.backward()
    assert model.mtp.proj_2.weight.grad is not None
    assert model.mtp.proj_3.weight.grad is not None


def test_mtp_loss_passes_token_position_and_embed_inputs():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "qwen3_5_moe_text", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    model.mtp = _RequiredInputsMTPHead(hidden_size, vocab_size)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype = torch.long)
    position_ids = torch.arange(4, dtype = torch.long).unsqueeze(0)

    loss = helpers["compute_mtp_loss"](
        model,
        torch.randn(1, 4, hidden_size),
        input_ids,
        loss_fn = helpers["fast_cross_entropy_loss"],
        input_ids = input_ids,
        position_ids = position_ids,
    )

    assert loss is not None
    assert model.mtp.seen == (input_ids, position_ids, model.model.embed_tokens)


def test_patch_mtp_loss_preserves_signature_and_return_dict_false():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7

    class _PatchedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                model_type = "qwen3_5_moe_text",
                vocab_size = vocab_size,
                output_hidden_states = False,
            )
            self.vocab_size = vocab_size
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.mtp = torch.nn.ModuleList([_MTPHead(hidden_size, vocab_size)])

        def forward(
            self,
            input_ids = None,
            labels = None,
            return_dict = True,
            output_hidden_states = False,
            **_kwargs,
        ):
            hidden_states = self.model.embed_tokens(input_ids)
            loss = hidden_states.sum() * 0.0 + 1.0
            hidden_states = (hidden_states,) if output_hidden_states else None
            if return_dict:
                return _ModelOutput(loss, hidden_states)
            return (loss, hidden_states)

    model = helpers["patch_mtp_loss"](_PatchedModel(), helpers["fast_cross_entropy_loss"])
    signature = inspect.signature(model.forward)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype = torch.long)
    labels = torch.tensor([[0, 1, 2, 3]], dtype = torch.long)

    outputs = model(
        input_ids = input_ids,
        labels = labels,
        return_dict = False,
    )

    assert "input_ids" in signature.parameters
    assert "labels" in signature.parameters
    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert outputs[0].item() > 1.0


def test_filter_mtp_kwargs_drops_explicitly_passed_arguments():
    helpers = _load_mtp_loss_helpers()
    kwargs = {
        "cache_position": object(),
        "position_embeddings": object(),
        "n_items": 4,
        "num_items_in_batch": 4,
        "packed_seq_lengths": [2, 2],
        "use_mtp_loss": True,
        "attention_mask": object(),
        "unknown_kwarg": object(),
    }

    filtered = helpers["filter_mtp_kwargs"](kwargs)

    assert filtered == {"unknown_kwarg": kwargs["unknown_kwarg"]}


def test_compute_mtp_loss_accepts_caller_kwarg_expansion():
    # Mirrors the llama.py fast forward: explicit arguments plus **filter_mtp_kwargs(kwargs).
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "qwen3_5_moe_text", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = torch.nn.ModuleList([_MTPHead(hidden_size, vocab_size)])
    kwargs = {
        "cache_position": torch.arange(4),
        "position_embeddings": (torch.randn(1, 4, 2), torch.randn(1, 4, 2)),
        "n_items": 4,
    }

    loss = helpers["compute_mtp_loss"](
        model,
        torch.randn(1, 4, hidden_size),
        torch.tensor([[0, 1, 2, 3]], dtype = torch.long),
        loss_fn = helpers["fast_cross_entropy_loss"],
        n_items = kwargs["n_items"],
        cache_position = kwargs["cache_position"],
        position_embeddings = kwargs["position_embeddings"],
        **helpers["filter_mtp_kwargs"](kwargs),
    )

    assert loss is not None


def test_mtp_loss_weight_falls_back_to_scaling_factor():
    helpers = _load_mtp_loss_helpers()
    scaling_model = types.SimpleNamespace(config = types.SimpleNamespace(mtp_loss_scaling_factor = 0.1))
    nested_model = types.SimpleNamespace(
        config = types.SimpleNamespace(mtp_config = {"loss_scaling_factor": 0.25})
    )
    plain_model = types.SimpleNamespace(config = types.SimpleNamespace())

    assert helpers["get_mtp_loss_weight"](scaling_model) == 0.1
    assert helpers["get_mtp_loss_weight"](nested_model) == 0.25
    assert helpers["get_mtp_loss_weight"](plain_model) == 1.0
    assert helpers["get_mtp_loss_weight"](scaling_model, 0.7) == 0.7


def test_mtp_loss_scaling_factor_is_applied():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7

    def _build_model(**config_kwargs):
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.config = types.SimpleNamespace(
            model_type = "qwen3_5_moe_text",
            vocab_size = vocab_size,
            **config_kwargs,
        )
        model.vocab_size = vocab_size
        model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
        model.mtp = torch.nn.ModuleList([_MTPHead(hidden_size, vocab_size)])
        return model

    torch.manual_seed(1)
    hidden_states = torch.randn(1, 5, hidden_size)
    labels = torch.tensor([[0, 1, 2, 3, 4]], dtype = torch.long)

    unscaled = helpers["compute_mtp_loss"](
        _build_model(),
        hidden_states,
        labels,
        loss_fn = helpers["fast_cross_entropy_loss"],
    )
    scaled = helpers["compute_mtp_loss"](
        _build_model(mtp_loss_scaling_factor = 0.1),
        hidden_states,
        labels,
        loss_fn = helpers["fast_cross_entropy_loss"],
    )

    assert torch.allclose(scaled, unscaled * 0.1)


def test_packed_attention_mask_is_block_causal():
    helpers = _load_mtp_loss_helpers()
    mask = helpers["build_mtp_packed_attention_mask"](
        [2, 2],
        4,
        torch.float32,
        torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 4, 4)
    allowed = mask[0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_packed_attention_mask_skips_non_packed_batches():
    helpers = _load_mtp_loss_helpers()

    assert (
        helpers["build_mtp_packed_attention_mask"](None, 4, torch.float32, torch.device("cpu"))
        is None
    )
    # Lengths that do not cover the flattened batch are not a packed sequence.
    assert (
        helpers["build_mtp_packed_attention_mask"]([2, 2], 6, torch.float32, torch.device("cpu"))
        is None
    )


def test_mtp_module_receives_packed_block_causal_mask():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type = "qwen3_5_moe_text", vocab_size = vocab_size)
    model.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
    model.mtp = torch.nn.ModuleList([_AttentionMaskMTPHead(hidden_size, vocab_size)])

    loss = helpers["compute_mtp_loss"](
        model,
        torch.randn(1, 6, hidden_size),
        torch.tensor([[0, 1, 2, 3, 4, 5]], dtype = torch.long),
        loss_fn = helpers["fast_cross_entropy_loss"],
        packed_seq_lengths = torch.tensor([3, 3], dtype = torch.int32),
    )

    seen = model.mtp[0].seen_attention_mask
    assert loss is not None
    assert seen is not None and seen.shape == (1, 1, 6, 6)
    # No attention across the packed document boundary.
    assert bool((seen[0, 0, 3:, :3] < 0).all())
    assert bool((seen[0, 0, :3, 3:] < 0).all())


def test_set_forward_argument_overrides_positional_values():
    helpers = _load_mtp_loss_helpers()

    def forward(
        input_ids = None,
        labels = None,
        return_dict = True,
    ):
        return input_ids, labels, return_dict

    args, kwargs = helpers["set_forward_argument"](forward, (1, 2, False), {}, "return_dict", True)
    assert args == (1, 2, True)
    assert kwargs == {}

    args, kwargs = helpers["set_forward_argument"](forward, (1,), {}, "return_dict", True)
    assert args == (1,)
    assert kwargs == {"return_dict": True}


def test_patch_mtp_loss_preserves_tuple_when_return_dict_defaults_false():
    helpers = _load_mtp_loss_helpers()
    hidden_size = 4
    vocab_size = 7

    class _ConfigDefaultTupleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                model_type = "qwen3_5_moe_text",
                vocab_size = vocab_size,
                output_hidden_states = False,
                use_return_dict = False,
            )
            self.vocab_size = vocab_size
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias = False)
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.mtp = torch.nn.ModuleList([_MTPHead(hidden_size, vocab_size)])

        def forward(
            self,
            input_ids = None,
            labels = None,
            return_dict = None,
            output_hidden_states = False,
            **_kwargs,
        ):
            hidden_states = self.model.embed_tokens(input_ids)
            loss = hidden_states.sum() * 0.0 + 1.0
            hidden_states = (hidden_states,) if output_hidden_states else None
            if return_dict is None:
                return_dict = self.config.use_return_dict
            if return_dict:
                return _ModelOutput(loss, hidden_states)
            return (loss, hidden_states)

    model = helpers["patch_mtp_loss"](
        _ConfigDefaultTupleModel(), helpers["fast_cross_entropy_loss"]
    )
    outputs = model(
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype = torch.long),
        labels = torch.tensor([[0, 1, 2, 3]], dtype = torch.long),
    )

    # `return_dict` was never passed, so the caller still gets the tuple layout it
    # would have without MTP: loss only, no internally requested hidden states.
    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert outputs[0].item() > 1.0
