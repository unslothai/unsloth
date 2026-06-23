import ast
import inspect
import types
from pathlib import Path

import torch


def _load_mtp_loss_helpers():
    source = Path(__file__).parents[1] / "unsloth" / "models" / "mtp.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    names = {
        "call_mtp_module",
        "bind_forward_arguments",
        "compute_mtp_loss",
        "filter_mtp_kwargs",
        "get_forward_argument",
        "get_mtp_modules",
        "get_output_loss_and_hidden_states",
        "get_tuple_hidden_states",
        "iter_mtp_outputs",
        "make_mtp_shift_labels",
        "mask_mtp_packed_sequence_boundaries",
        "patch_mtp_loss",
        "set_output_loss_and_hidden_states",
        "should_use_mtp_loss",
        "unwrap_mtp_output",
    }
    helpers = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body = helpers, type_ignores = [])
    ast.fix_missing_locations(module)

    def fast_cross_entropy_loss(logits, labels, **_kwargs):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index = -100,
        )

    namespace = {
        "inspect": inspect,
        "types": types,
        "torch": torch,
        "fast_cross_entropy_loss": fast_cross_entropy_loss,
    }
    exec(compile(module, str(source), "exec"), namespace)
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
