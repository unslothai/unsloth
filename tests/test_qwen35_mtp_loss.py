import ast
import types
from pathlib import Path

import torch


def _load_mtp_loss_helpers():
    source = Path(__file__).parents[1] / "unsloth" / "models" / "mtp.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    names = {
        "call_mtp_module",
        "compute_mtp_loss",
        "filter_mtp_kwargs",
        "get_mtp_modules",
        "make_mtp_shift_labels",
        "mask_mtp_packed_sequence_boundaries",
        "should_use_mtp_loss",
        "unwrap_mtp_output",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
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
