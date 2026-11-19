# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Host lengths ride the packed batch device move (meta stand-in raises on any read); loss-only eval skips empty_cache."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
accelerate_data_loader = pytest.importorskip("accelerate.data_loader")

try:
    import unsloth  # noqa: F401
    from unsloth.utils import packing
    from unsloth.models import rl as rl_module
except ImportError:
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)


def _padding_free_collator():
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(pad_token_id = 0, padding_free = True)


def test_packed_lengths_reach_device_without_a_device_read():
    collator = _padding_free_collator()
    trainer = type("T", (), {})()
    trainer.data_collator = collator
    trainer.args = type("A", (), {"remove_unused_columns": True})()
    packing.enable_padding_free_metadata(None, trainer)
    examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}, {"input_ids": [6, 7, 8, 9]}]
    batch = collator(examples)
    assert batch["packed_seq_lengths"].device.type == "cpu"

    moved = accelerate_data_loader.send_to_device(batch, torch.device("meta"))
    lengths = moved["packed_seq_lengths"]
    assert lengths.device.type == "meta"
    assert getattr(lengths, "_unsloth_lengths_tuple", None) == (3, 2, 4)

    packing._PACKED_INFO_CACHE.clear()
    packing._PACKED_HOST_LENGTHS_CACHE.clear()
    # A blocking device read would raise on meta ("Cannot copy out of meta tensor").
    _, cu_seqlens, max_seqlen = packing.get_packed_info_from_kwargs(
        {"packed_seq_lengths": lengths}, torch.device("meta")
    )
    assert max_seqlen == 4
    assert cu_seqlens.shape == (4,)
    assert packing._host_segment_lengths(lengths) == (3, 2, 4)
    packing._PACKED_INFO_CACHE.clear()
    packing._PACKED_HOST_LENGTHS_CACHE.clear()


def test_carry_leaves_other_moves_alone():
    # Installed here, not inherited from the test above: xdist can run this one first or alone.
    packing._install_host_lengths_carry()
    src = {"packed_seq_lengths": torch.tensor([2, 3], dtype = torch.int32)}
    same = dict(src)
    packing._carry_host_lengths(src, same)
    assert not hasattr(same["packed_seq_lengths"], "_unsloth_lengths_tuple")
    plain = accelerate_data_loader.send_to_device(
        {"input_ids": torch.ones(2, 3, dtype = torch.long)}, torch.device("meta")
    )
    assert not hasattr(plain["input_ids"], "_unsloth_lengths_tuple")
    assert accelerate_data_loader.send_to_device._unsloth_host_lengths_carry


class _TinyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids = None,
        labels = None,
        **kwargs,
    ):
        loss = (input_ids.float().mean() * self.weight - labels.float().mean()) ** 2
        return {"loss": loss, "logits": input_ids.float().unsqueeze(-1) * self.weight}


class _Rows(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i, i + 1]), "labels": torch.tensor([i, i])}


def _eval_flushes(
    trainer_class,
    monkeypatch,
    tmp_path,
    compute_metrics = None,
):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    args = transformers.TrainingArguments(
        output_dir = str(tmp_path),
        per_device_eval_batch_size = 1,
        eval_accumulation_steps = 2,
        report_to = "none",
        use_cpu = True,
    )
    trainer = trainer_class(
        model = _TinyLoss(),
        args = args,
        eval_dataset = _Rows(),
        compute_metrics = compute_metrics,
    )
    metrics = trainer.evaluate()
    assert trainer.args.eval_accumulation_steps == 2
    return len(calls), metrics["eval_loss"]


def test_loss_only_eval_skips_the_flush(monkeypatch, tmp_path):
    original = getattr(
        transformers.Trainer.evaluation_loop, "__wrapped__", transformers.Trainer.evaluation_loop
    )
    base = type("BaseTrainer", (transformers.Trainer,), {"evaluation_loop": original})
    head = type("HeadTrainer", (transformers.Trainer,), {"evaluation_loop": original})
    rl_module._patch_loss_only_eval_flush(head)
    assert head.evaluation_loop._unsloth_loss_only_no_flush

    base_flushes, base_loss = _eval_flushes(base, monkeypatch, tmp_path)
    head_flushes, head_loss = _eval_flushes(head, monkeypatch, tmp_path)
    assert base_flushes == 6
    assert head_flushes == 0
    assert head_loss == pytest.approx(base_loss, rel = 0, abs = 0)

    # Kept logits (compute_metrics) still offload and flush as before.
    metric = lambda prediction: {"n": float(len(prediction.predictions))}
    kept_flushes, _ = _eval_flushes(head, monkeypatch, tmp_path, compute_metrics = metric)
    assert kept_flushes == 6

    monkeypatch.setenv("UNSLOTH_EVAL_EMPTY_CACHE", "1")
    opt_out_flushes, _ = _eval_flushes(head, monkeypatch, tmp_path)
    assert opt_out_flushes == 6
