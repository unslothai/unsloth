"""An offloaded embedding must stay in host RAM through the model.to(device) that Trainer.train() and
Accelerator.prepare_model issue, else the first training step pulls it back onto the card
(unslothai/unsloth#7055: the 4.4 GB Gemma 4 E2B per-layer table on an 8 GB GPU). "meta" stands in for
the accelerator so this runs on CPU."""

import ast, functools, os, types
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")
NAMES = (
    "_host_offloaded_embeddings",
    "_wrap_move_keeps_host_embeddings",
    "_patch_trainer_move_keeps_host_embeddings",
)


def _load():
    src = open(VISION, encoding = "utf-8").read()
    ns = {"torch": torch, "functools": functools}
    found = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            exec(ast.get_source_segment(src, node), ns)
            found.add(node.name)
    assert found == set(NAMES), set(NAMES) - found
    return ns


NS = _load()


def _model(offloaded = True):
    model = nn.Module()
    model.embed = nn.Embedding(16, 4)
    model.inner = nn.Module()
    model.inner.per_layer = nn.Embedding(16, 4)
    model.proj = nn.Linear(4, 4)
    if offloaded:
        model.inner.per_layer._unsloth_offload_hooks_installed = True
    return model


def _owner_from_trainer():
    from transformers.trainer import Trainer
    class Owner:
        _move_model_to_device = Trainer._move_model_to_device

    return Owner


def _trainer_self():
    from transformers.training_args import ParallelMode
    return types.SimpleNamespace(
        args = types.SimpleNamespace(parallel_mode = ParallelMode.NOT_DISTRIBUTED)
    )


def test_trainer_move_keeps_offloaded_table_on_host():
    Owner = _owner_from_trainer()
    NS["_wrap_move_keeps_host_embeddings"](Owner, "_move_model_to_device")
    model = _model()
    Owner._move_model_to_device(_trainer_self(), model, torch.device("meta"))
    assert model.proj.weight.device.type == "meta"
    assert model.embed.weight.device.type == "meta"
    assert model.inner.per_layer.weight.device.type == "cpu"
    # Only that call is affected: the module keeps its own _apply afterwards.
    assert "_apply" not in model.inner.per_layer.__dict__
    model.to("meta")
    assert model.inner.per_layer.weight.device.type == "meta"


def test_unwrapped_trainer_move_pulls_the_table_back():
    # Negative control: what Trainer does without the wrapper.
    Owner = _owner_from_trainer()
    model = _model()
    Owner._move_model_to_device(_trainer_self(), model, torch.device("meta"))
    assert model.inner.per_layer.weight.device.type == "meta"


def test_unmarked_or_restored_embedding_moves():
    class Owner:
        def prepare_model(
            self,
            model,
            device_placement = None,
        ):
            return model.to("meta")

    NS["_wrap_move_keeps_host_embeddings"](Owner, "prepare_model")
    model = _model(offloaded = False)
    Owner().prepare_model(model)
    assert model.inner.per_layer.weight.device.type == "meta"
    # Marked but back on an accelerator (restore_input_embedding) is not pinned either.
    model = _model()
    model.inner.per_layer.to("meta")
    assert NS["_host_offloaded_embeddings"](model) == []


def test_wrap_is_idempotent_and_restores_on_error():
    class Owner:
        def prepare_model(self, model):
            raise RuntimeError("boom")

    NS["_wrap_move_keeps_host_embeddings"](Owner, "prepare_model")
    first = Owner.prepare_model
    NS["_wrap_move_keeps_host_embeddings"](Owner, "prepare_model")
    assert Owner.prepare_model is first
    model = _model()
    try:
        Owner().prepare_model(model)
    except RuntimeError:
        pass
    assert "_apply" not in model.inner.per_layer.__dict__


def test_patch_targets_exist():
    # Drift guard: the methods that issue the move must still exist under these names.
    from transformers.trainer import Trainer
    from accelerate import Accelerator

    assert callable(getattr(Trainer, "_move_model_to_device", None))
    assert callable(getattr(Accelerator, "prepare_model", None))
