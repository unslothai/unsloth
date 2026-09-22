# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# `embedding_learning_rate` routes optimizer creation through
# `_create_unsloth_optimizer`, which read the decay from `optimizer_kwargs`.
# transformers keeps the decay on the param groups and only puts it in
# optimizer_kwargs for schedule-free and stable_adamw, so the `.get(..., 0.0)`
# default always won and the whole run trained with no weight decay, silently,
# with the value the user set still on the config.

from types import MethodType, SimpleNamespace

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer


EMBEDDING = "model.embed_tokens.modules_to_save.default.weight"


def _model(torch, nn):
    model = nn.Module()
    model.proj = nn.Linear(4, 4)  # .weight decays, .bias does not
    model.norm = nn.LayerNorm(4)  # neither weight nor bias decays
    embed = nn.Module()
    embed.modules_to_save = nn.ModuleDict({"default": nn.Linear(4, 4, bias = False)})
    inner = nn.Module()
    inner.embed_tokens = embed
    model.model = inner
    return model


def _trainer(torch, weight_decay):
    from transformers import Trainer
    from trl import SFTConfig

    args = SFTConfig(
        output_dir = "/tmp/unsloth-weight-decay-test",
        weight_decay = weight_decay,
        learning_rate = 2e-4,
        optim = "adamw_torch",
        report_to = [],
    )
    args.embedding_learning_rate = 5e-5
    trainer = SimpleNamespace(args = args, model = None, optimizer = None)
    trainer.get_decay_parameter_names = MethodType(Trainer.get_decay_parameter_names, trainer)
    return trainer


def _groups(torch, nn, weight_decay):
    model = _model(torch, nn)
    trainer = _trainer(torch, weight_decay)
    trainer.model = model
    optimizer = UnslothTrainer.create_optimizer(trainer, model)
    named = {id(p): n for n, p in model.named_parameters()}
    return [
        {
            "names": sorted(named[id(p)] for p in group["params"]),
            "weight_decay": group["weight_decay"],
            "lr": group["lr"],
        }
        for group in optimizer.param_groups
    ]


def test_weight_decay_from_the_config_reaches_the_param_groups():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    decayed = {name for g in groups if g["weight_decay"] == 0.1 for name in g["names"]}

    assert "proj.weight" in decayed, f"weight_decay=0.1 never reached the optimizer: {groups}"
    assert EMBEDDING in decayed, "the embedding group must carry the decay too"


def test_biases_and_norms_stay_out_of_the_decay():
    # Trainer.get_decay_parameter_names excludes them on the path this replaces.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    undecayed = {name for g in groups if g["weight_decay"] == 0.0 for name in g["names"]}

    assert {"proj.bias", "norm.weight", "norm.bias"} <= undecayed, groups


def test_the_embedding_group_keeps_its_own_learning_rate():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    for group in groups:
        if EMBEDDING in group["names"]:
            assert group["lr"] == 5e-5, groups
        elif group["names"]:
            assert group["lr"] == 2e-4, groups


def test_zero_weight_decay_is_still_zero():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.0)
    assert all(group["weight_decay"] == 0.0 for group in groups), groups


def _lora_shaped(torch, nn):
    """A LoRA run trains no bias and no norm, so both no-decay groups come back empty."""
    model = _model(torch, nn)
    model.proj.bias.requires_grad_(False)
    model.norm.weight.requires_grad_(False)
    model.norm.bias.requires_grad_(False)
    return model


def _optimizer(
    torch,
    model,
    weight_decay,
    optim = "adamw_torch",
):
    trainer = _trainer(torch, weight_decay)
    trainer.args.optim = optim
    trainer.model = model
    return UnslothTrainer.create_optimizer(trainer, model)


def test_no_empty_param_groups():
    # An empty group is not free. AdafactorSchedule reads group["params"][0] with no
    # guard, and torch's load_state_dict insists the checkpoint have the same number
    # of groups, so an empty one is a crash and a broken resume for nothing.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    for model in (_model(torch, nn), _lora_shaped(torch, nn)):
        optimizer = _optimizer(torch, model, 0.1)
        assert all(group["params"] for group in optimizer.param_groups), optimizer.param_groups


def test_adafactor_schedule_survives_the_split():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from transformers.optimization import AdafactorSchedule

    model = _lora_shaped(torch, nn)
    optimizer = _optimizer(torch, model, 0.1, optim = "adafactor")
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    optimizer.step()

    assert AdafactorSchedule(optimizer).get_lr()  # IndexError if any group is empty


def test_a_lora_run_started_before_this_can_still_resume():
    # The old two-group build is what every checkpoint out there was written with.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _lora_shaped(torch, nn)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    old = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in trainable if name != EMBEDDING],
                "weight_decay": 0.0,
                "lr": 2e-4,
            },
            {
                "params": [p for name, p in trainable if name == EMBEDDING],
                "weight_decay": 0.0,
                "lr": 5e-5,
            },
        ],
        lr = 2e-4,
    )
    for _, param in trainable:
        param.grad = torch.randn_like(param)
    old.step()

    new = _optimizer(torch, model, 0.1)
    new.load_state_dict(old.state_dict())  # ValueError if the group count moved


def _decay_names(torch, model):
    from transformers import Trainer
    return Trainer.get_decay_parameter_names(None, model)


def _legacy_optimizer(
    torch,
    model,
    lr = 2e-4,
    embedding_lr = 5e-5,
):
    """The two groups every checkpoint written before the decay split carries."""
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {
                "params": [p for name, p in trainable if name != EMBEDDING],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [p for name, p in trainable if name == EMBEDDING],
                "weight_decay": 0.0,
                "lr": embedding_lr,
            },
        ],
        lr = lr,
    )


def test_a_full_finetune_checkpoint_from_before_the_split_still_resumes():
    # Trainable norms give the new layout three groups where the checkpoint has two.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _model(torch, nn)
    old = _legacy_optimizer(torch, model)
    for index, (_, param) in enumerate(model.named_parameters()):
        param.grad = torch.full_like(param, float(index + 1))
    old.step()
    saved = old.state_dict()
    flat = [p for group in old.param_groups for p in group["params"]]
    named = {id(p): n for n, p in model.named_parameters()}
    want = {named[id(p)]: float(saved["state"][i]["exp_avg"].sum()) for i, p in enumerate(flat)}

    new = _optimizer(torch, model, 0.1)
    assert len(new.param_groups) != len(saved["param_groups"]), "no migration exercised"
    new.load_state_dict(saved)

    # Not just "it loaded": every parameter must get ITS OWN moments back.
    state = new.state_dict()["state"]
    flat = [p for group in new.param_groups for p in group["params"]]
    for index, param in enumerate(flat):
        assert float(state[index]["exp_avg"].sum()) == pytest.approx(want[named[id(param)]])
    # and the decay the run was configured with, not the 0.0 the checkpoint carries.
    assert any(group["weight_decay"] == 0.1 for group in new.param_groups), new.param_groups


def test_a_checkpoint_that_is_not_the_old_shape_is_refused_not_guessed():
    # Fails closed: pairing a parameter with another parameter's moments would corrupt
    # the run silently, which is worse than the error this keeps.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _model(torch, nn)
    new = _optimizer(torch, model, 0.1)
    bogus = {
        "state": {},
        "param_groups": [{"params": [0], "lr": 2e-4, "weight_decay": 0.0}],
    }
    with pytest.raises(ValueError):
        new.load_state_dict(bogus)


def test_the_scheduler_survives_the_same_resume():
    # Fixing only the optimizer moves the failure one line later, into base_lrs.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from torch.optim.lr_scheduler import LambdaLR
    from unsloth.trainer import _install_legacy_scheduler_resume

    model = _model(torch, nn)
    old_scheduler = LambdaLR(_legacy_optimizer(torch, model), lambda step: 1.0)
    old_scheduler.step()
    saved = old_scheduler.state_dict()

    new_optimizer = _optimizer(torch, model, 0.1)
    new_optimizer.load_state_dict(_legacy_optimizer(torch, model).state_dict())
    new_scheduler = _install_legacy_scheduler_resume(
        LambdaLR(new_optimizer, lambda step: 1.0), new_optimizer
    )
    new_scheduler.load_state_dict(saved)
    new_scheduler.step()  # ValueError from zip(strict=True) if base_lrs was not expanded
    assert len(new_scheduler.base_lrs) == len(new_optimizer.param_groups)


def test_a_checkpoint_with_the_same_group_count_but_a_different_shape_migrates():
    # No trainable embedding: legacy is [all non-embeddings, empty], new is
    # [decayed, non-decayed]. Two groups either way, so a count check misses it.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from unsloth.trainer import _create_unsloth_optimizer

    model = nn.Module()
    model.proj = nn.Linear(4, 4)  # .weight decays, .bias does not
    trainable = [p for p in model.parameters() if p.requires_grad]
    old = torch.optim.AdamW(
        [
            {"params": trainable, "lr": 2e-4, "weight_decay": 0.0},
            {"params": [], "lr": 5e-5, "weight_decay": 0.0},
        ],
        lr = 2e-4,
    )
    for index, param in enumerate(model.parameters()):
        param.grad = torch.full_like(param, float(index + 1))
    old.step()
    saved = old.state_dict()

    new = _create_unsloth_optimizer(
        model,
        torch.optim.AdamW,
        {"lr": 2e-4},
        5e-5,
        weight_decay = 0.1,
        decay_parameter_names = _decay_names(torch, model),
    )
    assert len(new.param_groups) == len(saved["param_groups"]), "count check would have caught it"
    assert [len(g["params"]) for g in new.param_groups] != [
        len(g["params"]) for g in saved["param_groups"]
    ]
    new.load_state_dict(saved)
    assert sum(len(g["params"]) for g in new.param_groups) == len(trainable)


def test_the_plateau_scheduler_min_lrs_are_remapped_too():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from unsloth.trainer import _install_legacy_scheduler_resume

    model = _model(torch, nn)
    old = ReduceLROnPlateau(_legacy_optimizer(torch, model), min_lr = [1e-7, 2e-7])
    saved = old.state_dict()

    new_optimizer = _optimizer(torch, model, 0.1)
    # transformers loads the optimizer before the scheduler; that is what marks the
    # checkpoint legacy, so the sequence matters and is reproduced here.
    new_optimizer.load_state_dict(_legacy_optimizer(torch, model).state_dict())
    new_scheduler = _install_legacy_scheduler_resume(
        ReduceLROnPlateau(new_optimizer, min_lr = 1e-7), new_optimizer
    )
    new_scheduler.load_state_dict(saved)
    assert len(new_scheduler.min_lrs) == len(new_optimizer.param_groups)
    for _ in range(14):
        new_scheduler.step(1.0)  # RuntimeError if min_lrs is still the legacy length


def test_resuming_a_lora_checkpoint_keeps_the_corrected_decay():
    # Old and new layouts coincide here, so nothing needs reshaping, but torch takes the
    # hyperparameters from the saved dict: without the migration the run reloads the 0.0
    # this change exists to correct and trains undecayed again, silently.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _lora_shaped(torch, nn)
    old = _legacy_optimizer(torch, model)
    for _, param in [(n, p) for n, p in model.named_parameters() if p.requires_grad]:
        param.grad = torch.randn_like(param)
    old.step()
    saved = old.state_dict()

    new = _optimizer(torch, model, 0.1)
    assert [len(g["params"]) for g in new.param_groups] == [
        len(g["params"]) for g in saved["param_groups"]
    ], "shapes must coincide or this is testing the reshaping path instead"
    new.load_state_dict(saved)
    assert all(group["weight_decay"] == 0.1 for group in new.param_groups), new.param_groups


def test_scheduler_state_of_equal_length_is_remapped_by_role_not_position():
    # No trainable embedding: both current groups are non-embeddings, so the saved
    # [ordinary, embedding] pair is the same length but means something else.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from torch.optim.lr_scheduler import LambdaLR
    from unsloth.trainer import _create_unsloth_optimizer, _install_legacy_scheduler_resume

    model = nn.Module()
    model.proj = nn.Linear(4, 4)  # .weight decays, .bias does not
    trainable = [p for p in model.parameters() if p.requires_grad]
    old_optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": 2e-4}, {"params": [], "lr": 5e-5}], lr = 2e-4
    )
    saved = LambdaLR(old_optimizer, lambda step: 1.0).state_dict()
    assert saved["base_lrs"] == [2e-4, 5e-5]

    new_optimizer = _create_unsloth_optimizer(
        model,
        torch.optim.AdamW,
        {"lr": 2e-4},
        5e-5,
        weight_decay = 0.1,
        decay_parameter_names = _decay_names(torch, model),
    )
    assert new_optimizer._unsloth_group_roles == ["non_embeddings", "non_embeddings"]
    new_optimizer.load_state_dict(old_optimizer.state_dict())
    scheduler = _install_legacy_scheduler_resume(
        LambdaLR(new_optimizer, lambda step: 1.0), new_optimizer
    )
    scheduler.load_state_dict(saved)
    assert scheduler.base_lrs == [2e-4, 2e-4], scheduler.base_lrs


def test_the_scheduler_hook_stays_out_of_the_checkpoint():
    # A scheduler's state_dict is its __dict__ minus the optimizer, so hooking by
    # instance attribute puts an unpicklable closure into every checkpoint and the
    # FIRST save fails, resume or no resume.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    import io
    from torch.optim.lr_scheduler import LambdaLR
    from unsloth.trainer import _install_legacy_scheduler_resume

    model = _model(torch, nn)
    optimizer = _optimizer(torch, model, 0.1)
    scheduler = _install_legacy_scheduler_resume(LambdaLR(optimizer, lambda step: 1.0), optimizer)

    state = scheduler.state_dict()
    assert not [key for key, value in state.items() if callable(value)], state.keys()
    torch.save(state, io.BytesIO())  # PicklingError if the hook leaked into the state
    torch.save(optimizer.state_dict(), io.BytesIO())
    # and it is still an ordinary scheduler to everyone else
    assert isinstance(scheduler, LambdaLR)


def test_the_scheduler_hook_survives_an_optimizer_wrapper():
    # accelerator.prepare swaps in an AcceleratedOptimizer before create_scheduler runs.
    # It defines no __getattr__, so asking it directly finds no roles and the hook would
    # quietly do nothing on the ordinary training path. That property of the real class is
    # asserted here too, so this stops passing if accelerate ever changes it.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from torch.optim.lr_scheduler import LambdaLR
    from unsloth.trainer import _install_legacy_scheduler_resume

    accelerate_optimizer = pytest.importorskip("accelerate.optimizer")
    real = accelerate_optimizer.AcceleratedOptimizer
    assert not any(
        "__getattr__" in cls.__dict__ for cls in real.__mro__
    ), "AcceleratedOptimizer now delegates attributes; the unwrap may be unnecessary"

    class Wrapper(torch.optim.Optimizer):
        """The same shape as the real one: holds the optimizer, forwards nothing else."""

        def __init__(self, optimizer):
            self.optimizer = optimizer

        @property
        def param_groups(self):
            return self.optimizer.param_groups

    model = _model(torch, nn)
    optimizer = _optimizer(torch, model, 0.1)
    wrapped = Wrapper(optimizer)
    assert "_unsloth_group_roles" not in wrapped.__dict__

    scheduler = _install_legacy_scheduler_resume(LambdaLR(wrapped, lambda step: 1.0), wrapped)
    assert getattr(type(scheduler), "_unsloth_legacy_resume", False), "hook silently skipped"

    optimizer.load_state_dict(_legacy_optimizer(torch, model).state_dict())
    saved = LambdaLR(_legacy_optimizer(torch, model), lambda step: 1.0).state_dict()
    scheduler.load_state_dict(saved)
    assert len(scheduler.base_lrs) == len(optimizer.param_groups)


def test_migration_keeps_optimizer_specific_group_state():
    # Schedule-free and friends keep algorithm progress in the param group itself;
    # rebuilding from the fresh group would reset it while keeping the per-parameter
    # state, leaving the resumed optimizer inconsistent with itself.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _lora_shaped(torch, nn)
    old = _legacy_optimizer(torch, model)
    for group in old.param_groups:
        group["k"] = 17
        group["weight_sum"] = 3.5
    for _, param in [(n, p) for n, p in model.named_parameters() if p.requires_grad]:
        param.grad = torch.randn_like(param)
    old.step()

    new = _optimizer(torch, model, 0.1)
    new.load_state_dict(old.state_dict())
    assert all(group["k"] == 17 for group in new.param_groups), new.param_groups
    assert all(group["weight_sum"] == 3.5 for group in new.param_groups), new.param_groups
    # but the decay is still the corrected one, not the checkpoint's 0.0
    assert all(group["weight_decay"] == 0.1 for group in new.param_groups), new.param_groups


def test_a_current_two_group_checkpoint_is_not_mistaken_for_a_legacy_one():
    # With no trainable embedding both current groups are non_embeddings, so a checkpoint
    # this code wrote also has two scheduler entries. Remapping it by role would copy the
    # first group's value over the second, and distinct per-group min_lr floors would come
    # back collapsed. Length cannot tell the two apart; only the optimizer load can.
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from unsloth.trainer import _create_unsloth_optimizer, _install_legacy_scheduler_resume

    model = nn.Module()
    model.proj = nn.Linear(4, 4)  # .weight decays, .bias does not

    def build():
        optimizer = _create_unsloth_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 2e-4},
            5e-5,
            weight_decay = 0.1,
            decay_parameter_names = _decay_names(torch, model),
        )
        assert optimizer._unsloth_group_roles == ["non_embeddings", "non_embeddings"]
        return optimizer

    written_by_this_code = build()
    saved = _install_legacy_scheduler_resume(
        ReduceLROnPlateau(written_by_this_code, min_lr = [1e-7, 2e-7]), written_by_this_code
    ).state_dict()
    assert saved["min_lrs"] == [1e-7, 2e-7]

    resumed = build()
    resumed.load_state_dict(written_by_this_code.state_dict())  # not a legacy shape
    scheduler = _install_legacy_scheduler_resume(
        ReduceLROnPlateau(resumed, min_lr = [1e-7, 2e-7]), resumed
    )
    scheduler.load_state_dict(saved)
    assert scheduler.min_lrs == [1e-7, 2e-7], scheduler.min_lrs
