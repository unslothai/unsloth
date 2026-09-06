import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RL_SOURCE = ROOT / "unsloth" / "models" / "rl.py"


def _load_restore_helper():
    """Load the pure helper without importing Unsloth's optional RL stack."""
    source = RL_SOURCE.read_text(encoding = "utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restore_valid_grpo_train_batch_size"
    )
    namespace = {}
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body = [function], type_ignores = []), str(RL_SOURCE), "exec"), namespace)
    return namespace[function.name]


restore_batch_size = _load_restore_helper()


def test_restores_batch_when_gradient_accumulation_makes_it_divisible():
    args = SimpleNamespace(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_generations = 2,
        generation_batch_size = None,
    )

    changed = restore_batch_size(args, requested_batch_size = 1)

    assert changed is True
    assert args.per_device_train_batch_size == 1


def test_restores_batch_when_explicit_generation_batch_is_divisible():
    args = SimpleNamespace(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        num_generations = 4,
        generation_batch_size = 8,
    )

    changed = restore_batch_size(args, requested_batch_size = 1)

    assert changed is True
    assert args.per_device_train_batch_size == 1


def test_keeps_compatibility_rewrite_for_invalid_effective_batch():
    args = SimpleNamespace(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        num_generations = 4,
        generation_batch_size = None,
    )

    changed = restore_batch_size(args, requested_batch_size = 1)

    assert changed is False
    assert args.per_device_train_batch_size == 4


def test_does_not_reverse_an_unrelated_batch_size_change():
    args = SimpleNamespace(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        num_generations = 2,
        generation_batch_size = None,
    )

    changed = restore_batch_size(args, requested_batch_size = 1)

    assert changed is False
    assert args.per_device_train_batch_size == 8


def test_generated_trainer_calls_restore_after_version_specific_adjustments():
    source = RL_SOURCE.read_text(encoding = "utf-8")
    template_start = source.index("RLTrainer_replacement = '''")
    template_end = source.index("'''", template_start + len("RLTrainer_replacement = '''"))
    template = source[template_start:template_end]

    requested = template.index("_unsloth_requested_train_batch_size = getattr(")
    adjustments = template.index("{RLTrainer_extra_args}")
    restore = template.index("_restore_valid_grpo_train_batch_size(")

    assert requested < adjustments < restore
