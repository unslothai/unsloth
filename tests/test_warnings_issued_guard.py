"""transformers 5.1 dropped `warnings_issued`; eight trl trainers still write it.

`PreTrainedModel.__init__` set `self.warnings_issued = {}` up to and including
transformers 5.0.0. From 5.1.0 the name does not appear in modeling_utils.py at
all. trl did not follow, and does this unconditionally at the top of
`__init__` in grpo, dpo, online_dpo, kto, orpo, cpo, rloo and experimental bco:

    model.warnings_issued["estimate_tokens"] = True

With the attribute gone, nn.Module.__getattr__ raises before the trainer is
built:

    AttributeError: 'Qwen2ForCausalLM' object has no attribute 'warnings_issued'

Measured on Colab at transformers 5.13.1 + trl 0.25.1 running NeMo-Gym-Sudoku.

unsloth already guards it, but only inside the source it GENERATES for the
compiled trainer (`models/rl.py`), so the guard exists exactly when that
generation succeeds. When it fails, unsloth falls back to trl's own class and
the write is unguarded again, which is the gap the `trainer.py` wrapper closes.

Not UNSLOTH_COMPILE_DISABLE=1, despite the obvious guess: measured with a fresh
cache, that mode still writes `unsloth_compiled_cache/UnslothGRPOTrainer.py`
with the guard in it, and `trl.GRPOTrainer` is still the generated class.

trainer.py is loaded by AST here, not imported: importing `unsloth.trainer`
drags in GPU init, and none of this logic needs a GPU.
"""

import ast
import dataclasses
import inspect
from functools import wraps
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "unsloth" / "trainer.py"
SRC = SRC_PATH.read_text(encoding = "utf-8")


def _load(*names):
    """Exec the named top-level functions from trainer.py in a bare namespace."""
    tree = ast.parse(SRC)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    assert not missing, f"not found in trainer.py: {sorted(missing)}"
    ns = {
        "torch": torch,
        "inspect": inspect,
        "dataclasses": dataclasses,
        "wraps": wraps,
    }
    try:
        import trl
        from unsloth_zoo.utils import Version
        ns["trl"], ns["Version"] = trl, Version
    except Exception:  # pragma: no cover - only used by the integration tests
        pass
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(SRC_PATH), "exec"), ns)
    return ns


GUARD = _load("_ensure_warnings_issued")["_ensure_warnings_issued"]


class _Bare(torch.nn.Module):
    """A module with no `warnings_issued`, exactly like transformers >= 5.1."""


# ---- the behaviour ---------------------------------------------------------


def test_a_module_without_the_attribute_gets_a_dict():
    m = _Bare()
    with pytest.raises(AttributeError):
        m.warnings_issued
    GUARD(m)
    assert m.warnings_issued == {}


def test_the_dict_is_writable_the_way_trl_writes_it():
    m = _Bare()
    GUARD(m)
    m.warnings_issued["estimate_tokens"] = True
    assert m.warnings_issued == {"estimate_tokens": True}


def test_each_model_gets_its_own_dict():
    """A shared class-level dict would suppress the warning globally, which is
    a behaviour change rather than a compatibility fix."""
    a, b = _Bare(), _Bare()
    GUARD(a)
    GUARD(b)
    a.warnings_issued["estimate_tokens"] = True
    assert b.warnings_issued == {}


def test_an_existing_dict_is_left_exactly_as_it_was():
    """This is the whole backwards-compatibility claim: on transformers 4.x and
    5.0, where the attribute is already a dict, the guard must be a no-op --
    same object, same contents, nothing rebound."""
    m = _Bare()
    original = {"estimate_tokens": True, "other": 1}
    m.warnings_issued = original
    GUARD(m)
    assert m.warnings_issued is original
    assert m.warnings_issued == {"estimate_tokens": True, "other": 1}


def test_a_non_dict_mapping_is_preserved_not_discarded():
    m = _Bare()
    m.warnings_issued = [("estimate_tokens", True)]
    GUARD(m)
    assert m.warnings_issued == {"estimate_tokens": True}


def test_an_uncoercible_value_becomes_an_empty_dict():
    """Better a usable empty dict than a TypeError from dict(3)."""
    m = _Bare()
    m.warnings_issued = 3
    GUARD(m)
    assert m.warnings_issued == {}


def test_running_twice_is_idempotent():
    m = _Bare()
    GUARD(m)
    m.warnings_issued["estimate_tokens"] = True
    GUARD(m)
    assert m.warnings_issued == {"estimate_tokens": True}


# ---- what it deliberately does not touch -----------------------------------


@pytest.mark.parametrize("value", [None, "Qwen/Qwen2.5-1.5B", 7, object()])
def test_non_modules_are_left_alone_without_raising(value):
    """trl accepts a repo id string and builds the model itself. Attaching an
    attribute to whatever was passed is not this function's job, and it must
    never be the thing that raises."""
    GUARD(value)
    assert not hasattr(value, "warnings_issued")


def test_a_model_that_refuses_the_assignment_does_not_raise():
    """Whatever that model's problem is, trl should report it, not us."""

    class _Locked(torch.nn.Module):
        def __setattr__(self, name, value):
            if name == "warnings_issued":
                raise RuntimeError("no")
            super().__setattr__(name, value)

    GUARD(_Locked())  # must not propagate


# ---- through the real wrapper, against a trl-shaped trainer ----------------


@dataclasses.dataclass
class _FakeConfig:
    learning_rate: float = 1e-4


class _FakeTrainer:
    """The first three lines of trl's GRPOTrainer.__init__, in effect."""

    def __init__(
        self,
        model = None,
        args = None,
        **kwargs,
    ):
        model.warnings_issued["estimate_tokens"] = True
        self.model = model
        self.args = args


def _wrapped():
    ns = _load(
        "_ensure_warnings_issued",
        "_resolve_trainer_params",
        "_route_unknown_trainer_kwargs",
        "_backwards_compatible_trainer",
    )
    if "trl" not in ns:
        pytest.skip("trl not installed")
    for name in (
        "classify_config_kwarg",
        "rename_source",
        "removal_source",
        "rename_value_is_unset",
    ):
        ns[name] = getattr(_rl_config_compat(), name)

    class T(_FakeTrainer):
        pass

    T.__init__ = ns["_backwards_compatible_trainer"](T, _FakeConfig)
    return T


def test_the_unwrapped_trainer_reproduces_the_reported_failure():
    """Without this, the tests below could pass against a bug that no longer
    exists and nobody would notice."""
    with pytest.raises(AttributeError, match = "warnings_issued"):
        _FakeTrainer(_Bare())


def test_the_wrapper_fixes_it_for_a_positional_model():
    t = _wrapped()(_Bare())
    assert t.model.warnings_issued == {"estimate_tokens": True}


def test_the_wrapper_fixes_it_for_a_keyword_model():
    t = _wrapped()(model = _Bare())
    assert t.model.warnings_issued == {"estimate_tokens": True}


def test_the_wrapper_fixes_it_on_the_args_rebuilding_path():
    """`new_init` has a second branch that rebuilds the config when `args` is
    passed. The guard must be reached from both, not just the short one."""
    t = _wrapped()(_Bare(), args = _FakeConfig())
    assert t.model.warnings_issued == {"estimate_tokens": True}


def test_no_model_at_all_still_reaches_the_wrapped_init():
    """The guard must not swallow, or pre-empt, trl's own argument errors."""
    with pytest.raises(AttributeError):
        _wrapped()()


# ---- the source, so the fix cannot be half-applied -------------------------


def _new_init_body():
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "new_init"
            and any(
                isinstance(p, ast.FunctionDef) and p.name == "_backwards_compatible_trainer"
                for p in ast.walk(tree)
            )
        ):
            src = ast.get_source_segment(SRC, node)
            if src and "original_init(self" in src:
                return src
    raise AssertionError("new_init not found")


def test_the_guard_runs_before_the_wrapped_init():
    """After it, the AttributeError has already been raised."""
    body = _new_init_body()
    assert body.index("_ensure_warnings_issued(") < body.index("original_init(self")


def test_the_guard_is_outside_the_version_branch():
    """`new_init` only rebuilds the config when `args` is in kwargs. A guard
    nested in that branch would miss every positional-args call."""
    ns = ast.parse(SRC)
    found = []
    for node in ast.walk(ns):
        if isinstance(node, ast.FunctionDef) and node.name == "new_init":
            for stmt in node.body:  # top level of new_init only
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and getattr(sub.func, "id", None) == "_ensure_warnings_issued"
                    ):
                        found.append(type(stmt).__name__)
    assert "Expr" in found, f"guard is nested, not a top-level statement: {found}"


def test_the_generated_compiled_guard_is_still_there():
    """This change adds a second guard on the eager path. It must not have been
    made by moving the compiled one, which would regress the default path."""
    rl = (ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")
    assert "warnings_issued_check" in rl
    assert "model.warnings_issued = {}" in rl


# ---- the upstream facts this rests on --------------------------------------


def test_trl_still_writes_the_attribute_unconditionally():
    """If trl ever guards it themselves, the guard becomes a no-op.

    trl 1.x already did: the main trainers dropped the write, and the three
    experimental ones that kept it wrap it in `if hasattr(model, ...)`. So this
    is a signal, not a requirement -- asserting it would fail the whole file on
    every supported trl >= 1.0."""
    trl = pytest.importorskip("trl")
    grpo = Path(trl.__file__).parent / "trainer" / "grpo_trainer.py"
    if not grpo.exists():
        pytest.skip("trl layout changed")
    text = grpo.read_text(encoding = "utf-8")
    if 'model.warnings_issued["estimate_tokens"] = True' not in text:
        pytest.skip(f"trl {trl.__version__} no longer writes warnings_issued unguarded")


def test_the_installed_transformers_tells_us_which_side_of_5_1_we_are_on():
    """Not an assertion about which version is installed -- a check that the
    two possible worlds behave as claimed. 4.x/5.0 ship the attribute and the
    guard is a no-op; 5.1+ do not and the guard is what makes trl work."""
    transformers = pytest.importorskip("transformers")
    import transformers.modeling_utils as mu

    ships_it = "self.warnings_issued = {}" in Path(mu.__file__).read_text(encoding = "utf-8")
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if version >= (5, 1):
        assert not ships_it, "5.1+ reinstated it; re-check whether this is needed"
    elif version < (5, 0) or version == (5, 0):
        assert ships_it, f"transformers {transformers.__version__} dropped it early"


def test_every_trl_trainer_that_writes_it_goes_through_the_wrapper():
    """The guard only helps trainers `_patch_trl_trainer` actually wraps, and
    that loop's rule is "XTrainer and XConfig both exist in trl.trainer" -- not
    an explicit list. Eight trainers write the attribute; measured on trl
    0.25.1 all eight satisfy the rule. A future trl that ships a writer without
    a matching Config would slip through silently.

    Only UNGUARDED writers outside trl.experimental count: trl 1.x's sdft / ssd
    / sdpo trainers already test `hasattr(model, "warnings_issued")` first and
    are not exported from `trl.trainer`, so they need no wrapper."""
    trl = pytest.importorskip("trl")
    import trl.trainer

    names = dir(trl.trainer)
    wrapped = {x[: -len("Trainer")] for x in names if x.endswith("Trainer")} & {
        x[: -len("Config")] for x in names if x.endswith("Config")
    }

    root = Path(trl.__file__).parent
    writers = set()
    for path in root.rglob("*_trainer.py"):
        if "experimental" in path.relative_to(root).parts:
            continue  # not exported from trl.trainer, so never wrapped by design
        try:
            text = path.read_text(encoding = "utf-8")
        except OSError:
            continue
        if 'model.warnings_issued["estimate_tokens"] = True' in text:
            if 'hasattr(model, "warnings_issued")' in text:
                continue  # trl guards it itself here
            writers.add(path.stem[: -len("_trainer")])

    if not writers:
        pytest.skip("no trl trainer writes the attribute any more")

    # trl file stems are snake_case; the wrapped names are CamelCase.
    normalized = {w.replace("_", "").lower() for w in wrapped}
    unwrapped = sorted(w for w in writers if w.replace("_", "") not in normalized)
    assert unwrapped == [], f"trl trainers writing warnings_issued but unwrapped: {unwrapped}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- the kwargs the wrapper exists to move -------------------------------
#
# A real trl config, not a stand-in: `new_init` branches on isinstance(TrainingArguments), so a plain dataclass takes
# the other branch and the tests would pass against the bug.
def _sft_config():
    pytest.importorskip("trl")
    # Not `trl.SFTConfig`: on Apple Silicon `import unsloth` rebinds that name to
    # the MLX training config. The defining module still holds the real one.
    from trl.trainer.sft_config import SFTConfig
    return SFTConfig


class _RecordingTrainer:
    """trl.SFTTrainer's signature, enough of it to see what arrives."""

    def __init__(
        self,
        model = None,
        args = None,
        train_dataset = None,
        processing_class = None,
    ):
        self.args = args
        self.train_dataset = train_dataset


def _rl_config_compat():
    """Load by file spec: importing via the package drags in torch and unsloth_zoo."""
    import importlib.util

    path = ROOT / "unsloth" / "models" / "rl_config_compat.py"
    spec = importlib.util.spec_from_file_location("_rl_config_compat_for_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrapped_recording(trainer_base = None):
    config_class = _sft_config()
    ns = _load(
        "_ensure_warnings_issued",
        "_resolve_trainer_params",
        "_route_unknown_trainer_kwargs",
        "_backwards_compatible_trainer",
    )
    if "trl" not in ns:
        pytest.skip("trl not installed")
    # The relative import in the routed function cannot work in a bare namespace.
    for name in (
        "classify_config_kwarg",
        "rename_source",
        "removal_source",
        "rename_value_is_unset",
    ):
        ns[name] = getattr(_rl_config_compat(), name)

    class T(trainer_base or _RecordingTrainer):
        pass

    T.__init__ = ns["_backwards_compatible_trainer"](T, config_class)
    return T, config_class


def test_config_kwargs_reach_the_config_the_caller_passed(tmp_path):
    """Settings that used to be trainer kwargs have to end up on the config.
    They were computed and then dropped whenever `args` was given."""
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])
    assert config.packing is False and config.max_length != 2048

    trainer = Trainer(
        model = _Bare(),
        args = config,
        train_dataset = "DS",
        packing = True,
        max_length = 2048,
        dataset_num_proc = 4,
    )

    assert trainer.args.packing is True
    assert trainer.args.max_length == 2048
    assert trainer.args.dataset_num_proc == 4


def test_the_callers_own_config_object_is_the_one_used(tmp_path):
    """Reinitialising re-triggers trl's mutually exclusive checks, so the values
    have to be set rather than a new config built."""
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])

    trainer = Trainer(model = _Bare(), args = config, packing = True)

    assert trainer.args is config


def test_untouched_config_values_keep_what_the_caller_set(tmp_path):
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [], max_length = 777)

    trainer = Trainer(model = _Bare(), args = config, packing = True)

    assert trainer.args.max_length == 777


def test_a_kwarg_neither_side_takes_is_reported_not_swallowed(tmp_path):
    """Not `max_seq_length` as the sentinel: the generated SFT config restores it."""
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])
    unexpected = "definitely_not_a_trainer_or_config_kwarg"
    assert unexpected not in {f.name for f in dataclasses.fields(config_class)}

    with pytest.raises(TypeError, match = unexpected):
        Trainer(model = _Bare(), args = config, **{unexpected: 2048})


def test_a_field_trl_retired_is_reported_and_dropped_not_raised(tmp_path, capsys):
    """Must warn, not raise: scripts pinned to an older TRL still pass these."""
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])
    retired = "rpo_alpha"  # DPOConfig, removed in TRL 0.29.0
    assert retired not in {f.name for f in dataclasses.fields(config_class)}

    trainer = Trainer(model = _Bare(), args = config, **{retired: 1.0})

    assert trainer.args is config
    assert not hasattr(config, retired)
    assert retired in capsys.readouterr().out


def test_a_field_trl_renamed_is_carried_across_to_the_new_name(tmp_path, capsys):
    Trainer, config_class = _wrapped_recording()
    fields = {f.name for f in dataclasses.fields(config_class)}
    if "use_liger_kernel" not in fields or "use_liger_loss" in fields:
        pytest.skip("installed trl does not show this rename on SFTConfig")
    config = config_class(output_dir = str(tmp_path), report_to = [])

    trainer = Trainer(model = _Bare(), args = config, use_liger_loss = True)

    assert trainer.args.use_liger_kernel is True
    assert "use_liger_kernel" in capsys.readouterr().out


def test_the_explicitly_supplied_new_name_beats_the_renamed_old_one(tmp_path, capsys):
    """Migrating over an explicit new name would silently train with the opposite
    setting. Covers both branches: the caller's config, and the rebuilt one."""
    Trainer, config_class = _wrapped_recording()
    fields = {f.name for f in dataclasses.fields(config_class)}
    if "use_liger_kernel" not in fields or "use_liger_loss" in fields:
        pytest.skip("installed trl does not show this rename on SFTConfig")

    config = config_class(output_dir = str(tmp_path), report_to = [])
    trainer = Trainer(model = _Bare(), args = config, use_liger_loss = False, use_liger_kernel = True)
    assert trainer.args.use_liger_kernel is True
    assert "is ignored" in capsys.readouterr().out

    Rebuilt, _ = _wrapped_recording()
    rebuilt = Rebuilt(
        model = _Bare(),
        args = None,
        output_dir = str(tmp_path),
        report_to = [],
        use_liger_loss = False,
        use_liger_kernel = True,
    )
    assert rebuilt.args.use_liger_kernel is True


def test_a_name_trl_never_had_still_raises(tmp_path):
    """The retirement table must not become a way to swallow typos."""
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])

    with pytest.raises(TypeError, match = "learnign_rate"):
        Trainer(model = _Bare(), args = config, learnign_rate = 3e-4)


def test_a_variadic_trainer_receives_a_name_the_table_does_not_know(tmp_path):
    """A trainer declaring `**kwargs` opted into arbitrary names, so deliver."""

    class _Variadic(_RecordingTrainer):
        def __init__(
            self,
            model = None,
            args = None,
            train_dataset = None,
            processing_class = None,
            **kwargs,
        ):
            super().__init__(model, args, train_dataset, processing_class)
            self.extra = dict(kwargs)

    Trainer, config_class = _wrapped_recording(trainer_base = _Variadic)
    config = config_class(output_dir = str(tmp_path), report_to = [])

    trainer = Trainer(model = _Bare(), args = config, some_extension_kwarg = 5)

    assert trainer.extra == {"some_extension_kwarg": 5}


def test_a_trainer_whose_config_parameter_is_not_called_args_does_not_KeyError(tmp_path):
    """`discard`, not `remove`: this used to raise KeyError."""

    class _OddlyNamed:
        def __init__(
            self,
            model = None,
            config = None,
            **kwargs,
        ):
            self.model, self.config, self.extra = model, config, dict(kwargs)

    Trainer, config_class = _wrapped_recording(trainer_base = _OddlyNamed)
    config = config_class(output_dir = str(tmp_path), report_to = [])

    trainer = Trainer(model = _Bare(), args = config)

    assert trainer.extra["args"] is config


def test_trainer_kwargs_still_go_to_the_trainer(tmp_path):
    Trainer, config_class = _wrapped_recording()
    config = config_class(output_dir = str(tmp_path), report_to = [])

    trainer = Trainer(model = _Bare(), args = config, train_dataset = "DS", packing = True)

    assert trainer.train_dataset == "DS"


def test_auto_packing_wraps_inside_the_backwards_compatible_wrapper():
    """Auto-packing reads `packing` off the config to block VLMs, custom collators
    and the blocklist, so it has to wrap first and see the moved value. Wrapped
    last it decides on the old one, and the block is undone right after."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_trainer":
            body = ast.get_source_segment(SRC, node)
            break
    else:
        raise AssertionError("_patch_trl_trainer not found")

    assert body.index("_patch_sft_trainer_auto_packing(trl)") < body.index(
        "_backwards_compatible_trainer(trl."
    )


def test_auto_packing_failure_still_leaves_the_wrapper_installed():
    """Going first, a raise here would skip the wrapping loop and drop pre-0.13
    compatibility, so the call has to be guarded."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_trainer":
            guarded = any(
                "_patch_sft_trainer_auto_packing" in ast.dump(stmt)
                for stmt in node.body
                if isinstance(stmt, ast.Try)
            )
            assert guarded, "_patch_sft_trainer_auto_packing must be wrapped in try/except"
            return
    raise AssertionError("_patch_trl_trainer not found")
