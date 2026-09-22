# SPDX-License-Identifier: AGPL-3.0-only
"""Generated RL configs must keep the reentrant gradient checkpoint path.

Unsloth gradient checkpointing needs `use_reentrant=True`. The non-reentrant
path recomputes every packed forward during backward and compares what each
pass saved, so a region packed compiled and recomputed eagerly aborts with

    CheckpointError: A different number of tensors was saved during the
    original forward and recomputation.

Two things push a config to non-reentrant, and only one of them was handled:

  TRL 0.27.0+ sets use_reentrant=False explicitly.

  transformers substitutes {"use_reentrant": False} whenever
  gradient_checkpointing_kwargs is None, which is what an older TRL leaves.

The second is not hypothetical. GKDConfig turns gradient_checkpointing on by
default, so knowledge distillation on TRL 0.25.1 reached the non-reentrant path
without ever asking for gradient checkpointing, and died on a Kaggle 2x T4
with 81 tensors saved in the forward against 79 in the recomputation.
"""

import re
import textwrap


def _rl_source() -> str:
    import inspect
    from unsloth.models import rl
    return inspect.getsource(rl)


def _config_post_snippet() -> str:
    src = _rl_source()
    match = re.search(r"RLConfig_post = \(\n(.*?)\n    \)\n", src, re.S)
    assert match is not None, "RLConfig_post assignment not found"
    # The snippet is a concatenation of quoted lines; rebuild it the way the
    # module does rather than re-implementing the quoting.
    return eval("(\n" + match.group(1) + "\n)", {"__builtins__": {}})


class _Config:
    def __init__(
        self,
        gradient_checkpointing = True,
        kwargs = None,
    ):
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_kwargs = kwargs


def _run_post(config):
    snippet = textwrap.dedent(_config_post_snippet())
    body = "def _post(self):\n" + textwrap.indent(snippet, "    ")
    ns = {}
    exec(compile(body, "<RLConfig_post>", "exec"), ns)
    ns["_post"](config)
    return config


def test_a_config_that_never_set_the_kwargs_gets_reentrant_pinned():
    # This is the case transformers would otherwise fill in with False.
    config = _run_post(_Config(kwargs = None))
    assert config.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_an_explicit_false_is_overridden():
    config = _run_post(_Config(kwargs = {"use_reentrant": False}))
    assert config.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_other_checkpoint_kwargs_are_preserved():
    config = _run_post(_Config(kwargs = {"determinism_check": "none"}))
    assert config.gradient_checkpointing_kwargs == {
        "determinism_check": "none",
        "use_reentrant": True,
    }


def test_a_config_asking_for_context_fn_is_left_alone():
    # torch/utils/checkpoint.py raises "Passing `context_fn` or `debug` is only
    # supported when use_reentrant=False" as soon as a checkpointed forward
    # runs, so pinning here would turn a working setup into a crash.
    sentinel = object()
    config = _run_post(_Config(kwargs = {"use_reentrant": False, "context_fn": sentinel}))
    assert config.gradient_checkpointing_kwargs == {
        "use_reentrant": False,
        "context_fn": sentinel,
    }


def test_a_config_asking_for_debug_is_left_alone():
    config = _run_post(_Config(kwargs = {"use_reentrant": False, "debug": True}))
    assert config.gradient_checkpointing_kwargs == {"use_reentrant": False, "debug": True}


def test_a_falsy_debug_does_not_block_the_pin():
    # debug=False is the torch default, so it is not a non-reentrant request.
    config = _run_post(_Config(kwargs = {"debug": False}))
    assert config.gradient_checkpointing_kwargs == {"debug": False, "use_reentrant": True}


def test_checkpointing_off_is_left_completely_alone():
    # transformers never reads these kwargs in that case, so touching them
    # would only widen the blast radius.
    config = _run_post(_Config(gradient_checkpointing = False, kwargs = None))
    assert config.gradient_checkpointing_kwargs is None


def test_the_pin_is_not_gated_on_a_trl_version():
    # The previous guard only ran for TRL 0.27.0+, which is exactly why older
    # TRL leaked through to the non-reentrant path.
    src = _rl_source()
    match = re.search(r"RLConfig_post = \(\n(.*?)\n    \)\n", src, re.S)
    assert match is not None
    preceding = src[: match.start()]
    tail = preceding[-600:]
    assert 'Version("0.27.0")' not in tail, (
        "RLConfig_post is gated on a TRL version again; older TRL leaves "
        "gradient_checkpointing_kwargs as None and transformers then picks "
        "use_reentrant=False for it"
    )


def test_the_snippet_pins_true_rather_than_deleting_the_key():
    snippet = _config_post_snippet()
    assert "use_reentrant'] = True" in snippet or '"use_reentrant"] = True' in snippet, snippet
    assert "del " not in snippet, (
        "deleting the key is not enough: a config that never set it leaves "
        "None, and transformers substitutes False"
    )
