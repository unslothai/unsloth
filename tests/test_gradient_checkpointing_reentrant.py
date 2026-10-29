# SPDX-License-Identifier: AGPL-3.0-only
"""RL configs must keep reentrant checkpointing; non-reentrant recompute aborts on compiled regions."""

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
    # Rebuild the snippet the way the module does rather than re-implementing the quoting.
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
    # torch raises on `context_fn`/`debug` unless use_reentrant=False, so pinning here would crash.
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
    config = _run_post(_Config(gradient_checkpointing = False, kwargs = None))
    assert config.gradient_checkpointing_kwargs is None


def test_the_pin_is_not_gated_on_a_trl_version():
    # The previous guard only ran for TRL 0.27.0+, so older TRL leaked through.
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
