"""GRPO's MoE router-aux fail-fast lands on every known spelling of its anchor, offline."""

from __future__ import annotations

import pytest

from tests.version_compat._rl_anchors import RL_PY, _reject_aux_loss_opt_in


@pytest.mark.parametrize(
    "assignment",
    [
        # TRL 1.7.0 through the last release.
        "self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0",
        # TRL main after #7248: the coefficient falls back to the model config's when None.
        'self.aux_loss_enabled = hasattr(text_config, "output_router_logits") and self.router_aux_loss_coef != 0.0',
    ],
)
def test_the_aux_loss_fail_fast_lands_on_every_known_spelling(assignment: str):
    """Offline: both spellings get the fail-fast at the assignment's indentation, and it runs."""
    src = (
        "class GRPOTrainer:\n"
        "    def __init__(self, args, is_moe, text_config):\n"
        "        self.router_aux_loss_coef = args.router_aux_loss_coef\n"
        f"        {assignment}\n"
        "        self.done = True\n"
    )
    patched = _reject_aux_loss_opt_in(src)
    assert patched.count("raise NotImplementedError") == 1, patched
    ns: dict = {}
    exec(compile(patched, "<grpo>", "exec"), ns)

    class Args:
        router_aux_loss_coef = 0.001

    class MoEConfig:
        output_router_logits = False

    with pytest.raises(NotImplementedError, match = "router_aux_loss_coef = 0"):
        ns["GRPOTrainer"](Args, True, MoEConfig())
    Args.router_aux_loss_coef = 0.0
    assert ns["GRPOTrainer"](Args, True, MoEConfig()).done


def test_rl_py_applies_the_anchor_it_defines():
    """The contract tests exercise the constants; this holds rl.py to actually substituting with them."""
    src = RL_PY.read_text("utf-8")
    use = src.index("_GRPO_AUX_LOSS_ENABLED_LINE,", src.index("_GRPO_AUX_LOSS_ENABLED_LINE =") + 1)
    call = src[src.rindex("re.sub(", 0, use) : use + 400]
    assert "_GRPO_AUX_LOSS_REJECT" in call and "re.MULTILINE" in call, call
    assert 'RLTrainer_name == "GRPOTrainer"' in src[:use]
