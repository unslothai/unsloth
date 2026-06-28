# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenizer load-time compatibility shims.

Some converted/quantized checkpoints ship tokenizer_config.json with
``extra_special_tokens`` as a JSON array (``[]``) instead of an object (``{}``);
transformers then runs ``list(special_tokens.keys())`` and raises
"'list' object has no attribute 'keys'". We coerce non-dict values to ``{}``.
"""

from loggers import get_logger

logger = get_logger(__name__)

_PATCH_FLAG = "_unsloth_extra_special_tokens_compat"


def install_extra_special_tokens_compat() -> bool:
    """Coerce a non-dict ``extra_special_tokens`` to ``{}`` during tokenizer init.

    Wraps ``SpecialTokensMixin._set_model_specific_special_tokens`` so a model whose
    tokenizer_config.json has a malformed (array) ``extra_special_tokens`` loads
    instead of raising "'list' object has no attribute 'keys'". Idempotent and
    cheap; safe to call before every load. Returns True when active, False when the
    method is absent (such builds lack the bug).
    """
    try:
        import transformers.tokenization_utils_base as tub
    except Exception:
        return False

    mixin = getattr(tub, "SpecialTokensMixin", None)
    orig = getattr(mixin, "_set_model_specific_special_tokens", None)
    if mixin is None or orig is None:
        return False
    if getattr(mixin, _PATCH_FLAG, False):
        return True

    def _patched(self, special_tokens):
        if not isinstance(special_tokens, dict):
            logger.warning(
                "Coercing malformed extra_special_tokens (%s) to {}; "
                "tokenizer_config.json should use an object, not an array.",
                type(special_tokens).__name__,
            )
            special_tokens = {}
            try:
                self.extra_special_tokens = {}
            except Exception:
                pass
        return orig(self, special_tokens)

    _patched.__wrapped__ = orig  # keep original reachable
    mixin._set_model_specific_special_tokens = _patched
    setattr(mixin, _PATCH_FLAG, True)
    return True
