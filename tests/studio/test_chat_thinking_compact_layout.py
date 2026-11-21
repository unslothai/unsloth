"""Responsive contract for the composer Thinking control and the model chip that replaced it."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
COMPOSER_PICKER_TSX = (
    REPO
    / "studio/frontend/src/components/assistant-ui/composer-model-picker/composer-model-picker.tsx"
)
SHARED_TSX = REPO / "studio/frontend/src/features/chat/shared-composer.tsx"
INDEX_CSS = REPO / "studio/frontend/src/index.css"


def test_thinking_control_has_compact_hooks_in_the_compare_composer():
    source = SHARED_TSX.read_text(encoding = "utf-8")
    assert 'className="unsloth-thinking-label"' in source
    # Wrapped in the space scale so it grows with the composer; a bare 15px would not.
    assert (
        "unsloth-thinking-caret size-[calc(15px*var(--ui-space-scale,1))]" in source
    ), "the Thinking caret lost its hook or its scaled 15px size"
    assert 'data-pill-label="Thinking settings"' in source


def test_model_chip_keeps_the_compact_hooks():
    source = COMPOSER_PICKER_TSX.read_text(encoding = "utf-8")
    assert '"unsloth-thinking-pill composer-model-chip' in source
    assert "unsloth-thinking-label composer-model-chip-name" in source
    assert "unsloth-thinking-label composer-model-chip-detail" in source
    # The bulb-only chip names its model on hover from this attribute.
    assert "data-pill-label={chipLabel}" in source


def test_narrow_composer_collapses_thinking_to_the_bulb():
    css = INDEX_CSS.read_text(encoding = "utf-8")

    assert css.count("container-type: inline-size;") >= 2
    compact_start = css.index("@container (max-width: 36rem)")
    compact_end = css.index("\t.unsloth-tick {", compact_start)
    compact_rule = css[compact_start:compact_end]

    assert ".unsloth-thinking-pill" in compact_rule
    assert "@apply size-8" in compact_rule
    assert ".unsloth-thinking-label" in compact_rule
    assert ".unsloth-thinking-caret" in compact_rule
    assert "display: none;" in compact_rule
    # An empty chip keeps "Select model" readable instead of a blank bulb.
    assert ".composer-model-chip-empty .unsloth-thinking-label" in compact_rule
