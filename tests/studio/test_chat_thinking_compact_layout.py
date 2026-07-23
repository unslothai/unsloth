"""Responsive contract for the composer Thinking control."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
SHARED_TSX = REPO / "studio/frontend/src/features/chat/shared-composer.tsx"
INDEX_CSS = REPO / "studio/frontend/src/index.css"


def test_thinking_control_has_compact_hooks_in_both_composers():
    for path in (THREAD_TSX, SHARED_TSX):
        source = path.read_text(encoding = "utf-8")
        assert 'className="unsloth-thinking-label"' in source
        assert "unsloth-thinking-caret size-[15px]" in source
        assert 'data-pill-label="Thinking settings"' in source


def test_narrow_composer_collapses_thinking_to_the_bulb():
    css = INDEX_CSS.read_text(encoding = "utf-8")

    # Query the composer width instead of the full viewport.
    assert css.count("container-type: inline-size;") >= 2
    compact_start = css.index("@container (max-width: 576px)")
    compact_end = css.index("/* Smaller tick", compact_start)
    compact_rule = css[compact_start:compact_end]

    assert ".unsloth-thinking-pill" in compact_rule
    assert "@apply size-8" in compact_rule
    assert ".unsloth-thinking-label" in compact_rule
    assert ".unsloth-thinking-caret" in compact_rule
    assert "display: none;" in compact_rule
