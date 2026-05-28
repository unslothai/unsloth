"""Lock down the RTL bidi auto-detection contract on the chat composers.

The browser's Unicode bidi algorithm only flows Arabic / Hebrew / Persian /
Urdu right-to-left when the textarea carries `dir="auto"`. The three
composer surfaces (main chat, inline edit, compare mode) each need the
attribute, and the IME / i18n Playwright smoke must keep its env contract
minimal (no dead `STUDIO_OLD_PW`).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
SHARED_TSX = REPO / "studio/frontend/src/features/chat/shared-composer.tsx"
WORKFLOW_YML = REPO / ".github/workflows/studio-ui-smoke.yml"
IME_PY = REPO / "tests/studio/playwright_chat_ime_i18n.py"


def _block_around(src: str, anchor: str, radius: int = 600) -> str:
    idx = src.find(anchor)
    assert idx != -1, f"anchor {anchor!r} not found"
    return src[max(idx - radius, 0) : idx + radius]


def test_main_composer_has_dir_auto():
    # PR #5784 rewrote the literal attribute into a JSX conditional
    # (`aria-label={overlay ? "Image edit instructions" : "Message input"}`),
    # so anchor on the inner string literal instead -- it survives both
    # the old and new spellings.
    block = _block_around(THREAD_TSX.read_text(), '"Message input"')
    assert 'dir="auto"' in block, 'main composer is missing dir="auto"'


def test_edit_composer_has_dir_auto():
    block = _block_around(THREAD_TSX.read_text(), "aui-edit-composer-input")
    assert 'dir="auto"' in block, 'edit composer is missing dir="auto"'


def test_compare_composer_has_dir_auto():
    block = _block_around(SHARED_TSX.read_text(), "Send to both models")
    assert 'dir="auto"' in block, 'compare composer is missing dir="auto"'


def test_ime_workflow_step_does_not_set_studio_old_pw():
    yml = WORKFLOW_YML.read_text()
    drive_idx = yml.find("Drive IME + multilingual paste regression")
    assert drive_idx != -1, "IME drive step not found in workflow"
    next_step_idx = yml.find("- name:", drive_idx + 1)
    drive_block = yml[drive_idx : next_step_idx if next_step_idx != -1 else None]
    assert (
        "STUDIO_OLD_PW" not in drive_block
    ), "IME drive step still passes dead STUDIO_OLD_PW env var"
    assert "STUDIO_NEW_PW" in drive_block, "IME drive step missing STUDIO_NEW_PW"


def test_ime_pass_password_step_does_not_export_old_pw():
    yml = WORKFLOW_YML.read_text()
    pass_idx = yml.find("Pass bootstrap pw for IME / i18n test")
    assert pass_idx != -1, "IME password setup step not found"
    next_step_idx = yml.find("- name:", pass_idx + 1)
    pass_block = yml[pass_idx : next_step_idx if next_step_idx != -1 else None]
    assert (
        "STUDIO_IME_OLD_PW" not in pass_block
    ), "IME password setup still exports dead STUDIO_IME_OLD_PW"
    assert "STUDIO_IME_NEW_PW" in pass_block


def test_ime_playwright_script_does_not_read_studio_old_pw():
    src = IME_PY.read_text()
    code_only = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    assert (
        "STUDIO_OLD_PW" not in code_only
    ), "IME Playwright script still references dead STUDIO_OLD_PW env var"
    assert 'os.environ["STUDIO_NEW_PW"]' in code_only


def test_main_composer_has_stuck_compositionend_watchdog():
    """Issue #5546: Chrome on Windows over WSL never emits compositionend
    after the IME commit. The composer keeps a watchdog that releases the
    composing flag once events go silent; without it Send stays disabled
    forever and CJK input is effectively dropped."""
    src = THREAD_TSX.read_text()
    assert "IME_STUCK_TIMEOUT_MS" in src, (
        "main composer is missing the stuck-compositionend watchdog " "(issue #5546)"
    )
    assert "onCompositionUpdate" in src, (
        "main composer is missing onCompositionUpdate wiring; the "
        "watchdog only resets while the IME is actively emitting events"
    )


def test_compare_composer_has_stuck_compositionend_watchdog():
    src = SHARED_TSX.read_text()
    assert "IME_STUCK_TIMEOUT_MS" in src, (
        "compare composer is missing the stuck-compositionend watchdog " "(issue #5546)"
    )
    assert (
        "onCompositionUpdate" in src
    ), "compare composer is missing onCompositionUpdate wiring"


def test_main_composer_keydown_repins_composing_during_ime():
    """Issue #5546 watchdog can clear composingRef during a long candidate
    pause; the IME keydown gate must re-pin it so a follow-up Enter does not
    submit preedit text."""
    src = THREAD_TSX.read_text()
    assert "onKeyDown" in src, "main composer is missing onKeyDown IME gate"
    assert "e.nativeEvent.isComposing" in src and "keyCode === 229" in src, (
        "main composer keydown gate must check both nativeEvent.isComposing "
        "and the IME keyCode 229 sentinel"
    )


def test_compare_composer_keydown_repins_composing_during_ime():
    """Compare composer onKeyDown re-pins composingRef on IME keypress so a
    follow-up click-Send during the watchdog window does not slip preedit
    text through."""
    src = SHARED_TSX.read_text()
    assert "composingRef.current = true" in src, (
        "compare composer keydown gate must re-pin composingRef when the "
        "browser still considers the IME active"
    )


def _extract_block(src: str, anchor: str, opener: str = "(", closer: str = ")") -> str:
    """Return the source between the first balanced opener/closer that
    starts at or after `anchor`. Used to scope assertions to a specific
    handler so a re-arm call in some other function does not satisfy
    the gate test."""
    start = src.find(anchor)
    assert start != -1, f"anchor {anchor!r} not found"
    open_idx = src.find(opener, start)
    assert open_idx != -1, f"opener {opener!r} after {anchor!r} not found"
    depth = 0
    for i in range(open_idx, len(src)):
        c = src[i]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError(f"unbalanced {opener!r}/{closer!r} after {anchor!r}")


def test_main_composer_keydown_rearms_watchdog():
    """After the keydown re-pin sets composingRef=true the watchdog must
    be re-armed; otherwise the WSL+Chrome no-compositionend path this PR
    targets would lock Send permanently after any IME keypress
    (Codex P1 on commit 597af0d0)."""
    src = THREAD_TSX.read_text()
    block = _extract_block(src, "const onKeyDown = useCallback")
    assert "refreshStuckTimer" in block, (
        "main composer keydown gate must call refreshStuckTimer after "
        "re-pinning composingRef so the watchdog runs again on the "
        "stuck-compositionend path"
    )
    assert "clearStuckTimer();" not in block.replace("clearStuckTimer\n", "").replace(
        "clearStuckTimer,", ""
    ), (
        "main composer keydown gate must not leave the watchdog only "
        "cleared — that's the Codex P1 regression"
    )


def test_compare_composer_keydown_rearms_watchdog():
    """Same re-arm contract for the compare-mode composer."""
    src = SHARED_TSX.read_text()
    block = _extract_block(src, "function onKeyDown", opener="{", closer="}")
    assert "refreshStuckImeTimer" in block, (
        "compare composer keydown gate must call refreshStuckImeTimer "
        "after re-pinning composingRef"
    )
