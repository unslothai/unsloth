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
    block = _block_around(THREAD_TSX.read_text(), 'aria-label="Message input"')
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
    code_only = re.sub(r'""".*?"""', "", src, flags = re.DOTALL)
    assert (
        "STUDIO_OLD_PW" not in code_only
    ), "IME Playwright script still references dead STUDIO_OLD_PW env var"
    assert 'os.environ["STUDIO_NEW_PW"]' in code_only
