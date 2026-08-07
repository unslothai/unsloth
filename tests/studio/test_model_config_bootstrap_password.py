"""Guard the model-config UI smoke's first-boot password contract."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKFLOW_YML = REPO / ".github/workflows/studio-ui-smoke.yml"
PLAYWRIGHT_SCRIPT = REPO / "tests/studio/playwright_model_config.py"


def _step_block(source: str, name: str) -> str:
    start = source.find(f"- name: {name}")
    assert start != -1, f"workflow step {name!r} not found"
    end = source.find("- name:", start + 1)
    return source[start : end if end != -1 else None]


def test_model_config_smoke_passes_the_bootstrap_password():
    workflow = WORKFLOW_YML.read_text(encoding = "utf-8")
    password_step = _step_block(workflow, "Pass bootstrap pw for model-config test")
    assert "cat ~/.unsloth/studio/auth/.bootstrap_password" in password_step
    assert "STUDIO_MODELCFG_OLD_PW" in password_step

    drive_step = _step_block(workflow, "Drive model-picker per-model-config with Playwright")
    assert "STUDIO_OLD_PW: ${{ env.STUDIO_MODELCFG_OLD_PW }}" in drive_step


def test_model_config_playwright_fills_current_password_on_first_boot():
    script = PLAYWRIGHT_SCRIPT.read_text(encoding = "utf-8")
    assert 'BOOTSTRAP_PW = os.environ.get("STUDIO_OLD_PW")' in script
    assert "STUDIO_OLD_PW is required for the first-boot change-password flow" in script
    assert "cur_pw.fill(BOOTSTRAP_PW" in script
