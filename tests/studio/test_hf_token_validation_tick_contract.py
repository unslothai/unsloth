"""Contracts for the Hugging Face token validation indicator."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"


def test_success_tick_requires_the_current_token_to_be_validated():
    source = GENERAL_TAB.read_text(encoding = "utf-8")

    assert "tokenIsCurrent && tokenValidation.isValid === true" in source
    assert 'tokenValidated ? "pr-14" : "pr-8"' in source
    assert "{tokenValidated ? (" in source
    assert 'aria-label={t("settings.general.tokenValidated")}' in source
    assert 'aria-label={t("settings.general.tokenSaved")}' not in source
