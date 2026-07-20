"""Contracts for the Hugging Face token validation indicator."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"
VALIDATION_HOOK = REPO / "studio/frontend/src/hooks/use-hf-token-validation.ts"


def test_success_tick_requires_the_current_token_to_be_validated():
    source = GENERAL_TAB.read_text(encoding="utf-8")

    assert "tokenIsCurrent && tokenValidation.isValid === true" in source
    assert 'tokenValidated ? "pr-14" : "pr-8"' in source
    assert "{tokenValidated ? (" in source
    assert 'aria-label={t("settings.general.tokenValidated")}' in source
    assert 'aria-label={t("settings.general.tokenSaved")}' not in source


def test_validation_result_must_belong_to_the_current_normalized_token():
    source = VALIDATION_HOOK.read_text(encoding="utf-8")

    assert "const normalizedToken = token.trim()" in source
    assert "useDebouncedValue(normalizedToken, 500)" in source
    assert "if (!COMPLETE_HF_TOKEN.test(normalizedToken)) return INITIAL" in source
    assert "if (completed.token !== normalizedToken)" in source
    assert "if (completed.token !== debouncedToken)" not in source
