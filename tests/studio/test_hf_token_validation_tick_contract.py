"""Contracts for the Hugging Face token validation indicator."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"
VALIDATION_HOOK = REPO / "studio/frontend/src/hooks/use-hf-token-validation.ts"
TOKEN_INDICATOR = (
    REPO / "studio/frontend/src/features/hub/components/hf-token-indicator.tsx"
)
EN_LOCALE = REPO / "studio/frontend/src/i18n/locales/en.ts"
RUN_PREVIEW = (
    REPO / "studio/frontend/src/features/studio/wizard/run-preview-card.tsx"
)
TRAINING_READINESS = (
    REPO / "studio/frontend/src/features/training/hooks/use-training-readiness.ts"
)
START_TRAINING_CTA = (
    REPO / "studio/frontend/src/features/studio/wizard/start-training-cta.tsx"
)
TRAINING_ACTIONS = (
    REPO / "studio/frontend/src/features/training/hooks/use-training-actions.ts"
)
TRAIN_API = REPO / "studio/frontend/src/features/training/api/train-api.ts"


def test_success_tick_requires_the_current_token_to_be_validated():
    source = GENERAL_TAB.read_text(encoding = "utf-8")

    assert "tokenIsCurrent && tokenValidation.isValid === true" in source
    assert 'tokenValidated ? "pr-14" : "pr-8"' in source
    assert "{tokenValidated ? (" in source
    assert 'aria-label={t("settings.general.tokenValidated")}' in source
    assert 'aria-label={t("settings.general.tokenSaved")}' not in source


def test_validation_result_must_belong_to_the_current_normalized_token():
    source = VALIDATION_HOOK.read_text(encoding = "utf-8")

    assert "const normalizedToken = token.trim()" in source
    assert "useDebouncedValue(normalizedToken, 500)" in source
    assert "if (!COMPLETE_HF_TOKEN.test(normalizedToken)) return INITIAL" in source
    assert "if (completed.token !== normalizedToken)" in source
    assert "if (completed.token !== debouncedToken)" not in source


def test_saved_token_is_not_reported_as_connected():
    indicator = TOKEN_INDICATOR.read_text(encoding = "utf-8")
    en_locale = EN_LOCALE.read_text(encoding = "utf-8")
    preview = RUN_PREVIEW.read_text(encoding = "utf-8")

    assert 't("picker.hfToken.savedAriaLabel")' in indicator
    assert 't("picker.hfToken.savedHint")' in indicator
    assert 't("picker.hfToken.saved")' in indicator
    assert 'savedAriaLabel: "Hugging Face token saved"' in en_locale
    assert 'savedHint: "Token saved. Access is checked when you use it."' in en_locale
    assert 'saved: "Saved"' in en_locale
    assert "Allows access to private and gated repos" not in indicator
    assert 't("studio.preview.saved")' in preview
    assert 't("studio.preview.connected")' not in preview


def test_model_defaults_error_blocks_readiness_and_offers_retry():
    readiness = TRAINING_READINESS.read_text(encoding = "utf-8")
    cta = START_TRAINING_CTA.read_text(encoding = "utf-8")

    assert "modelError: string | null;" in readiness
    assert "const modelError = state.modelDefaultsError;" in readiness
    assert "!modelError &&" in readiness
    assert "readinessCache.modelError === next.modelError" in readiness
    assert 't("studio.training.modelUnverified")' in cta
    assert "ensureModelDefaultsLoaded: state.ensureModelDefaultsLoaded" in cta
    assert "onClick={ensureModelDefaultsLoaded}" in cta
    assert "{modelError && !startError && (" in cta


def test_training_start_prepares_token_once_before_transport():
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    api = TRAIN_API.read_text(encoding = "utf-8")

    assert actions.count("await prepareHfTokenForUse(") == 2
    assert "prepareHfTokenForUse" not in api

    start_transport = api.split("export async function startTraining", 1)[1]
    start_transport = start_transport.split(
        "export async function stopTraining", 1
    )[0]
    assert "body: JSON.stringify(payload)" in start_transport
    assert "hf_token: preparedToken.token" not in start_transport

    fresh_start = actions.split(
        "const startTrainingRun = useCallback", 1
    )[1].split("const resumeTrainingRunFromHistory = useCallback", 1)[0]
    resume_start = actions.split(
        "const resumeTrainingRunFromHistory = useCallback", 1
    )[1]
    for flow in (fresh_start, resume_start):
        assert flow.index("await prepareHfTokenForUse(") < flow.index(
            "await startTraining(payload)"
        )

    assert "const actionHfToken = preparedToken.token;" in fresh_start
    assert fresh_start.count("hfToken: actionHfToken") == 3
    assert "hf_token: actionHfToken" in fresh_start
    assert "hfToken: getHfToken()" not in fresh_start
