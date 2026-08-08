"""Contracts for the Hugging Face token validation indicator."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"
VALIDATION_HOOK = REPO / "studio/frontend/src/hooks/use-hf-token-validation.ts"
TOKEN_INDICATOR = REPO / "studio/frontend/src/features/hub/components/hf-token-indicator.tsx"
EN_LOCALE = REPO / "studio/frontend/src/i18n/locales/en.ts"
RUN_PREVIEW = REPO / "studio/frontend/src/features/studio/wizard/run-preview-card.tsx"
TRAINING_READINESS = REPO / "studio/frontend/src/features/training/hooks/use-training-readiness.ts"
START_TRAINING_CTA = REPO / "studio/frontend/src/features/studio/wizard/start-training-cta.tsx"
TRAINING_CONFIG_STORE = (
    REPO / "studio/frontend/src/features/training/stores/training-config-store.ts"
)
STUDIO_NAVIGATION = REPO / "studio/frontend/src/features/studio/use-studio-navigation.ts"
TRAIN_SUBNAV = REPO / "studio/frontend/src/features/studio/studio-navigation.tsx"
TRAINING_ACTIONS = REPO / "studio/frontend/src/features/training/hooks/use-training-actions.ts"
FRESH_TRAINING_START = (
    REPO / "studio/frontend/src/features/training/lib/start-fresh-training-run.ts"
)
TRAINING_START_INPUTS = REPO / "studio/frontend/src/features/training/lib/training-start-inputs.ts"
RESUME_TRAINING_START = REPO / "studio/frontend/src/features/training/lib/resume-training-run.ts"
TRAINING_RUNTIME_STORE = (
    REPO / "studio/frontend/src/features/training/stores/training-runtime-store.ts"
)
TRAINING_RUNTIME_LIFECYCLE = (
    REPO / "studio/frontend/src/features/training/hooks/use-training-runtime-lifecycle.ts"
)
TRAINING_START_RUNTIME = (
    REPO / "studio/frontend/src/features/training/lib/training-start-runtime.ts"
)
CONFIRM_TOKEN = REPO / "studio/frontend/src/features/hf-auth/confirm-token.ts"
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
    assert 'normalizedToken && !normalizedToken.startsWith("hf_")' in source
    assert 'error: "Token must start with hf_."' in source
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
    assert "hfApiToken(hfToken) !== undefined" in indicator
    assert 'savedAriaLabel: "Hugging Face token saved"' in en_locale
    assert 'savedHint: "Token saved. Access is checked when you use it."' in en_locale
    assert 'saved: "Saved"' in en_locale
    assert "Allows access to private and gated repos" not in indicator
    assert 't("studio.preview.saved")' in preview
    assert 't("studio.preview.connected")' not in preview
    assert "hfApiToken(hfToken) !== undefined" in preview


def test_model_defaults_error_warns_without_blocking_readiness_and_offers_retry():
    readiness = TRAINING_READINESS.read_text(encoding = "utf-8")
    cta = START_TRAINING_CTA.read_text(encoding = "utf-8")
    config_store = TRAINING_CONFIG_STORE.read_text(encoding = "utf-8")

    assert "modelError: string | null;" in readiness
    assert "const modelError = state.modelDefaultsError;" in readiness
    assert "!modelError &&" not in readiness
    assert "current.modelError === next.modelError" in readiness
    assert 't("studio.training.modelUnverified")' in cta
    assert "!!(modelError || datasetUnverified)" in cta
    assert "ensureModelDefaultsLoaded: state.ensureModelDefaultsLoaded" in cta
    assert "onClick={ensureModelDefaultsLoaded}" in cta
    assert "{showsModelWarning && (" in cta
    assert "const showsModelWarning =" in cta
    assert "!!modelError &&" in cta
    assert "!startError &&" in cta
    ensure_defaults = config_store.split("ensureModelDefaultsLoaded: () =>", 1)[1].split(
        "setProjectName:", 1
    )[0]
    assert "if (defaultsAlreadyApplied" not in ensure_defaults
    assert "applyTrainingDefaults:" in ensure_defaults
    assert "!defaultsAlreadyApplied ||" in ensure_defaults
    assert "canReapplyModelDefaults(state.selectedModel)" in ensure_defaults
    model_defaults_error = config_store.split(".catch((error) =>", 1)[1].split(
        "const runDatasetCheck", 1
    )[0]
    assert (
        model_defaults_error.count("controller.signal.aborted || !requestMatchesSelection()") == 2
    )


def test_training_start_prepares_token_once_before_transport():
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    fresh_start = FRESH_TRAINING_START.read_text(encoding = "utf-8")
    resume_start = RESUME_TRAINING_START.read_text(encoding = "utf-8")
    api = TRAIN_API.read_text(encoding = "utf-8")

    assert "prepareHfTokenForUse" not in actions
    assert fresh_start.count("await prepareHfTokenForUse(") == 1
    assert resume_start.count("await prepareHfTokenForUse(") == 1
    assert "prepareHfTokenForUse" not in api

    start_transport = api.split("export async function startTraining", 1)[1]
    start_transport = start_transport.split("export async function stopTraining", 1)[0]
    request_body = start_transport.split("body: JSON.stringify({", 1)[1].split("}),", 1)[0]
    assert "...payload" in request_body
    assert "start_request_id: startRequestId" in request_body
    assert "hf_token: preparedToken.token" not in start_transport

    resume_entrypoint = resume_start.split("export async function resumeTrainingRun", 1)[1].split(
        "type ResumeAttemptPhase", 1
    )[0]
    assert resume_entrypoint.index("await prepareResumeHfToken(") < (
        resume_entrypoint.index("submitResumeTrainingRun(")
    )
    resume_token = resume_start.split("async function prepareResumeHfToken", 1)[1].split(
        "async function confirmResumeRemoteCode", 1
    )[0]
    assert "await prepareHfTokenForUse(payload.hf_token)" in resume_token
    resume_transport = resume_start.split("async function submitResumeTrainingRun", 1)[1]
    assert "await startTraining(payload, attempt.startRequestId)" in resume_transport
    assert resume_transport.index("attempt.enterTransport()") < (
        resume_transport.index("await startTraining(payload, attempt.startRequestId)")
    )

    prepare_attempt = fresh_start.split("async function prepareAttemptHfToken", 1)[1].split(
        "async function prepareSelectedDataset", 1
    )[0]
    submit_attempt = fresh_start.split("async function submitFreshTrainingRun", 1)[1].split(
        "async function checkSelectedDataset", 1
    )[0]
    assert "await prepareHfTokenForUse(attempt.hfToken)" in prepare_attempt
    assert "buildTrainingStartPayload(attempt.config, hfToken)" in submit_attempt
    assert "payload.hf_token = hfToken" not in submit_attempt
    assert submit_attempt.index("attempt.enterTransport()") < submit_attempt.index(
        "await startTraining(payload, attempt.startRequestId)"
    )


def test_training_start_claims_runtime_before_first_await():
    fresh_start = FRESH_TRAINING_START.read_text(encoding = "utf-8")
    resume_start = RESUME_TRAINING_START.read_text(encoding = "utf-8")
    runtime_store = TRAINING_RUNTIME_STORE.read_text(encoding = "utf-8")
    start_runtime = TRAINING_START_RUNTIME.read_text(encoding = "utf-8")

    entrypoint = fresh_start.split("export async function startFreshTrainingRun", 1)[1].split(
        "type AttemptHfTokenResult", 1
    )[0]
    assert entrypoint.index("FreshTrainingStartAttempt.begin()") < entrypoint.index(
        "await prepareAttemptHfToken(attempt)"
    )

    begin = fresh_start.split("static begin(): FreshTrainingStartAttempt | null", 1)[1].split(
        "get config(): TrainingConfigStore", 1
    )[0]
    assert "tryAcquireTrainingStart()" in begin

    claim = runtime_store.split("tryBeginStarting: (startRequestId) =>", 1)[1].split(
        "setStarting:", 1
    )[0]
    assert "!startRequestId || isTrainingStartPending(state)" in claim
    assert "return { isStarting: true, startRequestId }" in claim
    assert "runtime.tryBeginStarting(startRequestId)" in start_runtime
    acquire = start_runtime.split("export function tryAcquireTrainingStart", 1)[1].split(
        "export function isTrainingStartLeaseActive", 1
    )[0]
    assert acquire.index("createTrainingStartRequestId()") < acquire.index(
        "runtime.tryBeginStarting(startRequestId)"
    )

    resume_entrypoint = resume_start.split("export async function resumeTrainingRun", 1)[1].split(
        "type ResumeAttemptPhase", 1
    )[0]
    assert resume_entrypoint.index("ResumeTrainingStartAttempt.begin()") < (
        resume_entrypoint.index("await loadResumePayload(runId, attempt)")
    )
    resume_begin = resume_start.split("static begin(): ResumeTrainingStartAttempt | null", 1)[
        1
    ].split("get hfToken(): string", 1)[0]
    assert "tryAcquireTrainingStart()" in resume_begin


def test_accepted_training_start_stays_locked_during_preparation():
    runtime_store = TRAINING_RUNTIME_STORE.read_text(encoding = "utf-8")
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    cta = START_TRAINING_CTA.read_text(encoding = "utf-8")
    navigation = STUDIO_NAVIGATION.read_text(encoding = "utf-8")
    subnav = TRAIN_SUBNAV.read_text(encoding = "utf-8")
    history_grid = (REPO / "studio/frontend/src/features/studio/history-card-grid.tsx").read_text(
        encoding = "utf-8"
    )
    history_view = (
        REPO / "studio/frontend/src/features/studio/historical-training-view.tsx"
    ).read_text(encoding = "utf-8")
    dataset_preview = (
        REPO / "studio/frontend/src/features/studio/sections/dataset-preview-dialog.tsx"
    ).read_text(encoding = "utf-8")
    sidebar = (REPO / "studio/frontend/src/components/app-sidebar.tsx").read_text(encoding = "utf-8")
    completion_watch = (
        REPO / "studio/frontend/src/features/training/hooks/use-training-completion-watch.ts"
    ).read_text(encoding = "utf-8")
    unload_guard = (
        REPO / "studio/frontend/src/features/training/hooks/use-training-unload-guard.ts"
    ).read_text(encoding = "utf-8")

    for phase in (
        '"downloading_model"',
        '"downloading_dataset"',
        '"loading_model"',
        '"loading_dataset"',
        '"configuring"',
        '"training"',
    ):
        assert phase in runtime_store
    active = runtime_store.split("export function isTrainingRunActive", 1)[1]
    active = active.split("export function isTrainingStartPending", 1)[0]
    assert "state.isTrainingRunning" in active
    assert "ACTIVE_TRAINING_PHASES.has(state.phase)" in active
    pending = runtime_store.split("export function isTrainingStartPending", 1)[1]
    pending = pending.split("const initialState", 1)[0]
    assert "stopRequested" in pending
    # Per term rather than one exact string: scoped cancellation added the
    # startRequestId disjunct, and formatting of the expression is not the contract.
    for term in (
        "state.stopRequested",
        "state.isStarting",
        "Boolean(state.startRequestId?.trim())",
        "isTrainingRunActive(state)",
    ):
        assert term in pending, term

    assert "useTrainingRuntimeStore(isTrainingStartPending)" in actions
    assert "startBlocked," in actions
    assert "stopRequested," in actions
    assert "startBlocked" in cta
    assert "stopRequested" in cta
    assert "const disabled = startBlocked || !isReady" in cta
    assert "disabled={startBlocked || isResuming}" in history_grid
    assert "disabled={startBlocked || resuming}" in history_view
    assert "startBlocked={startBlocked}" in dataset_preview
    assert "stopRequested={stopRequested}" in dataset_preview
    assert "useTrainingRuntimeStore(isTrainingStartPending)" in sidebar
    assert "useTrainingRuntimeStore(isTrainingStartPending)" in completion_watch
    assert "isTrainingStartPending(useTrainingRuntimeStore.getState())" in unload_guard

    assert "initialStudioTab(selectedHistoryRunId, trainingRunActive)" in navigation
    assert "activeStudioTab(" in navigation
    assert 'trainingRunActive && requestedTab !== "history"' in navigation
    assert "jobId !== previousJobId && trainingRunActive" in navigation
    assert "previousTrainingRunActive: isTrainingRunActive(previousState)" in navigation
    assert "disabled: trainingRunActive" in subnav
    assert "min-w-0 flex-1" in subnav
    assert "flex-wrap" in subnav
    assert "gap-3" in subnav and "sm:gap-6" in subnav


def test_async_training_views_scope_results_to_the_current_request():
    history_view = (
        REPO / "studio/frontend/src/features/studio/historical-training-view.tsx"
    ).read_text(encoding = "utf-8")
    dataset_preview = (
        REPO / "studio/frontend/src/features/studio/sections/dataset-preview-dialog.tsx"
    ).read_text(encoding = "utf-8")

    assert "result?.runId === runId" in history_view
    assert "previous?.runId === runId && previous.detail" in history_view
    assert "setDetail(null)" not in history_view
    assert "requestKey: symbol;" in dataset_preview
    assert "previewResult?.requestKey === requestKey" in dataset_preview
    assert "setPreviewResult({" in dataset_preview
    assert "setData(" not in dataset_preview


def test_training_start_aborts_when_semantic_config_or_token_changes():
    source = FRESH_TRAINING_START.read_text(encoding = "utf-8")
    start_inputs = TRAINING_START_INPUTS.read_text(encoding = "utf-8")

    input_guard = source.split("abortIfInputsChanged(): boolean", 1)[1].split(
        "enterTransport(): boolean", 1
    )[0]
    assert "!this.configInputsChanged()" in input_guard
    assert "getHfToken() === this.expectedHfToken" in input_guard
    assert "this.abortForChangedInputs()" in input_guard
    assert "TRAINING_SETUP_CHANGED_ERROR" in source
    snapshot = source.split("function captureTrainingStartInputs", 1)[1].split(
        "type TrainingStartInputs", 1
    )[0]
    assert "buildTrainingStartPayload(config, null)" in snapshot
    assert "payload.hf_token = null" not in snapshot
    assert "payload.model_known_cached =" not in snapshot
    assert "payload.model_local_path =" not in snapshot
    assert "payload.dataset_known_cached =" not in snapshot
    assert "payload.dataset_local_path =" not in snapshot
    # The identity shape now lives in createTrainingStartInputIdentity, so assert the
    # snapshot delegates to it and that the identity still normalizes and carries the flags.
    assert "createTrainingStartInputIdentity(" in snapshot
    assert "normalizeTrainingStartPayloadForComparison(" in start_inputs
    assert "isUntrainableModelFormat(payload.model_format)" in start_inputs
    assert "modelType: config.modelType" in start_inputs
    assert "isVisionModel: config.isVisionModel" in start_inputs
    assert "isAudioModel: config.isAudioModel" in start_inputs
    assert "useTrainingConfigStore.getState() === this.expectedConfig" not in source

    token_acceptance = source.split("acceptPreparedHfToken(token: string | null): boolean", 1)[
        1
    ].split("\n  updateConfig(", 1)[0]
    assert "currentToken !== this.expectedHfToken" in token_acceptance
    assert "currentToken !== nextToken" in token_acceptance
    assert "useHfTokenStore.getState().setToken(nextToken)" in token_acceptance


def test_anonymous_token_decision_does_not_erase_a_replacement_token():
    source = CONFIRM_TOKEN.read_text(encoding = "utf-8")
    anonymous = source.split('if (decision === "anonymous")', 1)[1].split(
        'if (decision === "replace")', 1
    )[0]

    assert "const tokenStore = useHfTokenStore.getState()" in anonymous
    assert "if (tokenStore.token === normalized)" in anonymous
    assert "tokenStore.clearToken()" in anonymous


def test_token_changes_clear_stale_training_start_errors():
    source = TRAINING_RUNTIME_LIFECYCLE.read_text(encoding = "utf-8")

    assert "useTrainingConfigStore.subscribe(" in source
    assert "state.userEditRevision !== previousState.userEditRevision" in source
    assert "useHfTokenStore.subscribe(" in source
    assert "state.token !== previousState.token" in source
    assert "unsubscribeConfig()" in source
    assert "unsubscribeToken()" in source


def test_accepted_training_start_survives_runtime_resync_failure():
    source = FRESH_TRAINING_START.read_text(encoding = "utf-8")
    runtime = TRAINING_START_RUNTIME.read_text(encoding = "utf-8")
    submit = source.split("async function submitFreshTrainingRun", 1)[1].split(
        "async function checkSelectedDataset", 1
    )[0]

    accepted = source.split("settleAccepted(jobId: string, message: string)", 1)[1].split(
        "private abortForChangedInputs", 1
    )[0]
    assert accepted.index('this.phase = "finished"') < accepted.index(
        "settleAcceptedTrainingStart("
    )
    assert "return attempt.settleAccepted(response.job_id, response.message)" in submit
    adopt = runtime.split("function adoptAcceptedTrainingStart", 1)[1].split(
        "export async function settleAcceptedTrainingStart", 1
    )[0]
    assert ".setStartPending(" in adopt
    settle = runtime.split("export async function settleAcceptedTrainingStart", 1)[1].split(
        "export function settleUnconfirmedTrainingStart", 1
    )[0]
    assert "if (!isTrainingStartLeaseActive(lease))" in settle
    assert "await cancelSupersededTrainingStart(jobId)" in settle
    assert settle.index("adoptAcceptedTrainingStart(jobId, message)") < settle.index(
        "await Promise.allSettled(["
    )
    assert "Promise.resolve().then(emitTrainingRunsChanged)" in settle
    assert "syncTrainingRuntimeFromBackend()" in settle

    cleanup = runtime.split("async function resetSupersededBackendJob", 1)[1].split(
        "export async function settleAcceptedTrainingStart", 1
    )[0]
    assert "await stopTraining(false, { expectedJobId: jobId })" in cleanup
    assert "await resetTraining({ expectedJobId: jobId })" in cleanup
    assert "resetRuntime()" not in cleanup
    assert "syncTrainingRuntimeFromBackend().catch(() => undefined)" in cleanup
    assert cleanup.index("await resetTraining(") < cleanup.index(
        "await syncTrainingRuntimeFromBackend()"
    )

    resume = RESUME_TRAINING_START.read_text(encoding = "utf-8")
    assert "attempt.settleAccepted(response.job_id, response.message)" in resume


def test_superseded_start_cleanup_scopes_both_backend_mutations():
    runtime = TRAINING_START_RUNTIME.read_text(encoding = "utf-8")
    api = TRAIN_API.read_text(encoding = "utf-8")

    cleanup = runtime.split("async function resetSupersededBackendJob", 1)[1].split(
        "export async function settleAcceptedTrainingStart", 1
    )[0]
    assert "cancelSupersededTrainingStart(jobId: string)" in runtime
    assert "stopTraining(false, { expectedJobId: jobId })" in cleanup
    assert "resetTraining({ expectedJobId: jobId })" in cleanup

    body_builder = api.split("function scopedTrainingBody", 1)[1].split(
        "export async function stopTraining", 1
    )[0]
    assert "expected_job_id" in body_builder
    assert "scope.expectedJobId" in body_builder
    reset_transport = api.split("export async function resetTraining", 1)[1].split(
        "export async function getTrainingStatus", 1
    )[0]
    # The scope is no longer conditional: resetTraining takes a RequiredTrainingJobScope
    # and always sends the body, which is strictly stronger than the old hasScope branch.
    assert "scope: RequiredTrainingJobScope" in reset_transport
    assert "body: scopedTrainingBody({}, scope)" in reset_transport


def test_resume_training_preserves_consent_and_error_contracts():
    source = RESUME_TRAINING_START.read_text(encoding = "utf-8")

    consent = source.split("async function confirmResumeRemoteCode", 1)[1].split(
        "async function submitResumeTrainingRun", 1
    )[0]
    assert "await confirmRemoteCodeIfNeeded({" in consent
    assert "trustRemoteCode = true" in consent
    assert "approvedRemoteCodeFingerprint = fingerprint" in consent
    assert "payload.trust_remote_code = trustRemoteCode" in consent
    assert "payload.approved_remote_code_fingerprint = approvedRemoteCodeFingerprint" in consent

    failure = source.split("async fail(error: unknown): Promise<boolean>", 1)[1].split(
        "async function loadResumePayload", 1
    )[0]
    assert 'this.phase === "preflight" && !this.isPreflightActive()' in failure
    assert "normalizeTrainingStartError(" in failure
    assert "failure instanceof Error" in failure
    assert "this.cancel(safeMessage)" in failure
    assert 'toast.error(translate("studio.training.resumeFailedTitle")' in failure
    assert "description: safeMessage" in failure


def test_resume_token_identity_is_guarded_through_preflight():
    source = RESUME_TRAINING_START.read_text(encoding = "utf-8")

    attempt = source.split("class ResumeTrainingStartAttempt", 1)[1].split(
        "async function loadResumePayload", 1
    )[0]
    assert "this.expectedHfToken = getHfToken()" in attempt
    assert "getHfToken() !== this.expectedHfToken" in attempt
    assert "this.cancel(TRAINING_SETUP_CHANGED_ERROR)" in attempt
    token_acceptance = attempt.split("acceptPreparedHfToken(token: string | null): boolean", 1)[
        1
    ].split("enterTransport(): boolean", 1)[0]
    assert "currentToken !== this.expectedHfToken" in token_acceptance
    assert "currentToken !== nextToken" in token_acceptance
    assert "useHfTokenStore.getState().setToken(nextToken)" in token_acceptance
    assert "this.expectedHfToken = getHfToken()" in token_acceptance

    load = source.split("async function loadResumePayload", 1)[1].split(
        "async function prepareResumeHfToken", 1
    )[0]
    assert load.index("await getTrainingRun(runId)") < load.index("attempt.isPreflightActive()")
    assert "payload.hf_token = attempt.hfToken || null" in load
    assert "getResumeHfToken" not in source

    token = source.split("async function prepareResumeHfToken", 1)[1].split(
        "async function confirmResumeRemoteCode", 1
    )[0]
    assert token.index("await prepareHfTokenForUse(payload.hf_token)") < token.index(
        "attempt.acceptPreparedHfToken(preparedToken.token)"
    )

    consent = source.split("async function confirmResumeRemoteCode", 1)[1].split(
        "async function submitResumeTrainingRun", 1
    )[0]
    after_consent = consent.split("await confirmRemoteCodeIfNeeded({", 1)[1]
    assert "attempt.isPreflightActive()" in after_consent

    transport = source.split("async function submitResumeTrainingRun", 1)[1]
    assert transport.index("attempt.enterTransport()") < transport.index(
        "await startTraining(payload, attempt.startRequestId)"
    )


def test_training_stop_failure_preserves_the_runtime_latch():
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    stop = actions.split("const stopTrainingRun = useCallback", 1)[1]
    stop = stop.split("const resumeTrainingRunFromHistory", 1)[0]
    # Scoped cancellation moved the request behind a try/catch, so slice on that rather
    # than on the first sync call, which is now the scope === null early return.
    attempt = stop.split("try {", 1)[1].split("} catch (error) {", 1)[0]
    failure = stop.split("} catch (error) {", 1)[1]

    assert "const scope = trainingStopScope(runtimeStore)" in stop
    assert "const expectedResetGeneration = runtimeStore.resetGeneration" in stop
    assert "{ expectedJobId: scope.jobId }" in attempt
    assert "!runtimeMatchesStopScope(currentRuntime, scope)" in failure
    assert "currentRuntime.resetGeneration !== expectedResetGeneration" in failure
    assert stop.count("await syncTrainingRuntimeFromBackend().catch(() => undefined)") == 5
    assert "currentRuntime.jobId === scope.jobId" in stop
    assert "currentRuntime.resetGeneration === expectedResetGeneration" in stop

    # The latch survives a failed *job* stop. The start branch clears it deliberately so a
    # pending-start cancel stays retryable, so assert per branch, not over the whole body.
    start_branch = failure.split('if (scope.kind === "start") {', 1)[1].split("} else {", 1)[0]
    job_branch = failure.split("} else {", 1)[1]
    assert "currentRuntime.setStopRequested(false)" in start_branch
    assert "currentRuntime.setStopRequested(false)" not in job_branch
    assert "currentRuntime.setRuntimeError(message)" in job_branch


def test_superseded_training_reset_preserves_the_current_runtime():
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    dismiss = actions.split("const dismissTrainingRun = useCallback", 1)[1]
    dismiss = dismiss.split("return {", 1)[0]
    guarded_reset = dismiss.split("const response = await resetTraining(", 1)[1].split(
        "currentRuntime.resetRuntime()", 1
    )[0]

    assert "const scope = trainingStopScope(runtimeStore)" in dismiss
    assert "const expectedResetGeneration = runtimeStore.resetGeneration" in dismiss
    assert "{ expectedJobId: scope.jobId }" in guarded_reset
    assert 'response.status === "superseded"' in guarded_reset
    assert "currentRuntime.jobId !== scope.jobId" in guarded_reset
    assert "currentRuntime.resetGeneration !== expectedResetGeneration" in guarded_reset
    assert "await syncTrainingRuntimeFromBackend().catch(() => undefined)" in guarded_reset
    assert "return;" in guarded_reset
    assert dismiss.count("await syncTrainingRuntimeFromBackend().catch(() => undefined)") == 3
    assert "await syncTrainingRuntimeFromBackend();" not in dismiss


def test_cancel_invalidates_fresh_and_resume_preflight_leases():
    fresh = FRESH_TRAINING_START.read_text(encoding = "utf-8")
    actions = TRAINING_ACTIONS.read_text(encoding = "utf-8")
    resume = RESUME_TRAINING_START.read_text(encoding = "utf-8")
    runtime_store = TRAINING_RUNTIME_STORE.read_text(encoding = "utf-8")
    start_runtime = TRAINING_START_RUNTIME.read_text(encoding = "utf-8")

    stop_setter = runtime_store.split("setStopRequested: (value)", 1)[1].split("setHydrating:", 1)[
        0
    ]
    assert "isStarting: value ? false : state.isStarting" in stop_setter
    assert "value && !state.stopRequested" in stop_setter
    assert "state.resetGeneration + 1" in stop_setter
    stop = actions.split("const stopTrainingRun = useCallback", 1)[1].split(
        "const resumeTrainingRunFromHistory", 1
    )[0]
    # Only the pending-start branch may clear the latch, and only to keep that cancel
    # retryable; a failed job stop must still leave it set.
    job_branch = stop.split("const message =", 1)[1].split("} else {", 1)[1]
    assert "setStopRequested(false)" not in job_branch

    lease_guard = start_runtime.split("export function isTrainingStartLeaseActive", 1)[1].split(
        "export function releaseTrainingStart", 1
    )[0]
    assert "runtime.startRequestId === lease.startRequestId" in lease_guard
    assert "runtime.isStarting" in lease_guard
    assert "!runtime.stopRequested" in lease_guard
    assert "isTrainingStartLeaseActive(this.lease)" in fresh

    pretransport = resume.split("async function submitResumeTrainingRun", 1)[0]
    assert pretransport.count("isTrainingStartLeaseActive(this.lease)") >= 2
    transport = resume.split("async function submitResumeTrainingRun", 1)[1]
    assert "attempt.enterTransport()" in transport
    assert "attempt.settleAccepted(" in transport
