// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import { getHfToken, useHfTokenStore } from "@/features/hub";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { translate } from "@/i18n";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { DatasetFormatError, checkDatasetFormat } from "../api/datasets-api";
import { buildTrainingStartPayload } from "../api/mappers";
import {
  TrainingStartError,
  isTrainingStartOutcomeUnknownError,
  startTraining,
} from "../api/train-api";
import { useDatasetPreviewDialogStore } from "../stores/dataset-preview-dialog-store";
import {
  clearDeletedDataset,
  useTrainingConfigStore,
} from "../stores/training-config-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingConfigState, TrainingConfigStore } from "../types/config";
import type { CheckFormatResponse } from "../types/datasets";
import { cacheLocalPathMatchesSelection } from "./cache-reference";
import {
  createDatasetCacheUsabilityIdentity,
  trainingDatasetCacheRejections,
} from "./dataset-cache-rejection";
import { shouldUseVisionDatasetCheck } from "./fresh-dataset-check";
import { isMissingLocalDatasetCacheError } from "./local-cache-errors";
import { isRawTextDatasetFormat } from "./training-methods";
import { normalizeTrainingStartError } from "./training-start-errors";
import { createTrainingStartInputIdentity } from "./training-start-inputs";
import {
  TRAINING_SETUP_CHANGED_ERROR,
  type TrainingStartLease,
  isTrainingStartLeaseActive,
  reconcileTrainingStartTransportFailure,
  releaseTrainingStart,
  settleAcceptedTrainingStart,
  settleUnconfirmedTrainingStart,
  tryAcquireTrainingStart,
} from "./training-start-runtime";
import {
  hasIncompatibleTrainingModalities,
  validateTrainingConfig,
} from "./validation";

const ROLE_REMAP: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

type AttemptPhase = "preflight" | "transport" | "finished";

function captureTrainingStartInputs(config: TrainingConfigState) {
  return createTrainingStartInputIdentity(
    buildTrainingStartPayload(config, null),
    config,
  );
}

type TrainingStartInputs = ReturnType<typeof captureTrainingStartInputs>;

function trainingStartInputsEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left)) {
    return (
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) =>
        trainingStartInputsEqual(value, right[index]),
      )
    );
  }
  if (
    left === null ||
    right === null ||
    typeof left !== "object" ||
    typeof right !== "object"
  ) {
    return false;
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord);
  return (
    leftKeys.length === Object.keys(rightRecord).length &&
    leftKeys.every(
      (key) =>
        Object.hasOwn(rightRecord, key) &&
        trainingStartInputsEqual(leftRecord[key], rightRecord[key]),
    )
  );
}

class FreshTrainingStartAttempt {
  private readonly lease: TrainingStartLease;
  private expectedConfig: TrainingConfigStore;
  private expectedInputs: TrainingStartInputs;
  private expectedHfToken: string;
  private phase: AttemptPhase = "preflight";

  constructor(lease: TrainingStartLease) {
    this.lease = lease;
    this.expectedConfig = useTrainingConfigStore.getState();
    this.expectedInputs = captureTrainingStartInputs(this.expectedConfig);
    this.expectedHfToken = getHfToken();
  }

  static begin(): FreshTrainingStartAttempt | null {
    const lease = tryAcquireTrainingStart();
    if (!lease) {
      return null;
    }
    const runtime = useTrainingRuntimeStore.getState();
    runtime.setStartError(null);
    const attempt = new FreshTrainingStartAttempt(lease);
    runtime.setStartResources(
      attempt.config.selectedModel,
      getHfDatasetName(attempt.config),
      false,
      attempt.config.projectName || "",
    );
    return attempt;
  }

  get config(): TrainingConfigStore {
    return this.expectedConfig;
  }

  get hfToken(): string {
    return this.expectedHfToken;
  }

  get startRequestId(): string {
    return this.lease.startRequestId;
  }

  acceptPreparedHfToken(token: string | null): boolean {
    if (this.phase !== "preflight") {
      return false;
    }
    if (!isTrainingStartLeaseActive(this.lease)) {
      return this.invalidate();
    }
    if (this.configInputsChanged()) {
      return this.abortForChangedInputs();
    }
    const nextToken = token ?? "";
    const currentToken = getHfToken();
    if (currentToken !== this.expectedHfToken && currentToken !== nextToken) {
      return this.abortForChangedInputs();
    }
    if (currentToken !== nextToken) {
      useHfTokenStore.getState().setToken(nextToken);
    }
    this.expectedHfToken = getHfToken();
    return true;
  }

  updateConfig(
    update: Partial<TrainingConfigState>,
    applyUpdate: () => void = () => {
      useTrainingConfigStore.setState(update);
    },
  ): boolean {
    if (this.abortIfInputsChanged()) {
      return false;
    }
    applyUpdate();
    this.expectedConfig = { ...this.expectedConfig, ...update };
    this.expectedInputs = captureTrainingStartInputs(this.expectedConfig);
    return !this.abortIfInputsChanged();
  }

  abortIfInputsChanged(): boolean {
    if (this.phase === "finished") {
      return true;
    }
    if (this.phase === "transport") {
      return false;
    }
    if (!isTrainingStartLeaseActive(this.lease)) {
      this.invalidate();
      return true;
    }
    if (!this.configInputsChanged() && getHfToken() === this.expectedHfToken) {
      return false;
    }
    this.abortForChangedInputs();
    return true;
  }

  enterTransport(): boolean {
    if (this.abortIfInputsChanged()) {
      return false;
    }
    this.phase = "transport";
    return true;
  }

  cancel(error?: string | null): false {
    if (this.phase === "finished") {
      return false;
    }
    this.phase = "finished";
    return releaseTrainingStart(this.lease, error);
  }

  settleAccepted(jobId: string, message: string): Promise<boolean> {
    this.phase = "finished";
    return settleAcceptedTrainingStart(this.lease, jobId, message);
  }

  settleUnconfirmed(message: string): boolean {
    this.phase = "finished";
    return settleUnconfirmedTrainingStart(this.lease, message);
  }

  async recoverTransportFailure() {
    if (this.phase !== "transport") {
      return { kind: "unknown" } as const;
    }
    const recovery = await reconcileTrainingStartTransportFailure(this.lease);
    if (recovery.kind === "recovered") {
      this.phase = "finished";
    }
    return recovery;
  }

  private abortForChangedInputs(): false {
    return this.cancel(TRAINING_SETUP_CHANGED_ERROR);
  }

  private configInputsChanged(): boolean {
    return !trainingStartInputsEqual(
      captureTrainingStartInputs(useTrainingConfigStore.getState()),
      this.expectedInputs,
    );
  }

  private invalidate(): false {
    this.phase = "finished";
    return false;
  }
}

export async function startFreshTrainingRun(): Promise<boolean> {
  const attempt = FreshTrainingStartAttempt.begin();
  if (!attempt) {
    return false;
  }

  const validation = validateTrainingConfig(
    attempt.config,
    usePlatformStore.getState().deviceType,
  );
  if (!validation.ok) {
    return attempt.cancel(translate(validation.errorKey));
  }

  try {
    const tokenResult = await prepareAttemptHfToken(attempt);
    if (!tokenResult.ready) {
      return false;
    }
    primeNativeNotificationPermission().catch(() => undefined);

    if (!(await prepareSelectedDataset(attempt, tokenResult.token))) {
      return false;
    }
    if (useTrainingRuntimeStore.getState().stopRequested) {
      return attempt.cancel();
    }
    if (!(await confirmSelectedModelRemoteCode(attempt, tokenResult.token))) {
      return false;
    }
    return await submitFreshTrainingRun(attempt, tokenResult.token);
  } catch (error) {
    if (isTrainingStartOutcomeUnknownError(error)) {
      const recovery = await attempt.recoverTransportFailure();
      if (recovery.kind === "recovered") {
        return true;
      }
      if (recovery.kind === "rejected") {
        return attempt.cancel(
          normalizeTrainingStartError(recovery.error, recovery.errorCode),
        );
      }
      const message = translate("studio.training.startUnconfirmed");
      toast.warning(message);
      return attempt.settleUnconfirmed(message);
    }
    if (attempt.abortIfInputsChanged()) {
      return false;
    }
    return attempt.cancel(
      normalizeTrainingStartError(
        error instanceof Error
          ? error
          : translate("studio.training.startFailed"),
      ),
    );
  }
}

type AttemptHfTokenResult =
  | { ready: false }
  | { ready: true; token: string | null };

async function prepareAttemptHfToken(
  attempt: FreshTrainingStartAttempt,
): Promise<AttemptHfTokenResult> {
  const preparedToken = await prepareHfTokenForUse(attempt.hfToken);
  if (!attempt.acceptPreparedHfToken(preparedToken.token)) {
    return { ready: false };
  }
  if (!preparedToken.proceed) {
    attempt.cancel();
    return { ready: false };
  }
  return { ready: true, token: preparedToken.token };
}

async function prepareSelectedDataset(
  attempt: FreshTrainingStartAttempt,
  hfToken: string | null,
): Promise<boolean> {
  const datasetName = getDatasetName(attempt.config);
  if (!datasetName) {
    return true;
  }

  const isVlm = shouldUseVisionDatasetCheck(attempt.config);
  const check = await checkSelectedDataset(
    attempt,
    datasetName,
    hfToken,
    isVlm,
  );
  if (!check) {
    return false;
  }

  const isAudio = check.is_audio === true;
  const isImage = check.is_image === true;
  const recheckDetectedVisionDataset =
    !isVlm && shouldUseVisionDatasetCheck(attempt.config, isImage);
  const recheckCachedDataset =
    attempt.config.datasetStreaming &&
    (isImage || isAudio) &&
    attempt.config.datasetSource === "huggingface" &&
    attempt.config.datasetKnownCached;
  if (!applyDetectedDatasetModality(attempt, isImage, isAudio)) {
    return false;
  }
  if (recheckCachedDataset || recheckDetectedVisionDataset) {
    return prepareSelectedDataset(attempt, hfToken);
  }
  if (hasIncompatibleTrainingModalities(attempt.config)) {
    return attempt.cancel();
  }
  if (!needsManualMapping(attempt.config, check, isVlm, isAudio)) {
    return true;
  }
  return openManualMapping(attempt, check, isVlm, isAudio);
}

function applyDetectedDatasetModality(
  attempt: FreshTrainingStartAttempt,
  isImage: boolean,
  isAudio: boolean,
): boolean {
  if (attempt.config.datasetStreaming && (isImage || isAudio)) {
    if (
      !attempt.updateConfig({
        datasetStreaming: false,
        isDatasetImage: isImage,
        isDatasetAudio: isAudio,
        datasetCheckFailed: false,
      })
    ) {
      return false;
    }
    toast.info(
      translate(
        "studio.dataset.streaming.notifications.disabledForDetectedModality",
      ),
    );
    return true;
  }
  if (
    isImage === attempt.config.isDatasetImage &&
    isAudio === attempt.config.isDatasetAudio &&
    !attempt.config.datasetCheckFailed
  ) {
    return true;
  }
  return attempt.updateConfig({
    isDatasetImage: isImage,
    isDatasetAudio: isAudio,
    datasetCheckFailed: false,
  });
}

function openManualMapping(
  attempt: FreshTrainingStartAttempt,
  check: CheckFormatResponse,
  isVlm: boolean,
  isAudio: boolean,
): false {
  const hint = buildManualMappingHint(attempt.config, check, isVlm, isAudio);
  if (
    Object.keys(hint).length > 0 &&
    !attempt.updateConfig({
      datasetManualMapping: hint,
    })
  ) {
    return false;
  }
  attempt.cancel();
  useDatasetPreviewDialogStore.getState().openMapping(check);
  return false;
}

async function confirmSelectedModelRemoteCode(
  attempt: FreshTrainingStartAttempt,
  hfToken: string | null,
): Promise<boolean> {
  const modelName = attempt.config.selectedModel;
  if (!modelName) {
    return true;
  }

  const preferLocalCache = attempt.config.modelKnownCached;
  let approvalApplied = true;
  const approved = await confirmRemoteCodeIfNeeded({
    modelName,
    hfToken,
    preferLocalCache,
    modelLocalPath: preferLocalCache ? attempt.config.modelLocalPath : null,
    requiresTrustRemoteCode: attempt.config.trustRemoteCode,
    onApprove: (fingerprint) => {
      approvalApplied = attempt.updateConfig({
        trustRemoteCode: true,
        approvedRemoteCodeFingerprint: fingerprint,
      });
    },
  });
  if (!approvalApplied || attempt.abortIfInputsChanged()) {
    return false;
  }
  return approved || attempt.cancel();
}

async function submitFreshTrainingRun(
  attempt: FreshTrainingStartAttempt,
  hfToken: string | null,
): Promise<boolean> {
  const validation = validateTrainingConfig(
    attempt.config,
    usePlatformStore.getState().deviceType,
  );
  if (!validation.ok) {
    return attempt.cancel(translate(validation.errorKey));
  }

  const payload = buildTrainingStartPayload(attempt.config, hfToken);
  if (!attempt.enterTransport()) {
    return false;
  }

  useTrainingRuntimeStore
    .getState()
    .setStartResources(
      payload.model_name,
      payload.hf_dataset,
      false,
      payload.project_name ?? "",
    );
  const response = await startTraining(payload, attempt.startRequestId);
  if (response.status === "error") {
    throw new TrainingStartError(
      response.error || response.message,
      response.error_code ?? null,
    );
  }

  return attempt.settleAccepted(response.job_id, response.message);
}

async function checkSelectedDataset(
  attempt: FreshTrainingStartAttempt,
  datasetName: string,
  hfToken: string | null,
  isVlm: boolean,
): Promise<CheckFormatResponse | null> {
  const config = attempt.config;
  const preferLocalCache =
    config.datasetSource === "huggingface" &&
    config.datasetKnownCached &&
    !config.datasetStreaming;
  const datasetLocalPath = preferLocalCache ? config.datasetLocalPath : null;
  const requestedCacheIdentity = preferLocalCache
    ? createDatasetCacheUsabilityIdentity({
        dataset: datasetName,
        cachePath: datasetLocalPath,
        subset: config.datasetSubset,
        split: config.datasetSplit,
        streaming: config.datasetStreaming,
      })
    : null;
  const requestedCacheValidation = requestedCacheIdentity
    ? trainingDatasetCacheRejections.beginValidation(requestedCacheIdentity)
    : null;

  try {
    const check = await checkDatasetFormat({
      datasetName,
      hfToken,
      subset: config.datasetSubset,
      split: config.datasetSplit,
      isVlm,
      preferLocalCache,
      localPath: datasetLocalPath,
    });
    return attempt.abortIfInputsChanged() ? null : check;
  } catch (error) {
    if (attempt.abortIfInputsChanged()) {
      return null;
    }
    if (
      error instanceof DatasetFormatError &&
      error.status === 404 &&
      clearDeletedDataset(datasetName)
    ) {
      attempt.cancel(error.message);
      return null;
    }
    if (!(preferLocalCache && isMissingLocalDatasetCacheError(error))) {
      throw error;
    }
  }

  if (
    requestedCacheValidation &&
    !trainingDatasetCacheRejections.rejectValidation(requestedCacheValidation)
  ) {
    attempt.cancel(TRAINING_SETUP_CHANGED_ERROR);
    return null;
  }

  if (
    !attempt.updateConfig(
      {
        datasetKnownCached: false,
        datasetLocalPath: null,
        browseDatasetSelection: {
          source: "huggingface",
          dataset: datasetName,
          knownCached: false,
          localPath: null,
        },
      },
      () => {
        clearMissingDatasetCacheReference(datasetName, datasetLocalPath);
      },
    )
  ) {
    return null;
  }

  const fallbackConfig = attempt.config;
  const check = await checkDatasetFormat({
    datasetName,
    hfToken,
    subset: fallbackConfig.datasetSubset,
    split: fallbackConfig.datasetSplit,
    isVlm,
    preferLocalCache: false,
    localPath: null,
  });
  return attempt.abortIfInputsChanged() ? null : check;
}

function needsManualMapping(
  config: TrainingConfigState,
  check: CheckFormatResponse,
  isVlm: boolean,
  isAudio: boolean,
): boolean {
  return (
    !isRawTextDatasetFormat(config.datasetFormat) &&
    (check.requires_manual_mapping ||
      check.detected_format === "custom_heuristic") &&
    !hasManualMapping(config, isVlm, isAudio)
  );
}

function buildManualMappingHint(
  config: TrainingConfigState,
  check: CheckFormatResponse,
  isVlm: boolean,
  isAudio: boolean,
): Record<string, string> {
  if (check.suggested_mapping) {
    return remapSuggestedColumns(config, check.suggested_mapping);
  }
  if (isAudio) {
    return detectedAudioMapping(check);
  }
  if (isVlm) {
    return detectedVisionMapping(check);
  }
  return {};
}

function remapSuggestedColumns(
  config: TrainingConfigState,
  suggestedMapping: Record<string, string>,
): Record<string, string> {
  const hint: Record<string, string> = {};
  const table = ROLE_REMAP[config.datasetFormat];
  for (const [column, role] of Object.entries(suggestedMapping)) {
    hint[column] = table ? (table[role] ?? role) : role;
  }
  return hint;
}

function detectedAudioMapping(
  check: CheckFormatResponse,
): Record<string, string> {
  const hint: Record<string, string> = {};
  if (check.detected_audio_column) {
    hint[check.detected_audio_column] = "audio";
  }
  if (check.detected_text_column) {
    hint[check.detected_text_column] = "text";
  }
  if (check.detected_speaker_column) {
    hint[check.detected_speaker_column] = "speaker_id";
  }
  return hint;
}

function detectedVisionMapping(
  check: CheckFormatResponse,
): Record<string, string> {
  const hint: Record<string, string> = {};
  if (check.detected_image_column) {
    hint[check.detected_image_column] = "image";
  }
  if (check.detected_text_column) {
    hint[check.detected_text_column] = "text";
  }
  return hint;
}

function clearMissingDatasetCacheReference(
  datasetName: string,
  datasetLocalPath: string | null,
): void {
  const current = useTrainingConfigStore.getState();
  if (
    current.datasetSource !== "huggingface" ||
    current.dataset !== datasetName ||
    !current.datasetKnownCached ||
    !cacheLocalPathMatchesSelection(current.datasetLocalPath, datasetLocalPath)
  ) {
    return;
  }
  useTrainingConfigStore.setState({
    datasetKnownCached: false,
    datasetLocalPath: null,
    browseDatasetSelection: {
      source: "huggingface",
      dataset: datasetName,
      knownCached: false,
      localPath: null,
    },
  });
}

function getDatasetName(config: TrainingConfigState): string | null {
  return config.datasetSource === "huggingface"
    ? config.dataset
    : config.uploadedFile;
}

function getHfDatasetName(config: TrainingConfigState): string | null {
  return config.datasetSource === "huggingface" ? config.dataset : null;
}

function hasManualMapping(
  config: TrainingConfigState,
  isVlm: boolean,
  isAudio: boolean,
): boolean {
  const roles = new Set(Object.values(config.datasetManualMapping));
  if (isAudio) {
    return roles.has("audio") && roles.has("text");
  }
  if (isVlm) {
    return roles.has("image") && roles.has("text");
  }
  if (config.datasetFormat === "alpaca") {
    return roles.has("instruction") && roles.has("output");
  }
  if (config.datasetFormat === "sharegpt") {
    return roles.has("human") && roles.has("gpt");
  }
  return roles.has("user") && roles.has("assistant");
}
