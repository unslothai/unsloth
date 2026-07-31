// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { prepareHfTokenForUse } from "@/features/hf-auth";
import { getHfToken, useHfTokenStore } from "@/features/hub";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { type TranslationKey, translate } from "@/i18n";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { getTrainingRun } from "../api/history-api";
import { startTraining } from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingStartRequest } from "../types/api";
import { resolveResumeRemoteCodeCache } from "./resume-remote-code-cache";
import { normalizeTrainingStartError } from "./training-start-errors";
import {
  TRAINING_SETUP_CHANGED_ERROR,
  type TrainingStartLease,
  isTrainingStartLeaseActive,
  reconcileTrainingStartTransportFailure,
  releaseTrainingStart,
  settleAcceptedTrainingStart,
  tryAcquireTrainingStart,
} from "./training-start-runtime";

const RESUME_UNAVAILABLE_ERROR =
  "studio.training.resumeUnavailable" satisfies TranslationKey;

export async function resumeTrainingRun(runId: string): Promise<boolean> {
  const attempt = ResumeTrainingStartAttempt.begin();
  if (!attempt) {
    return false;
  }

  try {
    const payload = await loadResumePayload(runId, attempt);
    if (!payload) {
      return false;
    }
    if (!(await prepareResumeHfToken(payload, attempt))) {
      return false;
    }

    useTrainingRuntimeStore
      .getState()
      .setStartResources(
        payload.model_name,
        payload.hf_dataset,
        true,
        payload.project_name ?? "",
      );

    if (!(await confirmResumeRemoteCode(payload, attempt))) {
      return false;
    }
    return await submitResumeTrainingRun(payload, attempt);
  } catch (error) {
    return attempt.fail(error);
  }
}

type ResumeAttemptPhase = "preflight" | "transport" | "finished";

class ResumeTrainingStartAttempt {
  private phase: ResumeAttemptPhase = "preflight";
  private expectedHfToken: string;
  private readonly lease: TrainingStartLease;

  private constructor(lease: TrainingStartLease) {
    this.lease = lease;
    this.expectedHfToken = getHfToken();
  }

  static begin(): ResumeTrainingStartAttempt | null {
    const lease = tryAcquireTrainingStart();
    if (!lease) {
      return null;
    }
    const runtime = useTrainingRuntimeStore.getState();
    runtime.setStartError(null);
    runtime.setStartResources(null, null, true, null);
    return new ResumeTrainingStartAttempt(lease);
  }

  get hfToken(): string {
    return this.expectedHfToken;
  }

  isPreflightActive(): boolean {
    if (this.phase !== "preflight") {
      return false;
    }
    if (!isTrainingStartLeaseActive(this.lease)) {
      this.phase = "finished";
      return false;
    }
    if (getHfToken() !== this.expectedHfToken) {
      return this.cancel(TRAINING_SETUP_CHANGED_ERROR);
    }
    return true;
  }

  acceptPreparedHfToken(token: string | null): boolean {
    if (this.phase !== "preflight" || !isTrainingStartLeaseActive(this.lease)) {
      this.phase = "finished";
      return false;
    }
    const currentToken = getHfToken();
    const nextToken = token ?? "";
    if (currentToken !== this.expectedHfToken && currentToken !== nextToken) {
      return this.cancel(TRAINING_SETUP_CHANGED_ERROR);
    }
    if (currentToken !== nextToken) {
      useHfTokenStore.getState().setToken(nextToken);
    }
    this.expectedHfToken = getHfToken();
    return true;
  }

  enterTransport(): boolean {
    if (!this.isPreflightActive()) {
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

  async fail(error: unknown): Promise<boolean> {
    if (this.phase === "finished") {
      return false;
    }
    if (
      this.phase === "transport" &&
      (await reconcileTrainingStartTransportFailure(this.lease))
    ) {
      this.phase = "finished";
      return true;
    }
    if (!isTrainingStartLeaseActive(this.lease)) {
      return false;
    }
    if (this.phase === "preflight" && !this.isPreflightActive()) {
      return false;
    }
    const rawMessage =
      error instanceof Error
        ? error.message
        : translate("studio.training.resumeFailed");
    const safeMessage = normalizeTrainingStartError(rawMessage);
    this.cancel(safeMessage);
    toast.error(translate("studio.training.resumeFailedTitle"), {
      description: safeMessage,
    });
    return false;
  }
}

async function loadResumePayload(
  runId: string,
  attempt: ResumeTrainingStartAttempt,
): Promise<TrainingStartRequest | null> {
  const detail = await getTrainingRun(runId);
  if (!attempt.isPreflightActive()) {
    return null;
  }

  const outputDir = detail.run.output_dir;
  if (!(detail.run.can_resume && outputDir)) {
    throw new Error(translate(RESUME_UNAVAILABLE_ERROR));
  }
  primeNativeNotificationPermission().catch(() => undefined);

  const payload = {
    ...(detail.config as Partial<TrainingStartRequest>),
  } as TrainingStartRequest;
  payload.hf_token = attempt.hfToken || null;
  payload.wandb_token = null;
  payload.resume_from_checkpoint = outputDir;
  return payload;
}

async function prepareResumeHfToken(
  payload: TrainingStartRequest,
  attempt: ResumeTrainingStartAttempt,
): Promise<boolean> {
  const preparedToken = await prepareHfTokenForUse(payload.hf_token);
  if (!attempt.acceptPreparedHfToken(preparedToken.token)) {
    return false;
  }
  if (!preparedToken.proceed) {
    return attempt.cancel();
  }
  payload.hf_token = preparedToken.token;
  return true;
}

async function confirmResumeRemoteCode(
  payload: TrainingStartRequest,
  attempt: ResumeTrainingStartAttempt,
): Promise<boolean> {
  if (!payload.model_name) {
    return attempt.isPreflightActive();
  }

  let trustRemoteCode = Boolean(payload.trust_remote_code);
  let approvedRemoteCodeFingerprint =
    payload.approved_remote_code_fingerprint ?? null;
  const remoteCodeCache = resolveResumeRemoteCodeCache({
    actualModelRepoId: payload.actual_model_repo_id,
    modelKnownCached: payload.model_known_cached,
    modelLocalPath: payload.model_local_path,
    modelSnapshotPath: payload.model_snapshot_path,
  });
  const remoteCodeOk = await confirmRemoteCodeIfNeeded({
    modelName: payload.model_name,
    hfToken: payload.hf_token ?? null,
    ...remoteCodeCache,
    requiresTrustRemoteCode: trustRemoteCode,
    onApprove: (fingerprint) => {
      trustRemoteCode = true;
      approvedRemoteCodeFingerprint = fingerprint;
    },
  });

  if (!attempt.isPreflightActive()) {
    return false;
  }
  if (!remoteCodeOk) {
    return attempt.cancel();
  }
  payload.trust_remote_code = trustRemoteCode;
  payload.approved_remote_code_fingerprint = approvedRemoteCodeFingerprint;
  return true;
}

async function submitResumeTrainingRun(
  payload: TrainingStartRequest,
  attempt: ResumeTrainingStartAttempt,
): Promise<boolean> {
  if (!attempt.enterTransport()) {
    return false;
  }
  const response = await startTraining(payload);
  if (response.status === "error") {
    throw new Error(response.error || response.message);
  }
  return attempt.settleAccepted(response.job_id, response.message);
}
