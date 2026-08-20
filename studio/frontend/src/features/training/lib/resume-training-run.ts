


import { prepareHfTokenForUse } from "@/features/hf-auth";
import { getHfToken, useHfTokenStore } from "@/features/hub";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { type TranslationKey, translate } from "@/i18n";
import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { toast } from "@/lib/toast";
import { getTrainingRun } from "../api/history-api";
import {
  TrainingStartError,
  isTrainingStartOutcomeUnknownError,
  startTraining,
} from "../api/train-api";
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
  settleUnconfirmedTrainingStart,
  tryAcquireTrainingStart,
} from "./training-start-runtime";
import { confirmTrainingTransformersUpgrade } from "./training-transformers-upgrade";

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

    // Upgrade consent first, then the custom-code gate, like a fresh start: a resume
    // after a reinstall can face the same unknown architecture the first run installed for.
    // The upgrade check already read this model's config, so carry its custom-code
    // verdict into the next gate rather than let that gate's fallback re-derive it.
    const upgradeVerdict = { requiresTrustRemoteCode: false };
    if (
      !(await confirmResumeTransformersUpgrade(
        runId,
        payload,
        attempt,
        upgradeVerdict,
      ))
    ) {
      return false;
    }
    if (
      !(await confirmResumeRemoteCode(
        payload,
        attempt,
        upgradeVerdict.requiresTrustRemoteCode,
      ))
    ) {
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

  get startRequestId(): string {
    return this.lease.startRequestId;
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

  settleUnconfirmed(message: string): boolean {
    this.phase = "finished";
    return settleUnconfirmedTrainingStart(this.lease, message);
  }

  async fail(error: unknown): Promise<boolean> {
    if (this.phase === "finished") {
      return false;
    }
    let failure = error;
    if (
      this.phase === "transport" &&
      isTrainingStartOutcomeUnknownError(failure)
    ) {
      const recovery = await reconcileTrainingStartTransportFailure(this.lease);
      if (recovery.kind === "recovered") {
        this.phase = "finished";
        return true;
      }
      if (recovery.kind === "rejected") {
        failure = new TrainingStartError(recovery.error, recovery.errorCode);
      } else {
        const message = translate("studio.training.startUnconfirmed");
        toast.warning(message);
        return this.settleUnconfirmed(message);
      }
    }
    if (!isTrainingStartLeaseActive(this.lease)) {
      return false;
    }
    if (this.phase === "preflight" && !this.isPreflightActive()) {
      return false;
    }
    const safeMessage = normalizeTrainingStartError(
      failure instanceof Error
        ? failure
        : translate("studio.training.resumeFailed"),
    );
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
    // The server explains a provenance refusal precisely; blaming the checkpoint here would
    // contradict it, and this guard fires before any request, so its wording is all the user sees.
    throw new Error(
      detail.run.resume_blocked_reason || translate(RESUME_UNAVAILABLE_ERROR),
    );
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

async function confirmResumeTransformersUpgrade(
  runId: string,
  payload: TrainingStartRequest,
  attempt: ResumeTrainingStartAttempt,
  verdict: { requiresTrustRemoteCode: boolean },
): Promise<boolean> {
  if (!payload.model_name) {
    return attempt.isPreflightActive();
  }
  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: payload.model_name,
    hfToken: payload.hf_token ?? null,
    // Same pin the custom-code gate below resolves, so both read one config.json.
    modelCachePin: resumeModelCachePin(payload),
    // The stored run decides whether an install is even offerable: this checkpoint may
    // be attested against a 4-bit load the latest sidecar permanently refuses.
    resumeRunId: runId,
  });
  verdict.requiresTrustRemoteCode = outcome.requiresTrustRemoteCode;
  if (!attempt.isPreflightActive()) {
    return false;
  }
  return outcome.proceed || attempt.cancel(outcome.error);
}

/** The copy of the model this resume loads, for every gate that has to inspect it.
 *
 * One resolution, so the upgrade check and the custom-code scan can never end up
 * reading different config.json files for the same start. */
function resumeModelCachePin(payload: TrainingStartRequest) {
  return resolveResumeRemoteCodeCache({
    actualModelRepoId: payload.actual_model_repo_id,
    modelKnownCached: payload.model_known_cached,
    modelLocalPath: payload.model_local_path,
    modelSnapshotPath: payload.model_snapshot_path,
  });
}

async function confirmResumeRemoteCode(
  payload: TrainingStartRequest,
  attempt: ResumeTrainingStartAttempt,
  upgradeRequiresTrustRemoteCode = false,
): Promise<boolean> {
  if (!payload.model_name) {
    return attempt.isPreflightActive();
  }

  let trustRemoteCode = Boolean(payload.trust_remote_code);
  let approvedRemoteCodeFingerprint =
    payload.approved_remote_code_fingerprint ?? null;
  const remoteCodeOk = await confirmRemoteCodeIfNeeded({
    modelName: payload.model_name,
    hfToken: payload.hf_token ?? null,
    ...resumeModelCachePin(payload),
    requiresTrustRemoteCode: trustRemoteCode || upgradeRequiresTrustRemoteCode,
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
  const response = await startTraining(payload, attempt.startRequestId);
  if (response.status === "error") {
    throw new TrainingStartError(
      response.error || response.message,
      response.error_code ?? null,
    );
  }
  return attempt.settleAccepted(response.job_id, response.message);
}
