// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAudioCppFolderId } from "@/features/audio/audio-cpp-catalog";
import {
  type SttEngine,
  cancelSttDownload,
  fetchSttStatus,
  loadSttModel,
  sttEngineFor,
  sttEngineStatusFor,
} from "@/features/chat";
import {
  finishExternalJob,
  invalidateGgufVariantsCache,
  startExternalJob,
  updateExternalJob,
} from "@/features/hub";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  type SttModel,
  getSttModelRepo,
  sttModelName,
  sttModelVariant,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";
import {
  shouldRecheckSttReplacement,
  SttDownloadTrackers,
  sttReplacementAction,
} from "./stt-download-trackers";

/**
 * Shows a dictation model download in the shared download panel, and loads the
 * model once it lands. The STT sidecars own the transfer, so progress is
 * polled from their status rather than driven by the hub poll loop.
 */

const POLL_MS = 750;
// A download reports nothing for a moment while the worker starts. Without this
// the first poll would read "not downloading" and call it finished.
const START_GRACE_MS = 8_000;

const trackers = new SttDownloadTrackers();
const warmSelectedVoiceModelOnComplete = new Map<string, boolean>();
// The quant each tracked download fetches, when its starter knew it.
const trackedVariants = new Map<string, string | null>();
const trackedDownloadIds = new Map<string, string | null>();
const trackedStartedAt = new Map<string, number>();
const replacementChecks = new Map<string, string>();

interface TrackSttDownloadOptions {
  warmSelectedVoiceModelOnComplete?: boolean;
  engine?: SttEngine;
  repoId?: string;
  /** The quant this download fetches, so a quant picked meanwhile is not warmed. */
  ggufVariant?: string | null;
  /** The backend attempt this row may cancel. */
  downloadId?: string | null;
}

function trackerKey(model: SttModel, engine?: SttEngine): string {
  return engine && engine !== "transformers" ? `${engine}:${model}` : model;
}

function jobKey(model: SttModel, engine?: SttEngine): string {
  return engine && engine !== "transformers"
    ? `stt:${engine}:${model}`
    : `stt:${model}`;
}

async function loadAndAnnounce(
  model: SttModel,
  engine?: SttEngine,
  ggufVariant?: string | null,
): Promise<void> {
  try {
    await loadSttModel(model, engine, undefined, undefined, ggufVariant);
    toast.success(
      translate("settings.voice.dictation.sttModelReady", {
        model: sttModelName(model),
      }),
    );
  } catch (error) {
    toast.error(translate("settings.voice.dictation.sttModelFailed"), {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}

function settle(
  model: SttModel,
  outcome: "complete" | "cancelled" | "error",
  error?: string | null,
  engine?: SttEngine,
): void {
  const key = trackerKey(model, engine);
  finishExternalJob(jobKey(model, engine), outcome, error);
  // A cached listing would still call the new quant not downloaded.
  if (outcome === "complete" && isAudioCppFolderId(model)) {
    invalidateGgufVariantsCache(model);
  }
  trackers.stop(key);
  const shouldWarmVoiceModel =
    warmSelectedVoiceModelOnComplete.get(key) ?? true;
  warmSelectedVoiceModelOnComplete.delete(key);
  const tracked = trackedVariants.get(key);
  trackedVariants.delete(key);
  trackedDownloadIds.delete(key);
  trackedStartedAt.delete(key);
  // Only warm what the user is still pointed at. Selecting another model or
  // quant, or leaving local dictation, during the download means this one is
  // not wanted and loading it would undo the unload that switch performed.
  const { sttModel, sttGgufVariant, dictationEngine } =
    useVoiceSettingsStore.getState();
  const variant = sttModelVariant(model, sttGgufVariant);
  if (
    shouldWarmVoiceModel &&
    outcome === "complete" &&
    dictationEngine === "model" &&
    sttModel === model &&
    // An adopted download's quant is unknown; it can only be what an unpinned row runs.
    (tracked === undefined ? variant === null : tracked === variant)
  ) {
    void loadAndAnnounce(model, engine, variant);
  }
}

async function poll(
  model: SttModel,
  startedAt: number,
  engine?: SttEngine,
): Promise<void> {
  const key = trackerKey(model, engine);
  const requestedDownloadId = trackedDownloadIds.get(key);
  let status: Awaited<ReturnType<typeof fetchSttStatus>>;
  try {
    status = await fetchSttStatus(
      undefined,
      engine === undefined || engine === "transformers" ? model : undefined,
    );
  } catch {
    // A dropped poll is not a failed download; the next one decides.
    return;
  }
  if (!trackers.has(key)) return;
  if (trackedDownloadIds.get(key) !== requestedDownloadId) return;

  const engineStatus = sttEngineStatusFor(status, model, engine);
  const download = engineStatus?.download;
  if (
    requestedDownloadId &&
    download?.download_id &&
    requestedDownloadId !== download.download_id
  ) {
    // A shared engine can move to another transfer before this row polls. The
    // attempt history is authoritative: downloaded_models is only row-level
    // and could describe an older audio.cpp quant.
    if (download.completed_download_ids?.includes(requestedDownloadId)) {
      settle(model, "complete", undefined, engine);
      return;
    }
    const elapsed = Date.now() - (trackedStartedAt.get(key) ?? startedAt);
    if (elapsed > START_GRACE_MS) {
      settle(
        model,
        "error",
        translate("settings.voice.dictation.sttDownloadFailed"),
        engine,
      );
    }
    return;
  }

  if (download?.downloading && download.model === model) {
    updateExternalJob(jobKey(model, engine), {
      downloadedBytes: download.bytes_done ?? 0,
      expectedBytes: download.bytes_total ?? 0,
    });
    return;
  }

  // Before the downloaded check: a row lists as downloaded once any quant of it is cached, so a
  // stopped download of another quant would read as complete. start() clears both flags.
  if (download?.cancelled && (download.cancelled_model ?? model) === model) {
    settle(model, "cancelled", undefined, engine);
    return;
  }
  if (download?.error) {
    settle(model, "error", download.error, engine);
    return;
  }
  if (engineStatus?.downloaded_models.includes(model)) {
    settle(model, "complete", undefined, engine);
    return;
  }
  if (Date.now() - (trackedStartedAt.get(key) ?? startedAt) > START_GRACE_MS) {
    settle(
      model,
      "error",
      translate("settings.voice.dictation.sttDownloadFailed"),
      engine,
    );
  }
}

/**
 * Mirror an already-started download of `model` into the panel. Any other
 * model's download keeps its own row: switching models does not stop it.
 */
function trackSttDownloadNow(
  model: SttModel,
  options: TrackSttDownloadOptions,
): void {
  const resolvedEngine = options.engine ?? sttEngineFor(model);
  const key = trackerKey(model, resolvedEngine);
  const wasTracking = trackers.has(key);
  const previousDownloadId = trackedDownloadIds.get(key);
  const changedAttempt =
    options.downloadId !== undefined &&
    previousDownloadId !== options.downloadId;
  if (options.ggufVariant !== undefined) {
    trackedVariants.set(key, options.ggufVariant);
  }
  if (options.downloadId !== undefined) {
    if (trackedDownloadIds.get(key) !== options.downloadId) {
      trackedStartedAt.set(key, Date.now());
    }
    trackedDownloadIds.set(key, options.downloadId);
  }
  // Another owner of the same attempt keeps its progress and poller. A newer
  // attempt reuses the poller but refreshes the row and cancellation identity.
  if (wasTracking && !changedAttempt) {
    if (options.warmSelectedVoiceModelOnComplete !== false)
      warmSelectedVoiceModelOnComplete.set(key, true);
    return;
  }
  if (wasTracking && !changedAttempt) {
    if (options.warmSelectedVoiceModelOnComplete !== false)
      warmSelectedVoiceModelOnComplete.set(key, true);
  } else {
    warmSelectedVoiceModelOnComplete.set(
      key,
      options.warmSelectedVoiceModelOnComplete ?? true,
    );
  }
  if (!trackedStartedAt.has(key)) trackedStartedAt.set(key, Date.now());
  startExternalJob({
    key: jobKey(model, resolvedEngine),
    repoId: options.repoId ?? getSttModelRepo(model),
    variant: sttModelName(model),
    expectedBytes: 0,
    cancel: async () => {
      try {
        await cancelSttDownload(
          model,
          resolvedEngine,
          trackedVariants.get(key),
          trackedDownloadIds.get(key),
        );
      } catch (error) {
        toast.error(
          translate("settings.voice.dictation.sttCancelDownloadFailed"),
          { description: error instanceof Error ? error.message : undefined },
        );
        // The row is already showing "cancelling" and progress updates never
        // reset state, so put it back or it stays there for the whole transfer.
        throw error;
      }
    },
  });
  if (wasTracking) return;
  const startedAt = Date.now();
  const timer = window.setInterval(() => {
    void poll(model, startedAt, resolvedEngine);
  }, POLL_MS);
  trackers.start(key, () => window.clearInterval(timer));
  void poll(model, startedAt, resolvedEngine);
}

async function confirmSttDownloadReplacement(
  model: SttModel,
  options: TrackSttDownloadOptions,
  resolvedEngine: SttEngine,
  key: string,
  previousDownloadId: string,
): Promise<void> {
  const candidate = options.downloadId;
  if (!candidate || replacementChecks.get(key) === candidate) return;
  replacementChecks.set(key, candidate);
  let retry = false;
  try {
    const status = await fetchSttStatus(
      undefined,
      resolvedEngine === "transformers" ? model : undefined,
    );
    const download = sttEngineStatusFor(status, model, resolvedEngine)?.download;
    const candidateCompleted =
      download?.completed_download_ids?.includes(candidate) ?? false;
    if (download?.download_id === candidate || candidateCompleted) {
      const current = trackedDownloadIds.get(key);
      const action = sttReplacementAction(
        trackers.has(key),
        current,
        previousDownloadId,
        candidate,
        download?.download_id,
        candidateCompleted,
      );
      if (action === "track") {
        trackSttDownloadNow(model, options);
      } else if (action === "retry") {
        retry = true;
      }
    }
  } catch {
    retry = true;
  } finally {
    if (replacementChecks.get(key) === candidate) replacementChecks.delete(key);
  }
  if (retry) {
    // Re-check after either a transient fetch failure or another candidate changing local state.
    window.setTimeout(() => {
      if (
        shouldRecheckSttReplacement(trackedDownloadIds.get(key), candidate)
      ) {
        void confirmSttDownloadReplacement(
          model,
          options,
          resolvedEngine,
          key,
          previousDownloadId,
        );
      }
    }, POLL_MS);
  }
}

export function trackSttDownload(
  model: SttModel,
  options: TrackSttDownloadOptions = {},
): void {
  const resolvedEngine = options.engine ?? sttEngineFor(model);
  const key = trackerKey(model, resolvedEngine);
  const previousDownloadId = trackedDownloadIds.get(key);
  if (
    trackers.has(key) &&
    previousDownloadId &&
    options.downloadId &&
    previousDownloadId !== options.downloadId
  ) {
    void confirmSttDownloadReplacement(
      model,
      options,
      resolvedEngine,
      key,
      previousDownloadId,
    );
    return;
  }
  trackSttDownloadNow(model, options);
}
