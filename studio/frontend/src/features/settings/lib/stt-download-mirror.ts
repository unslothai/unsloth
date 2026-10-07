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
import { SttDownloadTrackers } from "./stt-download-trackers";

/** The STT sidecars own the transfer, so progress is polled from their status. */

const POLL_MS = 750;
// A starting worker reports nothing at first; without this grace it would read as finished.
const START_GRACE_MS = 8_000;

const trackers = new SttDownloadTrackers();
const warmSelectedVoiceModelOnComplete = new Map<string, boolean>();
// The quant each tracked download fetches, when its starter knew it.
const trackedVariants = new Map<string, string | null>();

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
  const key = trackerKey(model, engine);
  if (!trackers.has(key)) return;

  const engineStatus = sttEngineStatusFor(status, model, engine);
  const download = engineStatus?.download;

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
  if (Date.now() - startedAt > START_GRACE_MS) {
    settle(
      model,
      "error",
      translate("settings.voice.dictation.sttDownloadFailed"),
      engine,
    );
  }
}

/** Lets a poller adopt a download started before this page load. */
export function isTrackingSttDownload(
  model: SttModel,
  engine?: SttEngine,
): boolean {
  return trackers.has(trackerKey(model, engine ?? sttEngineFor(model)));
}

/** Other models' downloads keep their rows: switching models does not stop them. */
export function trackSttDownload(
  model: SttModel,
  options: {
    warmSelectedVoiceModelOnComplete?: boolean;
    engine?: SttEngine;
    repoId?: string;
    /** The quant this download fetches, so a quant picked meanwhile is not warmed. */
    ggufVariant?: string | null;
  } = {},
): void {
  const resolvedEngine = options.engine ?? sttEngineFor(model);
  const key = trackerKey(model, resolvedEngine);
  // Adopting the same transfer must not reset its progress or replace its poller.
  if (trackers.has(key)) {
    if (options.warmSelectedVoiceModelOnComplete !== false)
      warmSelectedVoiceModelOnComplete.set(key, true);
    return;
  }
  warmSelectedVoiceModelOnComplete.set(
    key,
    options.warmSelectedVoiceModelOnComplete ?? true,
  );
  if (options.ggufVariant !== undefined) {
    trackedVariants.set(key, options.ggufVariant);
  }
  startExternalJob({
    key: jobKey(model, resolvedEngine),
    repoId: options.repoId ?? getSttModelRepo(model),
    variant: sttModelName(model),
    expectedBytes: 0,
    cancel: async () => {
      try {
        await cancelSttDownload(model, resolvedEngine);
      } catch (error) {
        toast.error(
          translate("settings.voice.dictation.sttCancelDownloadFailed"),
          { description: error instanceof Error ? error.message : undefined },
        );
        // The row already shows "cancelling" and progress never resets state, so restore it.
        throw error;
      }
    },
  });
  const startedAt = Date.now();
  const timer = window.setInterval(() => {
    void poll(model, startedAt, resolvedEngine);
  }, POLL_MS);
  trackers.start(key, () => window.clearInterval(timer));
  void poll(model, startedAt, resolvedEngine);
}
