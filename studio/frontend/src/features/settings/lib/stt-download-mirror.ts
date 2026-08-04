// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  cancelSttDownload,
  fetchSttStatus,
  loadSttModel,
  sttEngineStatusFor,
} from "@/features/chat";
import {
  finishExternalJob,
  startExternalJob,
  updateExternalJob,
} from "@/features/hub";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  type SttModel,
  getSttModelRepo,
  sttModelName,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

/**
 * Shows a dictation model download in the shared download panel, and loads the
 * model once it lands. The STT sidecars own the transfer, so progress is
 * polled from their status rather than driven by the hub poll loop.
 */

const POLL_MS = 750;
// A download reports nothing for a moment while the worker starts. Without this
// the first poll would read "not downloading" and call it finished.
const START_GRACE_MS = 8_000;

let timer: number | null = null;
let trackedModel: SttModel | null = null;

function jobKey(model: SttModel): string {
  return `stt:${model}`;
}

function stop(): void {
  if (timer !== null) window.clearInterval(timer);
  timer = null;
  trackedModel = null;
}

async function loadAndAnnounce(model: SttModel): Promise<void> {
  try {
    await loadSttModel(model);
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
): void {
  finishExternalJob(jobKey(model), outcome, error);
  stop();
  // Only warm what the user is still pointed at. Selecting another model, or
  // leaving local dictation, during the download means this one is not wanted
  // and loading it would undo the unload that switch performed.
  const { sttModel, dictationEngine } = useVoiceSettingsStore.getState();
  if (
    outcome === "complete" &&
    dictationEngine === "model" &&
    sttModel === model
  ) {
    void loadAndAnnounce(model);
  }
}

async function poll(model: SttModel, startedAt: number): Promise<void> {
  let status: Awaited<ReturnType<typeof fetchSttStatus>>;
  try {
    status = await fetchSttStatus(undefined, model);
  } catch {
    // A dropped poll is not a failed download; the next one decides.
    return;
  }
  if (trackedModel !== model) return;

  const engine = sttEngineStatusFor(status, model);
  const download = engine?.download;

  if (download?.downloading && download.model === model) {
    updateExternalJob(jobKey(model), {
      downloadedBytes: download.bytes_done ?? 0,
      expectedBytes: download.bytes_total ?? 0,
    });
    return;
  }

  if (engine?.downloaded_models.includes(model)) {
    settle(model, "complete");
    return;
  }
  if (download?.cancelled) {
    settle(model, "cancelled");
    return;
  }
  if (download?.error) {
    settle(model, "error", download.error);
    return;
  }
  if (Date.now() - startedAt > START_GRACE_MS) {
    settle(
      model,
      "error",
      translate("settings.voice.dictation.sttDownloadFailed"),
    );
  }
}

/** Whether a download is already mirrored, so a poller can adopt one that
 * started before this page load without duplicating the row. */
export function isTrackingSttDownload(model: SttModel): boolean {
  return trackedModel === model;
}

/**
 * Mirror an already-started download of `model` into the panel. Replaces any
 * previous tracking, since only one STT download runs at a time.
 */
export function trackSttDownload(model: SttModel): void {
  if (trackedModel) finishExternalJob(jobKey(trackedModel), "cancelled");
  stop();
  trackedModel = model;
  startExternalJob({
    key: jobKey(model),
    repoId: getSttModelRepo(model),
    variant: sttModelName(model),
    expectedBytes: 0,
    cancel: async () => {
      try {
        await cancelSttDownload(model);
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
  const startedAt = Date.now();
  timer = window.setInterval(() => {
    void poll(model, startedAt);
  }, POLL_MS);
  void poll(model, startedAt);
}
