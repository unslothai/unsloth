// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LLAMA_JOB_STARTED_STORAGE_KEY = "unsloth_llama_job_started_at";
const LLAMA_JOB_STARTED_EVENT = "unsloth:llama-job-started";

/** Prompt same-window and cross-tab listeners to fetch the shared job state. */
export function signalLlamaJobStarted(startedAt: string | null): void {
  try {
    localStorage.setItem(
      LLAMA_JOB_STARTED_STORAGE_KEY,
      `${startedAt ?? "unknown"}:${Date.now()}`,
    );
  } catch {
    // Storage can be unavailable in restricted browser contexts.
  }
  window.dispatchEvent(new Event(LLAMA_JOB_STARTED_EVENT));
}

/** Signal only when an API response carries the authoritative running job. */
export function signalRunningLlamaJob(job: {
  state: unknown;
  started_at: string | null;
}): boolean {
  if (job.state !== "running") {
    return false;
  }
  signalLlamaJobStarted(job.started_at);
  return true;
}

/** Subscribe to same-window and cross-tab job-start notifications. */
export function subscribeToLlamaJobStarted(listener: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === LLAMA_JOB_STARTED_STORAGE_KEY) {
      listener();
    }
  };
  window.addEventListener(LLAMA_JOB_STARTED_EVENT, listener);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(LLAMA_JOB_STARTED_EVENT, listener);
    window.removeEventListener("storage", onStorage);
  };
}
