// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type CapturedCall = {
  jobId: string;
  options: Record<string, unknown>;
};

export const trackerCalls = {
  analysis: [] as CapturedCall[],
  dataset: [] as CapturedCall[],
  status: [] as CapturedCall[],
  stream: [] as CapturedCall[],
};

export function resetTrackerCalls(): void {
  for (const calls of Object.values(trackerCalls)) {
    calls.length = 0;
  }
}

export class RecipeApiError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(`Recipe API request failed with status ${status}`);
    this.status = status;
  }
}

export function streamRecipeJobEvents(
  options: Record<string, unknown> & { jobId: string },
): Promise<void> {
  trackerCalls.stream.push({ jobId: options.jobId, options });
  return Promise.resolve();
}

export function getRecipeJobStatus(
  jobId: string,
  options: Record<string, unknown>,
): Promise<{ status: string }> {
  trackerCalls.status.push({ jobId, options });
  return Promise.resolve({ status: "completed" });
}

export function getRecipeJobAnalysis(
  jobId: string,
  options: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  trackerCalls.analysis.push({ jobId, options });
  return Promise.resolve({});
}

export function getRecipeJobDataset(
  jobId: string,
  options: Record<string, unknown>,
): Promise<{ dataset: Record<string, unknown>[]; total: number }> {
  trackerCalls.dataset.push({ jobId, options });
  return Promise.resolve({ dataset: [], total: 0 });
}

export function toastError(): void {
  // Notifications are outside this forwarding contract.
}

export function toastSuccess(): void {
  // Notifications are outside this forwarding contract.
}
