// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { createScopedSingleFlightRequest } from "../lib/single-flight-request";
import {
  type ParsedTrainingProgressEvent,
  consumeTrainingProgressStream,
} from "../lib/training-sse-stream";
import type {
  TrainingResetResponse,
  TrainingStartRequest,
  TrainingStartRequestStatusResponse,
  TrainingStartResponse,
  TrainingStopResponse,
} from "../types/api";
import type {
  TrainingMetricsResponse,
  TrainingStatusResponse,
} from "../types/runtime";

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "AbortError"
  );
}

const readError = (r: Response): Promise<string> => readFastApiError(r);
const TRAINING_START_TIMEOUT_MS = 60_000;
const TRAINING_STATUS_TIMEOUT_MS = 15_000;

function createAbortError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

async function runRequestWithTimeout<TResult>(
  timeoutMs: number,
  timeoutMessage: string,
  request: (signal: AbortSignal) => Promise<TResult>,
  parentSignal?: AbortSignal,
): Promise<TResult> {
  const controller = new AbortController();
  let abortReason = createAbortError(timeoutMessage);
  const abort = (reason: Error) => {
    if (controller.signal.aborted) {
      return;
    }
    abortReason = reason;
    controller.abort();
  };
  const onParentAbort = () => abort(createAbortError("Request was superseded"));
  if (parentSignal?.aborted) {
    onParentAbort();
    throw abortReason;
  }
  parentSignal?.addEventListener("abort", onParentAbort, { once: true });
  const timeoutId = setTimeout(
    () => abort(createAbortError(timeoutMessage)),
    timeoutMs,
  );
  let rejectAborted: ((reason: Error) => void) | null = null;
  const onRequestAbort = () => {
    rejectAborted?.(abortReason);
  };
  const aborted = new Promise<never>((_, reject) => {
    rejectAborted = reject;
    if (controller.signal.aborted) {
      reject(abortReason);
      return;
    }
    controller.signal.addEventListener("abort", onRequestAbort, { once: true });
  });

  try {
    const result = Promise.resolve().then(() => request(controller.signal));
    return await Promise.race([result, aborted]);
  } finally {
    clearTimeout(timeoutId);
    controller.signal.removeEventListener("abort", onRequestAbort);
    parentSignal?.removeEventListener("abort", onParentAbort);
  }
}

class TrainingStartOutcomeUnknownError extends Error {
  constructor(error: unknown) {
    super(
      error instanceof Error
        ? error.message
        : "Training start outcome is unknown.",
    );
    this.name = "TrainingStartOutcomeUnknownError";
  }
}

export class TrainingStartError extends Error {
  readonly errorCode: string | null;

  constructor(message: string, errorCode: string | null = null) {
    super(message);
    this.name = "TrainingStartError";
    this.errorCode = errorCode;
  }
}

export function isTrainingStartOutcomeUnknownError(error: unknown): boolean {
  return error instanceof TrainingStartOutcomeUnknownError;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

async function readTrainingStartError(
  response: Response,
): Promise<TrainingStartError> {
  const fallbackResponse = response.clone();
  try {
    const payload = (await response.json()) as { detail?: unknown };
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as { code?: unknown; message?: unknown };
      const message =
        typeof structured.message === "string" && structured.message
          ? structured.message
          : null;
      if (message) {
        return new TrainingStartError(
          message,
          typeof structured.code === "string" ? structured.code : null,
        );
      }
    }
  } catch {
    return new TrainingStartError(await readFastApiError(fallbackResponse));
  }
  return new TrainingStartError(await readFastApiError(fallbackResponse));
}

export async function startTraining(
  payload: TrainingStartRequest,
  startRequestId: string,
): Promise<TrainingStartResponse> {
  let result: TrainingStartResponse;
  try {
    result = await runRequestWithTimeout(
      TRAINING_START_TIMEOUT_MS,
      "Training start request timed out",
      async (signal) => {
        const response = await authFetch(
          "/api/train/start",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...payload,
              start_request_id: startRequestId,
            }),
            signal,
          },
          { retryNetworkErrors: false },
        );
        if (!response.ok) {
          throw await readTrainingStartError(response);
        }
        return (await response.json()) as TrainingStartResponse;
      },
    );
  } catch (error) {
    if (error instanceof TrainingStartError) {
      void acknowledgeTrainingStartRequest(startRequestId).catch(
        () => undefined,
      );
      throw error;
    }
    throw new TrainingStartOutcomeUnknownError(error);
  }
  if (result.status === "pending") {
    throw new TrainingStartOutcomeUnknownError(new Error(result.message));
  }
  if (result.status === "error") {
    void acknowledgeTrainingStartRequest(startRequestId).catch(() => undefined);
  }
  return result;
}

export async function getTrainingStartRequestStatus(
  startRequestId: string,
): Promise<TrainingStartRequestStatusResponse | null> {
  return runRequestWithTimeout(
    TRAINING_STATUS_TIMEOUT_MS,
    "Training start status request timed out",
    async (signal) => {
      const response = await authFetch(
        `/api/train/start-requests/${encodeURIComponent(startRequestId)}`,
        { signal },
        { retryNetworkErrors: false },
      );
      if (response.status === 404) {
        return null;
      }
      return parseJson<TrainingStartRequestStatusResponse>(response);
    },
  );
}

export async function acknowledgeTrainingStartRequest(
  startRequestId: string,
): Promise<void> {
  return runRequestWithTimeout(
    TRAINING_STATUS_TIMEOUT_MS,
    "Training start acknowledgement timed out",
    async (signal) => {
      const response = await authFetch(
        `/api/train/start-requests/${encodeURIComponent(startRequestId)}/acknowledge`,
        { method: "POST", signal },
        { retryNetworkErrors: false },
      );
      if (!response.ok) {
        throw new Error(await readError(response));
      }
    },
  );
}

export async function cancelTrainingStartRequest(
  startRequestId: string,
): Promise<TrainingStartRequestStatusResponse> {
  return runRequestWithTimeout(
    TRAINING_STATUS_TIMEOUT_MS,
    "Training start cancellation timed out",
    async (signal) => {
      const response = await authFetch(
        `/api/train/start-requests/${encodeURIComponent(startRequestId)}/cancel`,
        { method: "POST", signal },
        { retryNetworkErrors: false },
      );
      return parseJson<TrainingStartRequestStatusResponse>(response);
    },
  );
}

interface TrainingJobScope {
  expectedJobId?: string;
}

interface RequiredTrainingJobScope {
  expectedJobId: string;
}

function scopedTrainingBody(
  payload: Record<string, unknown>,
  scope?: TrainingJobScope,
): string {
  return JSON.stringify({
    ...payload,
    ...(scope?.expectedJobId !== undefined
      ? { expected_job_id: scope.expectedJobId }
      : {}),
  });
}

export async function stopTraining(
  save: boolean,
  scope: RequiredTrainingJobScope,
): Promise<TrainingStopResponse> {
  const response = await authFetch("/api/train/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: scopedTrainingBody({ save }, scope),
  });
  return parseJson<TrainingStopResponse>(response);
}

export async function resetTraining(
  scope: RequiredTrainingJobScope,
): Promise<TrainingResetResponse> {
  const response = await authFetch("/api/train/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: scopedTrainingBody({}, scope),
  });
  return parseJson<TrainingResetResponse>(response);
}

const requestTrainingStatus = createScopedSingleFlightRequest<
  undefined,
  TrainingStatusResponse
>(async (_scope, _input, signal) =>
  runRequestWithTimeout(
    TRAINING_STATUS_TIMEOUT_MS,
    "Training status request timed out",
    async (requestSignal) => {
      const response = await authFetch(
        "/api/train/status",
        { signal: requestSignal },
        { retryNetworkErrors: false },
      );
      return parseJson<TrainingStatusResponse>(response);
    },
    signal,
  ),
);

export function getTrainingStatus(
  requestKey: string,
  options?: { fresh?: boolean },
): Promise<TrainingStatusResponse> {
  return options?.fresh
    ? requestTrainingStatus.refresh(requestKey, undefined)
    : requestTrainingStatus.run(requestKey, undefined);
}

const requestTrainingMetrics = createScopedSingleFlightRequest<
  string,
  TrainingMetricsResponse
>(async (_scope, expectedJobId, signal) =>
  runRequestWithTimeout(
    TRAINING_STATUS_TIMEOUT_MS,
    "Training metrics request timed out",
    async (requestSignal) => {
      const jobId = encodeURIComponent(expectedJobId);
      const response = await authFetch(
        `/api/train/metrics?expected_job_id=${jobId}`,
        { signal: requestSignal },
        { retryNetworkErrors: false },
      );
      return parseJson<TrainingMetricsResponse>(response);
    },
    signal,
  ),
);

export function getTrainingMetrics(
  expectedJobId: string,
  requestKey: string,
  options?: { fresh?: boolean },
): Promise<TrainingMetricsResponse> {
  const scope = JSON.stringify([requestKey, expectedJobId]);
  return options?.fresh
    ? requestTrainingMetrics.refresh(scope, expectedJobId)
    : requestTrainingMetrics.run(scope, expectedJobId);
}

export async function streamTrainingProgress(options: {
  signal: AbortSignal;
  expectedJobId: string;
  lastEventId?: number | null;
  onOpen?: () => void;
  onEvent: (event: ParsedTrainingProgressEvent) => void;
}): Promise<void> {
  const headers = new Headers();
  if (typeof options.lastEventId === "number") {
    headers.set("Last-Event-ID", String(options.lastEventId));
  }

  const expectedJobId = encodeURIComponent(options.expectedJobId);
  const response = await authFetch(
    `/api/train/progress?expected_job_id=${expectedJobId}`,
    {
      method: "GET",
      headers,
      signal: options.signal,
    },
    { retryNetworkErrors: false },
  );

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  if (!response.body) {
    throw new Error("Progress stream unavailable");
  }

  if (options.signal.aborted) {
    return;
  }
  options.onOpen?.();
  await consumeTrainingProgressStream({
    body: response.body,
    signal: options.signal,
    onEvent: options.onEvent,
  });
}

export { isAbortError };
