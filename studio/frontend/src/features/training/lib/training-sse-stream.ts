


import { takeSseFrame } from "../api/sse-framing.ts";
import type { TrainingProgressPayload } from "../types/runtime";

export type TrainingProgressEventName =
  | "progress"
  | "heartbeat"
  | "complete"
  | "error";

export interface ParsedTrainingProgressEvent {
  event: TrainingProgressEventName;
  payload: TrainingProgressPayload;
  id: number | null;
}

const SSE_LINE_SEPARATOR = /\r?\n/;
const TRAINING_PROGRESS_EVENT_NAMES: ReadonlySet<string> = new Set([
  "progress",
  "heartbeat",
  "complete",
  "error",
]);
const NULLABLE_NUMERIC_FIELDS = [
  "loss",
  "learning_rate",
  "epoch",
  "elapsed_seconds",
  "eta_seconds",
  "grad_norm",
  "num_tokens",
  "eval_loss",
] as const;

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isTrainingProgressEventName(
  value: string,
): value is TrainingProgressEventName {
  return TRAINING_PROGRESS_EVENT_NAMES.has(value);
}

function isNullableFiniteNumber(value: unknown): boolean {
  return value === undefined || value === null || isFiniteNumber(value);
}

function normalizeNullableFiniteNumber(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function parseTrainingProgressPayload(
  value: unknown,
): TrainingProgressPayload | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }

  const payload = value as Record<string, unknown>;
  if (
    typeof payload.job_id !== "string" ||
    payload.job_id.trim().length === 0 ||
    !isFiniteNumber(payload.step) ||
    !isFiniteNumber(payload.total_steps) ||
    !isFiniteNumber(payload.progress_percent)
  ) {
    return null;
  }

  if (
    !NULLABLE_NUMERIC_FIELDS.every((field) =>
      isNullableFiniteNumber(payload[field]),
    )
  ) {
    return null;
  }

  return {
    job_id: payload.job_id,
    step: payload.step,
    total_steps: payload.total_steps,
    loss: normalizeNullableFiniteNumber(payload.loss),
    learning_rate: normalizeNullableFiniteNumber(payload.learning_rate),
    progress_percent: payload.progress_percent,
    epoch: normalizeNullableFiniteNumber(payload.epoch),
    elapsed_seconds: normalizeNullableFiniteNumber(payload.elapsed_seconds),
    eta_seconds: normalizeNullableFiniteNumber(payload.eta_seconds),
    grad_norm: normalizeNullableFiniteNumber(payload.grad_norm),
    num_tokens: normalizeNullableFiniteNumber(payload.num_tokens),
    eval_loss: normalizeNullableFiniteNumber(payload.eval_loss),
  };
}

function parseTrainingProgressData(
  dataLines: string[],
): TrainingProgressPayload | null {
  try {
    return parseTrainingProgressPayload(JSON.parse(dataLines.join("\n")));
  } catch {
    return null;
  }
}

interface TrainingSseFields {
  eventName: TrainingProgressEventName;
  id: number | null;
  dataLines: string[];
}

function applyTrainingSseLine(fields: TrainingSseFields, line: string): void {
  if (!line) {
    return;
  }
  if (line.startsWith("event:")) {
    const value = line.slice(6).trim();
    if (isTrainingProgressEventName(value)) {
      fields.eventName = value;
    }
    return;
  }
  if (line.startsWith("id:")) {
    const value = Number(line.slice(3).trim());
    fields.id = Number.isFinite(value) ? value : null;
    return;
  }
  if (line.startsWith("data:")) {
    fields.dataLines.push(line.slice(5).trimStart());
  }
}

function parseTrainingProgressEvent(
  rawEvent: string,
): ParsedTrainingProgressEvent | null {
  const lines = rawEvent.split(SSE_LINE_SEPARATOR);
  const fields: TrainingSseFields = {
    eventName: "progress",
    id: null,
    dataLines: [],
  };

  for (const line of lines) {
    applyTrainingSseLine(fields, line);
  }

  if (fields.dataLines.length === 0) {
    return null;
  }

  const payload = parseTrainingProgressData(fields.dataLines);
  if (!payload) {
    return null;
  }
  return { event: fields.eventName, payload, id: fields.id };
}

export async function consumeTrainingProgressStream(options: {
  body: ReadableStream<Uint8Array>;
  signal: AbortSignal;
  onEvent: (event: ParsedTrainingProgressEvent) => void;
}): Promise<void> {
  const reader = options.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!options.signal.aborted) {
      const { value, done } = await reader.read();
      if (done || options.signal.aborted) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      let frame = takeSseFrame(buffer);
      while (frame && !options.signal.aborted) {
        const rawEvent = frame.event;
        buffer = frame.remainder;

        if (!rawEvent.startsWith("retry:")) {
          const event = parseTrainingProgressEvent(rawEvent);
          if (event) {
            options.onEvent(event);
          }
        }

        frame = takeSseFrame(buffer);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
