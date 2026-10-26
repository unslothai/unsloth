// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A load refused because "Keep multiple models loaded" is set to Always: the model it would
// have to unload stays, so this one does not load. Chat, Images and Video all surface it, so it
// lives here rather than beside the image memory refusal.

import { readFastApiError } from "@/lib/format-fastapi-error";

/** The same header as MEMORY_REFUSAL_HEADER; the two refusals are told apart by its value. */
export const REFUSAL_HEADER = "x-unsloth-refusal";
export const MULTI_MODEL_REFUSAL_KIND = "multi-model-always";
/** Mirrors gpu_arbiter.MULTI_MODEL_REFUSAL_MARKERS: the owner copy, then the managed copy. */
export const MULTI_MODEL_REFUSAL_MARKERS = [
  "is set to Always",
  "is set to keep loaded",
] as const;
export const MULTI_MODEL_REFUSAL_TITLE = "Model not loaded";

type HeaderBag = Headers | Record<string, string> | null | undefined;

/** A failed load POST that keeps what the toast needs: the status and the refusal headers. */
export class LoadRequestError extends Error {
  status: number;
  headers: Record<string, string>;

  constructor(
    message: string,
    status: number,
    headers: Record<string, string>,
  ) {
    super(message);
    this.name = "LoadRequestError";
    this.status = status;
    this.headers = headers;
  }
}

/** The thrown message is exactly readFastApiError's, so a load that was not refused reads as before. */
export async function loadRequestError(
  response: Response,
): Promise<LoadRequestError> {
  const headers: Record<string, string> = {};
  for (const name of [REFUSAL_HEADER, "retry-after"]) {
    const value = response.headers.get(name);
    if (value != null) headers[name] = value;
  }
  return new LoadRequestError(
    await readFastApiError(response),
    response.status,
    headers,
  );
}

export async function parseLoadJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw await loadRequestError(response);
  return (await response.json()) as T;
}

/** The refusal headers a failure carries, when it carries any. */
export function refusalHeadersOf(failure: unknown): HeaderBag {
  if (!failure || typeof failure !== "object") return undefined;
  const headers = (failure as { headers?: unknown }).headers;
  if (!headers || typeof headers !== "object") return undefined;
  return headers as HeaderBag;
}

export function failureMessage(failure: unknown): string {
  if (failure instanceof Error) return failure.message;
  return typeof failure === "string" ? failure : "";
}

function refusalHeaderValue(headers: HeaderBag): string | null {
  if (!headers) return null;
  if (typeof (headers as Headers).get === "function") {
    return (headers as Headers).get(REFUSAL_HEADER);
  }
  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() === REFUSAL_HEADER && typeof value === "string") {
      return value;
    }
  }
  return null;
}

/**
 * The header decides when it is there: response headers, or the plain dict a padded body
 * carries in `_deferred_error.headers`. Without one, the backend's wording does. The status is
 * accepted but never required, since a padded body answers 200.
 */
export function isMultiModelRefusal(
  _status: number | null | undefined,
  headers: HeaderBag,
  message: string | null | undefined,
): boolean {
  const value = refusalHeaderValue(headers)?.trim().toLowerCase();
  if (value) return value === MULTI_MODEL_REFUSAL_KIND;
  const text = message ?? "";
  return MULTI_MODEL_REFUSAL_MARKERS.some((marker) => text.includes(marker));
}
