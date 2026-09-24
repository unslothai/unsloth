// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The generate-time memory refusal, kept out of the page so it is testable. The backend refuses a
 *  size it estimates will not fit the free GPU memory even with the VAE decoding tile by tile, and
 *  the only way past it used to be an environment variable, which a desktop install has no terminal
 *  to set. The same override now travels with the request (`allow_oversized`), from a persisted
 *  setting or from the refusal toast's "Generate anyway". */

/** Response header /images/generate puts on this refusal's 400 (backend: diffusion_memory). */
export const MEMORY_REFUSAL_HEADER = "X-Unsloth-Refusal";
export const MEMORY_REFUSAL_KIND = "memory-estimate";

/** The setting's label. The backend quotes it in the refusal text, so keep the two identical. */
export const ALLOW_OVERSIZED_LABEL = "Allow oversized generations";
export const ALLOW_OVERSIZED_HINT =
  "Run a generation even when the memory check estimates it will not fit this GPU. Applies to the next generation, no reload needed. Sizes that fit with tiled VAE decoding already run without this. An oversized run can stop with an out-of-memory error, or on Windows spill into system RAM and slow the whole computer.";
export const MEMORY_REFUSAL_TITLE = "Not enough GPU memory for this size";
export const GENERATE_ANYWAY_LABEL = "Generate anyway";

/** A generate 400 the backend tagged as the memory-estimate refusal. */
export class MemoryEstimateRefusalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryEstimateRefusalError";
  }
}

/** Whether a failed /images/generate response is the memory-estimate refusal. */
export function isMemoryEstimateRefusal(
  status: number,
  refusalHeader: string | null | undefined,
): boolean {
  return (
    status === 400 &&
    (refusalHeader ?? "").trim().toLowerCase() === MEMORY_REFUSAL_KIND
  );
}

/** Whether the refusal toast should offer "Generate anyway". Not when the run already asked for
 *  the override: the backend then never refuses, so a refusal means something else changed. */
export function shouldOfferGenerateAnyway(input: {
  error: unknown;
  allowOversizedSent: boolean;
}): boolean {
  return (
    input.error instanceof MemoryEstimateRefusalError && !input.allowOversizedSent
  );
}

/** The request field: sent only when on, so a default request is byte-identical to before. */
export function allowOversizedField(
  persistedSetting: boolean,
  oneShot: boolean,
): true | undefined {
  return persistedSetting || oneShot ? true : undefined;
}

/** Whether a queued "Generate anyway" should start now. The toast appears while the refused run is
 *  still in its cleanup (busy is released only after an awaited status refresh), and a retry started
 *  then is dropped by the generate guard, so the click waits until nothing is busy. */
export function shouldRunQueuedOversizedRetry(input: {
  queued: boolean;
  busy: unknown;
}): boolean {
  return input.queued && input.busy === null;
}
