// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Backend: diffusion_memory.IMAGE_REFUSAL_HEADER. */
export const MEMORY_REFUSAL_HEADER = "X-Unsloth-Refusal";
export const MEMORY_REFUSAL_KIND = "memory-estimate";

/** The backend quotes this label in the refusal text; keep the two identical. */
export const ALLOW_OVERSIZED_LABEL = "Allow oversized generations";
export const ALLOW_OVERSIZED_HINT =
  "Run a generation even when the memory check estimates it will not fit this GPU. Applies to the next generation, no reload needed. Sizes that fit with tiled VAE decoding already run without this. An oversized run can stop with an out-of-memory error, or on Windows spill into system RAM and slow the whole computer.";
export const MEMORY_REFUSAL_TITLE = "Not enough GPU memory for this size";
export const GENERATE_ANYWAY_LABEL = "Generate anyway";

export class MemoryEstimateRefusalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryEstimateRefusalError";
  }
}

export function isMemoryEstimateRefusal(
  status: number,
  refusalHeader: string | null | undefined,
): boolean {
  return (
    status === 400 &&
    (refusalHeader ?? "").trim().toLowerCase() === MEMORY_REFUSAL_KIND
  );
}

/** Not when the run already sent the override: then the refusal means something else changed. */
export function shouldOfferGenerateAnyway(input: {
  error: unknown;
  allowOversizedSent: boolean;
}): boolean {
  return (
    input.error instanceof MemoryEstimateRefusalError && !input.allowOversizedSent
  );
}

export function allowOversizedField(
  persistedSetting: boolean,
  oneShot: boolean,
): true | undefined {
  return persistedSetting || oneShot ? true : undefined;
}

/** A retry started while the refused run is still busy is dropped by the generate guard. */
export function shouldRunQueuedOversizedRetry(input: {
  queued: boolean;
  busy: unknown;
}): boolean {
  return input.queued && input.busy === null;
}
