// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const H3_REFERENCE_MIN_SECONDS = 2;
export const H3_REFERENCE_MAX_SECONDS = 15;

export interface ReferenceVideoTrimFeedback {
  message: string;
  invalid: boolean;
}

export interface ReferenceVideoTrim {
  start: number | null;
  end: number | null;
}

/** Select the first model-sized interval when a source is too long. */
export function defaultReferenceVideoTrim(
  sourceDuration?: number,
): ReferenceVideoTrim {
  return sourceDuration !== undefined &&
    sourceDuration > H3_REFERENCE_MAX_SECONDS
    ? { start: 0, end: H3_REFERENCE_MAX_SECONDS }
    : { start: null, end: null };
}

/** Return user-facing validation for one optional reference-video interval. */
export function referenceVideoTrimError(
  label: string,
  start: number | null,
  end: number | null,
  sourceDuration?: number,
): string | null {
  if ((start === null) !== (end === null)) {
    return `Set both start and end times for ${label}`;
  }
  if (start === null || end === null) {
    return sourceDuration !== undefined &&
      sourceDuration > H3_REFERENCE_MAX_SECONDS
      ? `Select a 2 to 15 second section for ${label}`
      : null;
  }
  const duration = end - start;
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end <= start ||
    duration < H3_REFERENCE_MIN_SECONDS ||
    duration > H3_REFERENCE_MAX_SECONDS ||
    (sourceDuration !== undefined && end > sourceDuration)
  ) {
    return `Choose 2 to 15 seconds within ${label}`;
  }
  return null;
}

/** Describe the current trim in the form without presenting a valid selection as a warning. */
export function referenceVideoTrimFeedback(
  label: string,
  start: number | null,
  end: number | null,
  sourceDuration?: number,
): ReferenceVideoTrimFeedback {
  const source =
    sourceDuration !== undefined
      ? `${sourceDuration.toFixed(1)}s source. `
      : "";
  const error = referenceVideoTrimError(label, start, end, sourceDuration);
  if (error) {
    return { message: `${source}${error}.`, invalid: true };
  }
  if (start !== null && end !== null) {
    if (
      sourceDuration !== undefined &&
      sourceDuration > H3_REFERENCE_MAX_SECONDS &&
      start === 0 &&
      end === H3_REFERENCE_MAX_SECONDS
    ) {
      return {
        message: `${source}First ${H3_REFERENCE_MAX_SECONDS.toFixed(1)}s selected automatically. Adjust the times to use another section.`,
        invalid: false,
      };
    }
    return {
      message: `${source}Selected ${start.toFixed(1)}s to ${end.toFixed(1)}s (${(
        end - start
      ).toFixed(1)}s).`,
      invalid: false,
    };
  }
  return {
    message: source
      ? `${source}Trim is optional.`
      : "Trim is optional for clips up to 15 seconds.",
    invalid: false,
  };
}
