// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The few-shot count to request: what the field shows, 0 included, once it shows the task's
 * default or the user's pick. null (the task's own default) only while the field shows neither,
 * or when the task takes no few-shot examples.
 */
export function fewshotParam(
  supported: boolean,
  known: boolean,
  picked: boolean,
  value: number,
): number | null {
  if (!supported) return null;
  return known || picked ? value : null;
}
