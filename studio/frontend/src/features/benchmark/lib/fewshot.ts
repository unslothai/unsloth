// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The few-shot count a newly picked task starts at: its YAML default, or null to let lm-eval decide. */
export function seedFewshot(taskDefault: number | null | undefined): number | null {
  return taskDefault != null && taskDefault > 0 ? taskDefault : null;
}

/** The num_fewshot to send: null uses the task's default, 0 is an explicit zero-shot run. */
export function fewshotParam(value: number | null, supported: boolean): number | null {
  return supported ? value : null;
}
