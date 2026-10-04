// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// {{$now}} / {{$time}} change every second, so the prompt prefix never matches the cache (#9177).
// Case-sensitive like resolveSystemPromptVariables: {{$NOW}} is never substituted.
const HIGH_PRECISION_TIME_VARIABLE = /{{\s*\$(?:now|time)\s*}}/;

export function promptUsesHighPrecisionTimeVariables(prompt: string): boolean {
  return HIGH_PRECISION_TIME_VARIABLE.test(prompt ?? "");
}
