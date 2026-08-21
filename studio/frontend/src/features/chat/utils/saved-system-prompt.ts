// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InferenceParams } from "../types/runtime";

export const UNTITLED_SAVED_PROMPT_NAME = "Untitled Prompt";
export const SAVED_PROMPT_NAME_MAX_LENGTH = 80;

const FIRST_LINE_PATTERN = /\r?\n/;
const COLLAPSE_WHITESPACE_PATTERN = /\s+/g;

export function savedPromptNameFromText(text: string): string {
  const firstLine =
    text
      .split(FIRST_LINE_PATTERN)
      .map((line) => line.trim())
      .find((line) => line.length > 0) ?? "";
  const collapsed = firstLine.replace(COLLAPSE_WHITESPACE_PATTERN, " ");
  if (!collapsed) {
    return UNTITLED_SAVED_PROMPT_NAME;
  }
  return collapsed.length > SAVED_PROMPT_NAME_MAX_LENGTH
    ? collapsed.slice(0, SAVED_PROMPT_NAME_MAX_LENGTH)
    : collapsed;
}

export function applySystemPromptToParams(
  params: InferenceParams,
  text: string,
): InferenceParams {
  return { ...params, systemPrompt: text };
}
