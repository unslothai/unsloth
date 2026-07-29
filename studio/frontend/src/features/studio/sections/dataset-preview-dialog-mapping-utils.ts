// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CheckFormatResponse } from "@/features/training";

const CHATML_ROLES = ["system", "user", "assistant"] as const;
const ALPACA_ROLES = ["instruction", "input", "output"] as const;
const SHAREGPT_ROLES = ["system", "human", "gpt"] as const;
const VLM_ROLES = ["image", "text"] as const;
const AUDIO_ROLES = ["audio", "text", "speaker_id"] as const;

const TO_CANONICAL: Record<string, string> = {
  user: "user",
  assistant: "assistant",
  system: "system",
  instruction: "user",
  input: "system",
  output: "assistant",
  human: "user",
  gpt: "assistant",
  image: "image",
  text: "text",
  audio: "audio",
  speaker_id: "speaker_id",
};

const FROM_CANONICAL: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

export function getAvailableRoles(
  isVlm: boolean,
  format?: string,
  isAudio?: boolean,
): readonly string[] {
  if (isAudio) {
    return AUDIO_ROLES;
  }
  if (isVlm) {
    return VLM_ROLES;
  }
  if (format === "alpaca") {
    return ALPACA_ROLES;
  }
  if (format === "sharegpt") {
    return SHAREGPT_ROLES;
  }
  return CHATML_ROLES;
}

export function isMappingComplete(
  mapping: Record<string, string>,
  isVlm: boolean,
  format?: string,
  isAudio?: boolean,
): boolean {
  const roles = new Set(Object.values(mapping));
  if (isAudio) {
    return roles.has("audio") && roles.has("text");
  }
  if (isVlm) {
    return roles.has("image") && roles.has("text");
  }
  if (format === "alpaca") {
    return roles.has("instruction") && roles.has("output");
  }
  if (format === "sharegpt") {
    return roles.has("human") && roles.has("gpt");
  }
  return roles.has("user") && roles.has("assistant");
}

export function remapRolesForFormat(
  mapping: Record<string, string>,
  format?: string,
): Record<string, string> {
  const table = format ? FROM_CANONICAL[format] : undefined;
  const result: Record<string, string> = {};
  for (const [column, role] of Object.entries(mapping)) {
    const canonical = TO_CANONICAL[role] ?? role;
    result[column] = table ? (table[canonical] ?? canonical) : canonical;
  }
  return result;
}

export function deriveDefaultMapping(
  data: CheckFormatResponse,
  isVlm: boolean,
  format?: string,
  isAudio?: boolean,
): Record<string, string> {
  if (data.suggested_mapping) {
    return remapRolesForFormat({ ...data.suggested_mapping }, format);
  }
  if (isAudio) {
    const result: Record<string, string> = {};
    if (data.detected_audio_column) {
      result[data.detected_audio_column] = "audio";
    }
    if (data.detected_text_column) {
      result[data.detected_text_column] = "text";
    }
    if (data.detected_speaker_column) {
      result[data.detected_speaker_column] = "speaker_id";
    }
    return result;
  }
  if (isVlm) {
    const result: Record<string, string> = {};
    if (data.detected_image_column) {
      result[data.detected_image_column] = "image";
    }
    if (data.detected_text_column) {
      result[data.detected_text_column] = "text";
    }
    return result;
  }
  return {};
}
