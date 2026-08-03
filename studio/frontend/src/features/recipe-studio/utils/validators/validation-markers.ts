// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorConfig } from "../../types";

export const TOOL_VALIDATION_FN_MARKER = "unsloth_tool_validator";
export const CUSTOM_VALIDATION_FN_MARKER = "unsloth_custom_validator";

export type ToolValidatorSpec = {
  ext: string;
  command: string;
};

const TOOL_FILE_EXT_RE = /^[A-Za-z0-9.+-]{1,20}$/;
const LEADING_DOTS_RE = /^\.+/;
const BASE64_PLUS_RE = /\+/g;
const BASE64_SLASH_RE = /\//g;
const BASE64_PADDING_RE = /=+$/;

function toBase64Url(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replace(BASE64_PLUS_RE, "-")
    .replace(BASE64_SLASH_RE, "_")
    .replace(BASE64_PADDING_RE, "");
}

function fromBase64Url(input: string): string {
  const padded =
    input.replace(/-/g, "+").replace(/_/g, "/") +
    "=".repeat((4 - (input.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function encodeToolSpec(spec: ToolValidatorSpec): string {
  return toBase64Url(JSON.stringify({ ext: spec.ext, command: spec.command }));
}

export function decodeToolSpec(encoded: string): ToolValidatorSpec | null {
  let decoded: string;
  try {
    decoded = fromBase64Url(encoded);
  } catch {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(decoded);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      !("ext" in parsed) ||
      !("command" in parsed)
    ) {
      return null;
    }
    const record = parsed as Record<string, unknown>;
    const ext =
      typeof record.ext === "string"
        ? record.ext.trim().replace(LEADING_DOTS_RE, "")
        : "";
    const command =
      typeof record.command === "string" ? record.command.trim() : "";
    if (!(command && TOOL_FILE_EXT_RE.test(ext))) {
      return null;
    }
    return { ext, command };
  } catch {
    return null;
  }
}

export function encodeCustomSource(source: string): string {
  return toBase64Url(source);
}

export function decodeCustomSource(encoded: string): string {
  try {
    return fromBase64Url(encoded);
  } catch {
    return "";
  }
}

export function validationFunctionFromConfig(
  config: ValidatorConfig,
): string | null {
  if (config.validator_type === "tool") {
    const command = (config.tool_command ?? "").trim();
    const ext = (config.tool_ext ?? "").trim().replace(LEADING_DOTS_RE, "");
    if (!(command && TOOL_FILE_EXT_RE.test(ext))) {
      return null;
    }
    return `${TOOL_VALIDATION_FN_MARKER}:${encodeToolSpec({ ext, command })}`;
  }
  if (config.validator_type === "custom") {
    const source = (config.custom_source ?? "").trim();
    if (!source) {
      return null;
    }
    return `${CUSTOM_VALIDATION_FN_MARKER}:${encodeCustomSource(source)}`;
  }
  return null;
}
