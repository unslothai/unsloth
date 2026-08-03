// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ToolScaffoldFile, ValidatorConfig } from "../../types";

export const TOOL_VALIDATION_FN_MARKER = "unsloth_tool_validator";
export const CUSTOM_VALIDATION_FN_MARKER = "unsloth_custom_validator";

export type ToolValidatorSpec = {
  ext: string;
  command: string;
  scaffold?: ToolScaffoldFile[];
};

const TOOL_FILE_EXT_RE = /^[A-Za-z0-9.+-]{1,20}$/;
const TOOL_SCAFFOLD_PATH_RE = /^[A-Za-z0-9._+-]+(?:\/[A-Za-z0-9._+-]+)*$/;
const TOOL_SCAFFOLD_MAX_ROWS = 10;
const TOOL_SCAFFOLD_MAX_TOTAL_CHARS = 32 * 1024;
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

function normalizeToolScaffoldEntry(entry: unknown): ToolScaffoldFile | null {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const record = entry as Record<string, unknown>;
  const path = typeof record.path === "string" ? record.path.trim() : "";
  const content = typeof record.content === "string" ? record.content : "";
  if (!path) {
    return null;
  }
  if (
    !TOOL_SCAFFOLD_PATH_RE.test(path) ||
    path.split("/").some((segment) => segment === "." || segment === "..")
  ) {
    return null;
  }
  return { path, content };
}

export function firstInvalidToolScaffoldPath(
  scaffold: ToolScaffoldFile[] | undefined,
): string | null {
  if (!Array.isArray(scaffold)) {
    return null;
  }
  for (const entry of scaffold) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const path = typeof record.path === "string" ? record.path.trim() : "";
    if (!path) {
      continue;
    }
    if (
      !TOOL_SCAFFOLD_PATH_RE.test(path) ||
      path.split("/").some((segment) => segment === "." || segment === "..")
    ) {
      return path;
    }
  }
  return null;
}

function normalizedToolScaffoldRows(
  scaffold: ToolScaffoldFile[] | undefined,
): { rows: ToolScaffoldFile[]; totalChars: number } {
  const rows: ToolScaffoldFile[] = [];
  let totalChars = 0;
  if (Array.isArray(scaffold)) {
    for (const entry of scaffold) {
      const normalized = normalizeToolScaffoldEntry(entry);
      if (normalized === null) {
        continue;
      }
      rows.push(normalized);
      totalChars += normalized.path.length + normalized.content.length;
    }
  }
  return { rows, totalChars };
}

export function toolScaffoldLimitError(
  scaffold: ToolScaffoldFile[] | undefined,
): string | null {
  const { rows, totalChars } = normalizedToolScaffoldRows(scaffold);
  if (rows.length > TOOL_SCAFFOLD_MAX_ROWS) {
    return `Too many scaffold files (max ${TOOL_SCAFFOLD_MAX_ROWS}).`;
  }
  if (totalChars > TOOL_SCAFFOLD_MAX_TOTAL_CHARS) {
    return "Scaffold content is too large (max 32 KiB).";
  }
  return null;
}

export function normalizeToolScaffold(
  scaffold: ToolScaffoldFile[] | undefined,
): ToolScaffoldFile[] {
  const { rows, totalChars } = normalizedToolScaffoldRows(scaffold);
  if (
    rows.length > TOOL_SCAFFOLD_MAX_ROWS ||
    totalChars > TOOL_SCAFFOLD_MAX_TOTAL_CHARS
  ) {
    return [];
  }
  return rows;
}

export function encodeToolSpec(spec: ToolValidatorSpec): string {
  const payload: {
    ext: string;
    command: string;
    scaffold?: ToolScaffoldFile[];
  } = {
    ext: spec.ext,
    command: spec.command,
  };
  const scaffold = normalizeToolScaffold(spec.scaffold);
  if (scaffold.length > 0) {
    payload.scaffold = scaffold;
  }
  return toBase64Url(JSON.stringify(payload));
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
    const rawScaffold = Array.isArray(record.scaffold) ? record.scaffold : [];
    const scaffold = normalizeToolScaffold(rawScaffold as ToolScaffoldFile[]);
    if (rawScaffold.length > 0 && scaffold.length === 0) {
      return null;
    }
    return scaffold.length > 0 ? { ext, command, scaffold } : { ext, command };
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
    const scaffold = normalizeToolScaffold(config.tool_scaffold);
    return `${TOOL_VALIDATION_FN_MARKER}:${encodeToolSpec({
      ext,
      command,
      ...(scaffold.length > 0 ? { scaffold } : {}),
    })}`;
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
