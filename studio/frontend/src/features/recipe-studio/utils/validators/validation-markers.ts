// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ToolScaffoldFile, ValidatorConfig } from "../../types";

export const TOOL_VALIDATION_FN_MARKER = "unsloth_tool_validator";
export const CUSTOM_VALIDATION_FN_MARKER = "unsloth_custom_validator";

export type ToolValidatorSpec = {
  ext: string;
  command: string;
  scaffold?: ToolScaffoldFile[];
  // biome-ignore lint/style/useNamingConvention: marker schema (mirrors the backend spec)
  output_max_chars?: number;
  // biome-ignore lint/style/useNamingConvention: marker schema (mirrors the backend spec)
  source_file_max_chars?: number;
};

const TOOL_FILE_EXT_RE = /^[A-Za-z0-9.+-]{1,20}$/;
const TOOL_SCAFFOLD_PATH_RE = /^[A-Za-z0-9._+-]+(?:\/[A-Za-z0-9._+-]+)*$/;
export const TOOL_SCAFFOLD_MAX_ROWS = 10;
const TOOL_SCAFFOLD_MAX_TOTAL_CHARS = 32 * 1024;

export const TOOL_COMMAND_MAX_CHARS = 8 * 1024;
export const CUSTOM_SOURCE_MAX_CHARS = 64 * 1024;
export const BATCH_SIZE_MAX = 512;
export const MARKER_MAX_CHARS = 128 * 1024;

export const TOOL_OUTPUT_MAX_KIB_DEFAULT = 8;
export const TOOL_OUTPUT_MAX_KIB_MIN = 1;
export const TOOL_OUTPUT_MAX_KIB_MAX = 256;
export const TOOL_SCAFFOLD_FILE_MAX_KIB_DEFAULT = 32;
export const TOOL_SCAFFOLD_FILE_MAX_KIB_MIN = 1;
export const TOOL_SCAFFOLD_FILE_MAX_KIB_MAX = 64;

const LEADING_DOTS_RE = /^\.+/;
const PATH_HOSTILE_EXT_RE = /[/\\]/;
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
  if (input.length > MARKER_MAX_CHARS) {
    throw new Error("marker payload exceeds size limit");
  }
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

function normalizedToolScaffoldRows(scaffold: ToolScaffoldFile[] | undefined): {
  rows: ToolScaffoldFile[];
  totalChars: number;
} {
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
    // biome-ignore lint/style/useNamingConvention: marker schema
    output_max_chars?: number;
    // biome-ignore lint/style/useNamingConvention: marker schema
    source_file_max_chars?: number;
  } = {
    ext: spec.ext,
    command: spec.command,
  };
  const scaffold = normalizeToolScaffold(spec.scaffold);
  if (scaffold.length > 0) {
    payload.scaffold = scaffold;
  }
  if (spec.output_max_chars !== undefined) {
    payload.output_max_chars = spec.output_max_chars;
  }
  if (spec.source_file_max_chars !== undefined) {
    payload.source_file_max_chars = spec.source_file_max_chars;
  }
  return toBase64Url(JSON.stringify(payload));
}

export function normalizeToolExt(ext: string): string {
  return ext.trim().replace(LEADING_DOTS_RE, "");
}

export function isValidToolExt(ext: string): boolean {
  return TOOL_FILE_EXT_RE.test(normalizeToolExt(ext));
}

function parseKib(value: string | undefined): number | null {
  const trimmed = (value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  // Fractional values are accepted: imported markers carry exact char counts
  // (chars / 1024 is exact in binary), so "195.3125" round-trips losslessly.
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeToolOutputMaxKib(value: string | undefined): string {
  const parsed = parseKib(value);
  if (parsed === null) {
    return String(TOOL_OUTPUT_MAX_KIB_DEFAULT);
  }
  return String(
    Math.min(
      Math.max(parsed, TOOL_OUTPUT_MAX_KIB_MIN),
      TOOL_OUTPUT_MAX_KIB_MAX,
    ),
  );
}

export function normalizeToolScaffoldFileMaxKib(
  value: string | undefined,
): string {
  const parsed = parseKib(value);
  if (parsed === null) {
    return String(TOOL_SCAFFOLD_FILE_MAX_KIB_DEFAULT);
  }
  return String(
    Math.min(
      Math.max(parsed, TOOL_SCAFFOLD_FILE_MAX_KIB_MIN),
      TOOL_SCAFFOLD_FILE_MAX_KIB_MAX,
    ),
  );
}

export function toolOutputMaxKibError(
  value: string | undefined,
): string | null {
  const trimmed = (value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return `Max tool output must be a number between ${TOOL_OUTPUT_MAX_KIB_MIN} and ${TOOL_OUTPUT_MAX_KIB_MAX} KiB.`;
  }
  if (parsed < TOOL_OUTPUT_MAX_KIB_MIN || parsed > TOOL_OUTPUT_MAX_KIB_MAX) {
    return `Max tool output must be between ${TOOL_OUTPUT_MAX_KIB_MIN} and ${TOOL_OUTPUT_MAX_KIB_MAX} KiB.`;
  }
  return null;
}

export function toolScaffoldFileMaxKibError(
  value: string | undefined,
): string | null {
  const trimmed = (value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return `Max scaffold file size must be a number between ${TOOL_SCAFFOLD_FILE_MAX_KIB_MIN} and ${TOOL_SCAFFOLD_FILE_MAX_KIB_MAX} KiB.`;
  }
  if (
    parsed < TOOL_SCAFFOLD_FILE_MAX_KIB_MIN ||
    parsed > TOOL_SCAFFOLD_FILE_MAX_KIB_MAX
  ) {
    return `Max scaffold file size must be between ${TOOL_SCAFFOLD_FILE_MAX_KIB_MIN} and ${TOOL_SCAFFOLD_FILE_MAX_KIB_MAX} KiB.`;
  }
  return null;
}

function toolOutputMaxChars(config: ValidatorConfig): number {
  // Math.round is exact for chars/1024 (power-of-two division), so imported
  // fractional KiB values round-trip losslessly.
  return Math.round(
    Number(normalizeToolOutputMaxKib(config.tool_output_max_kib)) * 1024,
  );
}

function toolSourceFileMaxChars(config: ValidatorConfig): number {
  return Math.round(
    Number(normalizeToolScaffoldFileMaxKib(config.tool_scaffold_file_max_kib)) *
      1024,
  );
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
    // A missing/invalid ext or command is NOT rejected here: the config
    // validators flag those. Rejecting them would make an invalid mid-edit
    // state un-importable and lose the rest of the tool's state (scaffold
    // rows, command) on recipe reload. Path-hostile values (path separators)
    // are dropped from the round-trip instead, so they can never reach a
    // filename/shell string; the validators then flag the missing extension.
    const safeExt = PATH_HOSTILE_EXT_RE.test(ext) ? "" : ext;
    const rawScaffold = Array.isArray(record.scaffold) ? record.scaffold : [];
    const scaffold = normalizeToolScaffold(rawScaffold as ToolScaffoldFile[]);
    if (rawScaffold.length > 0 && scaffold.length === 0) {
      return null;
    }
    const outputMaxChars =
      typeof record.output_max_chars === "number"
        ? record.output_max_chars
        : undefined;
    const sourceFileMaxChars =
      typeof record.source_file_max_chars === "number"
        ? record.source_file_max_chars
        : undefined;
    const spec: ToolValidatorSpec = { ext: safeExt, command };
    if (scaffold.length > 0) {
      spec.scaffold = scaffold;
    }
    if (outputMaxChars !== undefined) {
      spec.output_max_chars = outputMaxChars;
    }
    if (sourceFileMaxChars !== undefined) {
      spec.source_file_max_chars = sourceFileMaxChars;
    }
    return spec;
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
    // Never serialize a marker whose scaffold silently lost rows: an
    // over-limit scaffold must surface as a validation error instead.
    if (toolScaffoldLimitError(config.tool_scaffold) !== null) {
      return null;
    }
    const ext = normalizeToolExt(config.tool_ext ?? "");
    const command = (config.tool_command ?? "").trim();
    const scaffold = normalizeToolScaffold(config.tool_scaffold);
    const outputMaxChars = toolOutputMaxChars(config);
    const sourceFileMaxChars = toolSourceFileMaxChars(config);
    // A missing command/ext still serializes so an invalid mid-edit state
    // round-trips as a tool check; the config validators flag the gap.
    // Caps are only serialized when they differ from the defaults, so legacy
    // markers stay byte-identical and the backend falls back to defaults.
    return `${TOOL_VALIDATION_FN_MARKER}:${encodeToolSpec({
      ext,
      command,
      ...(scaffold.length > 0 ? { scaffold } : {}),
      ...(outputMaxChars !== TOOL_OUTPUT_MAX_KIB_DEFAULT * 1024
        ? {
            // biome-ignore lint/style/useNamingConvention: marker schema
            output_max_chars: outputMaxChars,
          }
        : {}),
      ...(sourceFileMaxChars !== TOOL_SCAFFOLD_FILE_MAX_KIB_DEFAULT * 1024
        ? {
            // biome-ignore lint/style/useNamingConvention: marker schema
            source_file_max_chars: sourceFileMaxChars,
          }
        : {}),
    })}`;
  }
  if (config.validator_type === "custom") {
    const source = (config.custom_source ?? "").trim();
    // Same round-trip guarantee as the tool branch: an empty source still
    // serializes as a custom check so the block type survives a reload.
    return `${CUSTOM_VALIDATION_FN_MARKER}:${encodeCustomSource(source)}`;
  }
  return null;
}
