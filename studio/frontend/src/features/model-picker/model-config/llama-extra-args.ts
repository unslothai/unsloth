// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const LLAMA_EXTRA_ARGS_MAX_TOKENS = 256;
export const LLAMA_EXTRA_ARGS_MAX_TOTAL_BYTES = 32 * 1024;

export function llamaExtraArgsPayload(args: string[] | undefined): {
  llama_extra_args?: string[];
} {
  return args === undefined ? {} : { llama_extra_args: [...args] };
}

export function serializeLlamaExtraArgsRequestBody<T extends object>(
  payload: T,
  args: string[] | undefined,
): string {
  return JSON.stringify({ ...payload, ...llamaExtraArgsPayload(args) });
}

export function moveLlamaExtraArgsSelection(
  current: number,
  direction: "next" | "previous",
  count: number,
): number {
  if (count <= 0) return 0;
  return direction === "next"
    ? (current + 1) % count
    : (current - 1 + count) % count;
}

export interface LlamaServerArgument {
  name: string;
  aliases: string[];
  value_hint: string | null;
  choices: string[];
  description: string;
  default_value: string | null;
  env_var: string | null;
  group: string | null;
  policy_category: string;
  value_arity: number;
  deprecated: boolean;
  managed_by_studio: boolean;
  overlaps_studio_control: boolean;
}

export interface LlamaServerArgumentsResponse {
  available: boolean;
  authoritative: boolean;
  installed_tag: string | null;
  error_code: string | null;
  arguments: LlamaServerArgument[];
  managed_flags: string[];
  managed_flag_groups: string[][];
}

export interface ParsedLlamaExtraArgs {
  tokens: string[];
  spans: { start: number; end: number; value: string }[];
  error: { message: string; offset: number } | null;
}

export type LlamaExtraArgsDiagnosticKind =
  | "syntax"
  | "limit"
  | "managed"
  | "unknown"
  | "deprecated"
  | "duplicate"
  | "missing-value"
  | "invalid-choice"
  | "overlap";

export interface LlamaExtraArgsDiagnostic {
  kind: LlamaExtraArgsDiagnosticKind;
  severity: "error" | "warning";
  message: string;
  tokenIndex?: number;
}

export interface LlamaExtraArgsCompletion {
  kind: "flag" | "value";
  insertText: string;
  label: string;
  argument: LlamaServerArgument;
  replaceStart: number;
  replaceEnd: number;
}

const encoder = new TextEncoder();

function utf8Bytes(value: string): number {
  return encoder.encode(value).byteLength;
}

function hasDisallowedControlCharacter(value: string): boolean {
  return [...value].some((char) => {
    const codePoint = char.codePointAt(0) ?? 0;
    return (
      char !== "\t" &&
      (codePoint < 32 ||
        (codePoint >= 0x7f && codePoint <= 0x9f) ||
        (codePoint >= 0xd800 && codePoint <= 0xdfff) ||
        codePoint === 0x2028 ||
        codePoint === 0x2029)
    );
  });
}

function hasMalformedArgToken(token: string): boolean {
  return token === "" || token !== token.trim() || token === "-" || token === "--";
}

export function areLlamaExtraArgsWithinLimits(
  tokens: readonly string[],
): boolean {
  return (
    tokens.length <= LLAMA_EXTRA_ARGS_MAX_TOKENS &&
    tokens.every(
      (token) =>
        !(hasDisallowedControlCharacter(token) || hasMalformedArgToken(token)),
    ) &&
    tokens.reduce((sum, token) => sum + utf8Bytes(token), 0) <=
      LLAMA_EXTRA_ARGS_MAX_TOTAL_BYTES
  );
}

export function llamaExtraArgsCatalogBlocksPersistence(
  tokens: readonly string[],
  catalogAvailable: boolean,
  catalogAuthoritative: boolean,
): boolean {
  return tokens.length > 0 && (!catalogAvailable || !catalogAuthoritative);
}

function isEscapable(next: string): boolean {
  return /\s/.test(next) || next === "\\" || next === '"' || next === "'";
}

/**
 * Tokenize a command-line fragment without invoking or emulating a shell.
 * Whitespace separates tokens; matching quotes group text; a backslash escapes
 * whitespace, quotes, or another backslash. Other backslashes stay literal so
 * an unquoted Windows path is not silently rewritten.
 */
export function parseLlamaExtraArgs(text: string): ParsedLlamaExtraArgs {
  const tokens: string[] = [];
  const spans: ParsedLlamaExtraArgs["spans"] = [];
  let i = 0;
  while (i < text.length) {
    while (i < text.length && /\s/.test(text[i])) i += 1;
    if (i >= text.length) break;
    const start = i;
    let value = "";
    let quote: "'" | '"' | null = null;
    let quoteOffset = -1;
    while (i < text.length) {
      const char = text[i];
      if (quote === null && /\s/.test(char)) break;
      if (char === "\\") {
        if (i + 1 >= text.length) {
          return {
            tokens,
            spans,
            error: {
              message: "A trailing backslash must escape another character.",
              offset: i,
            },
          };
        }
        const next = text[i + 1];
        if (isEscapable(next)) {
          value += next;
          i += 2;
          continue;
        }
        value += char;
        i += 1;
        continue;
      }
      if (quote === null && (char === "'" || char === '"')) {
        quote = char;
        quoteOffset = i;
        i += 1;
        continue;
      }
      if (quote === char) {
        quote = null;
        i += 1;
        continue;
      }
      value += char;
      i += 1;
    }
    if (quote !== null) {
      return {
        tokens,
        spans,
        error: {
          message: `Unterminated ${quote === '"' ? "double" : "single"} quote.`,
          offset: quoteOffset,
        },
      };
    }
    if (value.length === 0) {
      return {
        tokens,
        spans,
        error: { message: "Arguments cannot be empty.", offset: start },
      };
    }
    tokens.push(value);
    spans.push({ start, end: i, value });
  }
  return { tokens, spans, error: null };
}

/** A stable editor representation that round-trips every token, including paths. */
export function formatLlamaExtraArgs(tokens: readonly string[]): string {
  return tokens
    .map((token) => {
      if (token.length === 0) return '""';
      if (/^[\p{L}\p{N}_./:=+@%,-]+$/u.test(token)) return token;
      return `"${token.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
    })
    .join(" ");
}

function canonicalFlag(token: string): string {
  return token.split("=", 1)[0].replaceAll("_", "-");
}

function looksLikeFlag(token: string): boolean {
  return /^--?[A-Za-z]/.test(token);
}

function catalogIndex(
  catalog: readonly LlamaServerArgument[],
): Map<string, LlamaServerArgument> {
  const index = new Map<string, LlamaServerArgument>();
  for (const argument of catalog) {
    for (const name of [argument.name, ...argument.aliases]) {
      index.set(canonicalFlag(name), argument);
    }
  }
  return index;
}

interface ResolvedCatalogFlagToken {
  argument: LlamaServerArgument | null;
  rawFlag: string;
  attachedValue: string | undefined;
  separator: "none" | "equals" | "attached";
}

function resolveCatalogFlagToken(
  token: string,
  catalog: readonly LlamaServerArgument[],
  index = catalogIndex(catalog),
): ResolvedCatalogFlagToken {
  const [rawFlag, equalsValue] = token.split(/=(.*)/s, 2);
  const exact = index.get(canonicalFlag(rawFlag)) ?? null;
  if (exact) {
    return {
      argument: exact,
      rawFlag,
      attachedValue: equalsValue,
      separator: equalsValue === undefined ? "none" : "equals",
    };
  }
  if (
    equalsValue !== undefined ||
    !rawFlag.startsWith("-") ||
    rawFlag.startsWith("--")
  ) {
    return {
      argument: null,
      rawFlag,
      attachedValue: equalsValue,
      separator: equalsValue === undefined ? "none" : "equals",
    };
  }

  const normalized = canonicalFlag(rawFlag);
  const attached = catalog
    .flatMap((argument) =>
      [argument.name, ...argument.aliases].map((spelling) => ({
        argument,
        spelling,
        normalizedSpelling: canonicalFlag(spelling),
      })),
    )
    .filter(
      ({ argument, normalizedSpelling }) =>
        argument.value_arity > 0 &&
        normalizedSpelling.startsWith("-") &&
        !normalizedSpelling.startsWith("--") &&
        normalized.length > normalizedSpelling.length &&
        normalized.startsWith(normalizedSpelling),
    )
    .sort(
      (left, right) =>
        right.normalizedSpelling.length - left.normalizedSpelling.length,
    )[0];
  if (!attached) {
    return {
      argument: null,
      rawFlag,
      attachedValue: undefined,
      separator: "none",
    };
  }
  return {
    argument: attached.argument,
    rawFlag: rawFlag.slice(0, attached.spelling.length),
    attachedValue: rawFlag.slice(attached.spelling.length),
    separator: "attached",
  };
}

function managedPolicyArgument(group: readonly string[]): LlamaServerArgument {
  return {
    name: group[0],
    aliases: [...group.slice(1)],
    value_hint: null,
    choices: [],
    description: "Managed by Run Settings.",
    default_value: null,
    env_var: null,
    group: null,
    policy_category: "Server administration",
    value_arity: 0,
    deprecated: false,
    managed_by_studio: true,
    overlaps_studio_control: false,
  };
}

/** Merge the backend's stable policy into optional installed-help metadata. */
export function llamaServerDiagnosticCatalog(
  response: Pick<
    LlamaServerArgumentsResponse,
    "arguments" | "managed_flags" | "managed_flag_groups"
  >,
): LlamaServerArgument[] {
  const catalog = response.arguments.map((argument) => ({
    ...argument,
    aliases: [...argument.aliases],
    choices: [...argument.choices],
  }));
  const grouped = new Set(response.managed_flag_groups.flat());
  const groups = [
    ...response.managed_flag_groups,
    ...response.managed_flags
      .filter((flag) => !grouped.has(flag))
      .map((flag) => [flag]),
  ].filter((group) => group.length > 0);

  for (const group of groups) {
    const normalized = new Set(group.map(canonicalFlag));
    const existing = catalog.find((argument) =>
      [argument.name, ...argument.aliases].some((name) =>
        normalized.has(canonicalFlag(name)),
      ),
    );
    if (!existing) {
      catalog.push(managedPolicyArgument(group));
      continue;
    }
    existing.managed_by_studio = true;
    const aliases = new Set([existing.name, ...existing.aliases]);
    for (const flag of group) aliases.add(flag);
    aliases.delete(existing.name);
    existing.aliases = [...aliases];
  }
  return catalog;
}

export function llamaServerArgumentGroupLabel(
  argument: Pick<LlamaServerArgument, "group" | "policy_category">,
): string {
  return argument.policy_category?.trim() || "Unclassified";
}

function takesValue(argument: LlamaServerArgument): boolean {
  return argument.value_arity > 0;
}

export function llamaServerArgumentTakesValue(
  argument: LlamaServerArgument,
): boolean {
  return takesValue(argument);
}

export interface LlamaExtraArgRow {
  start: number;
  end: number;
  flag: string;
  value: string | undefined;
  separator: "none" | "separate" | "equals" | "attached";
  argument: LlamaServerArgument | null;
  valueExpected: boolean;
}

/** Group the flat request tokens into compact flag/value rows for the editor. */
export function llamaExtraArgRows(
  tokens: readonly string[],
  catalog: readonly LlamaServerArgument[],
): LlamaExtraArgRow[] {
  const index = catalogIndex(catalog);
  const rows: LlamaExtraArgRow[] = [];
  for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex += 1) {
    const token = tokens[tokenIndex];
    const resolved = resolveCatalogFlagToken(token, catalog, index);
    const { argument, rawFlag, attachedValue } = resolved;
    const valueExpected = argument ? takesValue(argument) : true;

    if (attachedValue !== undefined) {
      rows.push({
        start: tokenIndex,
        end: tokenIndex + 1,
        flag: rawFlag,
        value: attachedValue,
        separator: resolved.separator,
        argument,
        valueExpected,
      });
      continue;
    }

    const next = tokens[tokenIndex + 1];
    if (valueExpected && next !== undefined && !looksLikeFlag(next)) {
      rows.push({
        start: tokenIndex,
        end: tokenIndex + 2,
        flag: rawFlag,
        value: next,
        separator: "separate",
        argument,
        valueExpected,
      });
      tokenIndex += 1;
      continue;
    }

    rows.push({
      start: tokenIndex,
      end: tokenIndex + 1,
      flag: rawFlag,
      value: undefined,
      separator: "none",
      argument,
      valueExpected,
    });
  }
  return rows;
}

/** Replace one visual row while retaining the original equals/separate style. */
export function replaceLlamaExtraArgRow(
  tokens: readonly string[],
  row: LlamaExtraArgRow,
  value: string,
): string[] {
  const replacement =
    value.length === 0
      ? [row.flag]
      : row.separator === "equals"
        ? [`${row.flag}=${value}`]
        : row.separator === "attached"
          ? [`${row.flag}${value}`]
          : [row.flag, value];
  return [
    ...tokens.slice(0, row.start),
    ...replacement,
    ...tokens.slice(row.end),
  ];
}

/** Rename one visual row while retaining its value and separator style. */
export function replaceLlamaExtraArgRowFlag(
  tokens: readonly string[],
  row: LlamaExtraArgRow,
  flag: string,
): string[] {
  const replacement =
    row.value === undefined
      ? [flag]
      : row.separator === "equals"
        ? [`${flag}=${row.value}`]
        : row.separator === "attached"
          ? [`${flag}${row.value}`]
          : [flag, row.value];
  return [
    ...tokens.slice(0, row.start),
    ...replacement,
    ...tokens.slice(row.end),
  ];
}

export function diagnoseLlamaExtraArgs(
  text: string,
  catalog: readonly LlamaServerArgument[] | null,
  catalogAuthoritative = true,
): LlamaExtraArgsDiagnostic[] {
  const parsed = parseLlamaExtraArgs(text);
  const diagnostics: LlamaExtraArgsDiagnostic[] = [];
  if (hasDisallowedControlCharacter(text)) {
    diagnostics.push({
      kind: "limit",
      severity: "error",
      message: "Line separators and control characters are not allowed.",
    });
  }
  if (parsed.error) {
    diagnostics.push({
      kind: "syntax",
      severity: "error",
      message: parsed.error.message,
    });
  }
  if (parsed.tokens.some(hasMalformedArgToken)) {
    diagnostics.push({
      kind: "syntax",
      severity: "error",
      message: "Arguments cannot contain empty or whitespace-padded tokens.",
    });
  }
  if (parsed.tokens.length > LLAMA_EXTRA_ARGS_MAX_TOKENS) {
    diagnostics.push({
      kind: "limit",
      severity: "error",
      message: `Use at most ${LLAMA_EXTRA_ARGS_MAX_TOKENS} argument tokens.`,
    });
  }
  const totalBytes = parsed.tokens.reduce(
    (sum, token) => sum + utf8Bytes(token),
    0,
  );
  if (totalBytes > LLAMA_EXTRA_ARGS_MAX_TOTAL_BYTES) {
    diagnostics.push({
      kind: "limit",
      severity: "error",
      message: `Arguments may use at most ${LLAMA_EXTRA_ARGS_MAX_TOTAL_BYTES / 1024} KiB total.`,
    });
  }
  if (!catalog) return diagnostics;

  const index = catalogIndex(catalog);
  const seen = new Map<string, number>();
  for (let tokenIndex = 0; tokenIndex < parsed.tokens.length; tokenIndex += 1) {
    const token = parsed.tokens[tokenIndex];
    if (!looksLikeFlag(token)) continue;
    const resolved = resolveCatalogFlagToken(token, catalog, index);
    const { rawFlag, attachedValue } = resolved;
    const argument = resolved.argument;
    if (!argument) {
      if (catalogAuthoritative) {
        diagnostics.push({
          kind: "unknown",
          severity: "warning",
          tokenIndex,
          message: `Unknown llama.cpp argument ${rawFlag}; the installed build may still accept it.`,
        });
      }
      continue;
    }
    if (argument.managed_by_studio) {
      diagnostics.push({
        kind: "managed",
        severity: "error",
        tokenIndex,
        message: `${rawFlag} is managed by Studio and cannot be passed here.`,
      });
    }
    if (argument.deprecated) {
      diagnostics.push({
        kind: "deprecated",
        severity: "warning",
        tokenIndex,
        message: `${rawFlag} is deprecated by the installed llama.cpp build.`,
      });
    }
    if (argument.overlaps_studio_control) {
      diagnostics.push({
        kind: "overlap",
        severity: "warning",
        tokenIndex,
        message: `${rawFlag} overrides the matching Run Setting.`,
      });
    }
    const canonical = argument.name;
    const first = seen.get(canonical);
    if (first !== undefined) {
      diagnostics.push({
        kind: "duplicate",
        severity: "warning",
        tokenIndex,
        message: `${rawFlag} is repeated; llama.cpp will use its last applicable value.`,
      });
    } else {
      seen.set(canonical, tokenIndex);
    }

    if (!takesValue(argument)) continue;
    const values = attachedValue === undefined ? [] : [attachedValue];
    let consumedSeparateValues = 0;
    while (values.length < argument.value_arity) {
      const next = parsed.tokens[tokenIndex + consumedSeparateValues + 1];
      if (next === undefined || looksLikeFlag(next)) break;
      values.push(next);
      consumedSeparateValues += 1;
    }
    if (
      values.length < argument.value_arity ||
      values.some((value) => value === "")
    ) {
      diagnostics.push({
        kind: "missing-value",
        severity: "error",
        tokenIndex,
        message: `${rawFlag} expects ${argument.value_arity === 1 ? (argument.value_hint ?? "a value") : `${argument.value_arity} values`}.`,
      });
      continue;
    }
    if (argument.choices.length > 0 && !argument.choices.includes(values[0])) {
      diagnostics.push({
        kind: "invalid-choice",
        severity: "warning",
        tokenIndex,
        message: `${rawFlag} expects one of: ${argument.choices.join(", ")}.`,
      });
    }
    tokenIndex += consumedSeparateValues;
  }
  return diagnostics;
}

function rawTokenSpans(text: string): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = [];
  let start: number | null = null;
  let quote: "'" | '"' | null = null;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (start === null) {
      if (/\s/.test(char)) continue;
      start = i;
    }
    if (char === "\\" && i + 1 < text.length && isEscapable(text[i + 1])) {
      i += 1;
      continue;
    }
    if (quote === null && (char === "'" || char === '"')) quote = char;
    else if (quote === char) quote = null;
    else if (quote === null && /\s/.test(char)) {
      spans.push({ start, end: i });
      start = null;
    }
  }
  if (start !== null) spans.push({ start, end: text.length });
  return spans;
}

function spanAtCaret(
  text: string,
  caret: number,
): { start: number; end: number } {
  const clamped = Math.max(0, Math.min(text.length, caret));
  for (const span of rawTokenSpans(text)) {
    if (clamped >= span.start && clamped <= span.end) return span;
  }
  return { start: clamped, end: clamped };
}

function normalizedSearchName(value: string): string {
  return value.replace(/^-+/, "").toLowerCase();
}

function pendingCatalogValue(
  tokens: readonly string[],
  catalog: readonly LlamaServerArgument[],
  index: Map<string, LlamaServerArgument>,
): { argument: LlamaServerArgument; valueIndex: number } | null {
  let pending: {
    argument: LlamaServerArgument;
    valueIndex: number;
    remaining: number;
  } | null = null;
  for (const token of tokens) {
    if (looksLikeFlag(token)) {
      const resolved = resolveCatalogFlagToken(token, catalog, index);
      if (!resolved.argument || !takesValue(resolved.argument)) {
        pending = null;
        continue;
      }
      const attachedCount = resolved.attachedValue === undefined ? 0 : 1;
      const remaining = resolved.argument.value_arity - attachedCount;
      pending =
        remaining > 0
          ? {
              argument: resolved.argument,
              valueIndex: attachedCount,
              remaining,
            }
          : null;
      continue;
    }
    if (!pending) continue;
    pending.valueIndex += 1;
    pending.remaining -= 1;
    if (pending.remaining === 0) pending = null;
  }
  return pending
    ? { argument: pending.argument, valueIndex: pending.valueIndex }
    : null;
}

export function completeLlamaExtraArgs(
  text: string,
  caret: number,
  catalog: readonly LlamaServerArgument[],
): LlamaExtraArgsCompletion[] {
  const span = spanAtCaret(text, caret);
  const raw = text.slice(span.start, span.end);
  const before = parseLlamaExtraArgs(text.slice(0, span.start));
  const index = catalogIndex(catalog);

  const equals = raw.indexOf("=");
  if (equals > 0) {
    const argument = index.get(canonicalFlag(raw.slice(0, equals)));
    if (argument && takesValue(argument)) {
      const query = raw.slice(equals + 1).toLowerCase();
      return argument.choices
        .filter((choice) => choice.toLowerCase().startsWith(query))
        .slice(0, 8)
        .map((choice) => ({
          kind: "value" as const,
          insertText: choice,
          label: choice,
          argument,
          replaceStart: span.start + equals + 1,
          replaceEnd: span.end,
        }));
    }
  }

  const attached = resolveCatalogFlagToken(raw, catalog, index);
  if (
    attached.argument &&
    attached.separator === "attached" &&
    takesValue(attached.argument)
  ) {
    const query = (attached.attachedValue ?? "").toLowerCase();
    return attached.argument.choices
      .filter((choice) => choice.toLowerCase().startsWith(query))
      .slice(0, 8)
      .map((choice) => ({
        kind: "value" as const,
        insertText: choice,
        label: choice,
        argument: attached.argument as LlamaServerArgument,
        replaceStart: span.start + attached.rawFlag.length,
        replaceEnd: span.end,
      }));
  }

  const pending = pendingCatalogValue(before.tokens, catalog, index);
  if (pending) {
    if (raw.startsWith("-") || pending.valueIndex > 0) return [];
    const query = raw.toLowerCase();
    return pending.argument.choices
      .filter((choice) => choice.toLowerCase().startsWith(query))
      .slice(0, 8)
      .map((choice) => ({
        kind: "value" as const,
        insertText: choice,
        label: choice,
        argument: pending.argument,
        replaceStart: span.start,
        replaceEnd: span.end,
      }));
  }

  const query = normalizedSearchName(raw);
  if (!query || /\s/.test(raw)) return [];
  const ranked = catalog
    .filter((argument) => !argument.managed_by_studio)
    .map((argument, order) => {
      const names = [argument.name, ...argument.aliases];
      const matches = names.map((name, nameOrder) => {
        const normalized = normalizedSearchName(name);
        const rank =
          normalized === query
            ? 0
            : normalized.startsWith(query)
              ? 1
              : normalized.includes(query)
                ? 2
                : 3;
        return { name, nameOrder, rank };
      });
      const best = matches.sort(
        (a, b) => a.rank - b.rank || a.nameOrder - b.nameOrder,
      )[0];
      return {
        argument,
        order,
        rank: best.rank,
        spelling: best.name,
      };
    })
    .filter(({ rank }) => rank < 3)
    .sort((a, b) => a.rank - b.rank || a.order - b.order)
    .slice(0, 8);
  return ranked.map(({ argument, spelling }) => ({
    kind: "flag",
    insertText: spelling,
    label: spelling,
    argument,
    replaceStart: span.start,
    replaceEnd: span.end,
  }));
}

export function applyLlamaExtraArgsCompletion(
  text: string,
  completion: LlamaExtraArgsCompletion,
): { text: string; caret: number } {
  const hasSeparator = /\s/.test(text[completion.replaceEnd] ?? "");
  const suffix = hasSeparator ? "" : " ";
  const insertion = `${completion.insertText}${suffix}`;
  const next =
    text.slice(0, completion.replaceStart) +
    insertion +
    text.slice(completion.replaceEnd);
  return {
    text: next,
    caret: completion.replaceStart + insertion.length + (hasSeparator ? 1 : 0),
  };
}

export function countLlamaExtraArgFlags(
  tokens: readonly string[] | undefined,
): number {
  return (tokens ?? []).filter(looksLikeFlag).length;
}
