// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Parse a shell-style flag string into llama-server argv tokens. */
export function parseLlamaExtraArgsInput(input: string): string[] {
  const trimmed = input.trim();
  if (!trimmed) {
    return [];
  }
  const tokens: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < trimmed.length; i += 1) {
    const ch = trimmed[i];
    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }
    current += ch;
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}

export function formatLlamaExtraArgs(
  args: string[] | null | undefined,
): string {
  if (!args?.length) {
    return "";
  }
  return args
    .map((token) => (/\s/.test(token) ? `"${token}"` : token))
    .join(" ");
}

export function normalizeLlamaExtraArgs(
  value: unknown,
): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const out: string[] = [];
  for (const raw of value) {
    if (typeof raw !== "string") {
      continue;
    }
    const token = raw.trim();
    if (!token) {
      continue;
    }
    out.push(token);
  }
  return out.length > 0 ? out : undefined;
}

/** Omit empty lists so /load inherits; send only when the user configured flags. */
export function llamaExtraArgsForLoad(
  args: string[] | null | undefined,
): string[] | undefined {
  if (!args?.length) {
    return undefined;
  }
  return args;
}
