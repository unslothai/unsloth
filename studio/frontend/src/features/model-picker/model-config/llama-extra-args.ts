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
  // Track "a token was opened" rather than keying on `current`, which would drop
  // a deliberate `--flag ""` and shift the argv by one.
  let started = false;
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < trimmed.length; i += 1) {
    const ch = trimmed[i];
    if (quote) {
      // Shell rules: only a quote or another backslash escapes, so a quoted
      // "C:\Program Files\t.jinja" survives as itself. Double quotes only, since
      // a single-quoted value is verbatim down to its backslashes.
      const next = i + 1 < trimmed.length ? trimmed[i + 1] : "";
      if (quote === '"' && ch === "\\" && (next === quote || next === "\\")) {
        current += next;
        i += 1;
        continue;
      }
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      started = true;
      continue;
    }
    if (/\s/.test(ch)) {
      if (started) {
        tokens.push(current);
        current = "";
        started = false;
      }
      continue;
    }
    current += ch;
    started = true;
  }
  if (started) {
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
    .map((token) => {
      // An empty argument only survives re-parsing as an explicit `""`.
      if (token === "") {
        return '""';
      }
      // Quote chars need quoting too, else the next blur re-parses `{"a":1}` as
      // `{a:1}`.
      if (!/[\s"']/.test(token)) {
        return token;
      }
      return `"${token.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
    })
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
    // Verbatim: the parser only emits blank or padded tokens the user quoted, so
    // rewriting would persist a different argv than loaded.
    out.push(raw);
  }
  return out;
}

/** undefined/null omits the field (inherit); [] clears inherited args on reload. */
export function llamaExtraArgsForLoad(
  args: string[] | null | undefined,
): string[] | undefined {
  if (args == null) {
    return undefined;
  }
  return args;
}
