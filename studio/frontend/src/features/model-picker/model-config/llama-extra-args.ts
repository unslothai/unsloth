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
      // Only a quote or another backslash escapes, as in a shell. Anything
      // else keeps its backslash, so a quoted Windows path survives
      // ("C:\Program Files\t.jinja", not "C:Program Filest.jinja").
      const next = i + 1 < trimmed.length ? trimmed[i + 1] : "";
      if (ch === "\\" && (next === quote || next === "\\")) {
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
    .map((token) => {
      // Quote chars need quoting too, else re-parsing the field strips them
      // (`{"a":1}` -> `{a:1}`) and silently corrupts the value on the next blur.
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
    // Validate on the trimmed form but store the token verbatim. Only explicit
    // quoting can give a token edge whitespace, and trimming it would persist a
    // different argv than the same config already loaded with.
    if (!raw.trim()) {
      continue;
    }
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
