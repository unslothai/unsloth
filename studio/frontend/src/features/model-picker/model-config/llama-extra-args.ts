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
  // A token exists once one is opened, whatever it ends up holding. Keying on
  // `current` instead would swallow a deliberate empty argument (`--flag ""`),
  // and dropping it shifts the argv: llama-server then reads the NEXT token as
  // the flag's value and silently ignores it.
  let started = false;
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
    // Store every string token verbatim. The parser only emits a blank or
    // edge-padded token when the user quoted one, so rewriting or dropping it
    // here would persist a different argv than the same config just loaded with.
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
