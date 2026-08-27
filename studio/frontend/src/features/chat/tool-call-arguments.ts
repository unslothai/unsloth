// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `true` when `text` is exactly one JSON object and nothing else. */
function isSingleJsonObject(text: string): boolean {
  try {
    const value = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value);
  } catch {
    return false;
  }
}

/**
 * One streaming slot's accumulated argument text, cut into the top-level JSON
 * objects it holds: `complete` are the ones that closed, `tail` is the one
 * still being written.
 *
 * A call's `function.arguments` is one JSON object, so a second top-level `{`
 * means the stream reused this slot for a second parallel call, which is what
 * turns two calls into the unparsable `{"url":"a"}{"url":"b"}`.
 *
 * Text that is not a run of whole objects (top-level array or scalar, trailing
 * junk, an unbalanced brace) comes back whole in `tail` with `complete` empty,
 * so a stream this was never meant for is left alone.
 */
export function splitTopLevelJsonObjects(text: string): {
  complete: string[];
  tail: string;
} {
  const unsplit = { complete: [] as string[], tail: text };
  const complete: string[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      // A backslash escapes one character, so a run of them toggles rather
      // than accumulates and `\\"` really does end the string.
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (depth === 0) {
      // Between objects only whitespace, "\r\n" as readily as "\n". Anything
      // else, a quote included, means this is not a run of objects at all.
      if (ch === "{") {
        depth = 1;
        start = i;
        continue;
      }
      if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") continue;
      return unsplit;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const segment = text.slice(start, i + 1);
        try {
          JSON.parse(segment);
        } catch {
          // Balanced but invalid, so the brace count was a coincidence and
          // cutting here would invent a call the model never made.
          return unsplit;
        }
        complete.push(segment);
        start = -1;
      }
    }
  }

  return {
    complete,
    tail: start === -1 ? "" : text.slice(start),
  };
}

/**
 * The `function.arguments` string to replay for a stored tool call.
 *
 * `argsText` is the text the provider streamed and is preferred so replay is
 * byte-exact. Text that does not parse would be replayed on every later request
 * in the thread, and strict chat templates reject the whole request rather than
 * one call, so it falls back to the structured args the part already carries.
 *
 * `{ _raw }` is the adapter's own marker for text it could not parse, not a
 * parameter any tool declares, so threads stored before arguments were split
 * per call replay as `{}` rather than as a blob.
 */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (
    typeof argsText === "string" &&
    argsText.length > 0 &&
    // One object, because that is what `function.arguments` is. A run of them,
    // an array, a scalar or a half-written object all get the whole request
    // rejected, not just the one call.
    isSingleJsonObject(argsText)
  ) {
    return argsText;
  }
  const serialized = JSON.stringify(args ?? {});
  // Not a set of named parameters, whatever else it might be.
  if (serialized === undefined || !isSingleJsonObject(serialized)) {
    return "{}";
  }
  const keys = Object.keys(JSON.parse(serialized) as Record<string, unknown>);
  if (keys.length === 1 && keys[0] === "_raw") {
    return "{}";
  }
  return serialized;
}
