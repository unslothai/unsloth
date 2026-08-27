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
 * The accumulated argument text of one streaming slot, cut into the top-level
 * JSON objects it holds.
 *
 * One tool call's `function.arguments` is exactly one JSON object, so a second
 * top-level `{` can only mean the stream reused a single index slot for a
 * second parallel call. That is what turns two calls into the unparsable
 * `{"url":"a"}{"url":"b"}`.
 *
 * `complete` holds the objects that have closed, in order, each one valid JSON
 * on its own. `tail` is the object still being written, if any. Anything the
 * scanner cannot read as a run of whole objects -- a top-level array or scalar,
 * trailing junk, an unbalanced brace -- is returned whole in `tail` with
 * `complete` empty, so a stream this was never meant for keeps its old
 * behaviour rather than being cut in a place that means nothing.
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
      // A backslash escapes exactly the next character, so a run of them
      // toggles rather than accumulates and `\\"` really does end the string.
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (depth === 0) {
      // Between objects only whitespace is allowed. Providers wrap in "\r\n"
      // as readily as "\n", and both are whitespace here. Anything else, a
      // quote included, means this is not a run of objects at all.
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
          // Balanced but not valid JSON, so the brace count was a coincidence
          // and cutting here would invent a call the model never made.
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
 * `{ _raw }` is the adapter's own marker for text it could not parse, not
 * something any tool declares, so replaying it would hand the tool a parameter
 * it has never heard of. Threads stored before arguments were split per call
 * still hold those, and they replay as `{}` rather than as a blob.
 */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (
    typeof argsText === "string" &&
    argsText.length > 0 &&
    // One JSON object, because that is what `function.arguments` is. A run of
    // them, an array, a bare scalar or a half-written object are all things a
    // strict chat template rejects for the whole request rather than for the
    // one call, so the structured args below stand in instead.
    isSingleJsonObject(argsText)
  ) {
    return argsText;
  }
  const serialized = JSON.stringify(args ?? {});
  // An array, a scalar, `null`, or something JSON.stringify refuses is not a
  // set of named parameters, whatever else it might be.
  if (serialized === undefined || !isSingleJsonObject(serialized)) {
    return "{}";
  }
  const keys = Object.keys(JSON.parse(serialized) as Record<string, unknown>);
  if (keys.length === 1 && keys[0] === "_raw") {
    return "{}";
  }
  return serialized;
}
