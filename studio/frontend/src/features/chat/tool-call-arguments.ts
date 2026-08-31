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
 * One slot's accumulated argument text, cut into the top-level JSON objects it
 * holds: `complete` closed, `tail` is still being written.
 *
 * A second top-level `{` means the stream reused the slot for a second parallel
 * call, which is what turns two calls into `{"url":"a"}{"url":"b"}`. Text that
 * is not a run of whole objects comes back whole in `tail`, so a stream this
 * was never meant for is left alone.
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
      // A backslash escapes one character, so a run of them toggles.
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (depth === 0) {
      // Between objects only whitespace, "\r\n" as readily as "\n".
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
          // Balanced but invalid: cutting here would invent a call.
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
 * `splitTopLevelJsonObjects` over a string that only ever grows.
 *
 * Rescanning per fragment is O(N^2), and a 20 KB argument sent a character at a
 * time took about a second and a half on the thread that paints the stream.
 * `feed` takes the same string extended, never a rewritten one, so a caller
 * that splits a slot drops its scan.
 */
export function createBoundaryScan(): {
  feed: (text: string) => { complete: string[]; tail: string };
} {
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  let scanned = 0;
  // Once unsplittable, appending can never make it splittable again.
  let unsplittable = false;
  const complete: string[] = [];

  return {
    feed(text: string) {
      if (unsplittable) return { complete: [], tail: text };
      for (let i = scanned; i < text.length; i += 1) {
        const ch = text[i];
        if (inString) {
          if (escaped) escaped = false;
          else if (ch === "\\") escaped = true;
          else if (ch === '"') inString = false;
          continue;
        }
        if (depth === 0) {
          if (ch === "{") {
            depth = 1;
            start = i;
            continue;
          }
          if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") continue;
          unsplittable = true;
          return { complete: [], tail: text };
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
              unsplittable = true;
              return { complete: [], tail: text };
            }
            complete.push(segment);
            start = -1;
          }
        }
      }
      scanned = text.length;
      return {
        complete: [...complete],
        tail: start === -1 ? "" : text.slice(start),
      };
    },
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
 * `{ _raw }` is the adapter's marker for text it could not parse, so a thread
 * carrying one replays as `{}` rather than as the blob.
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
  const parsed = JSON.parse(serialized) as Record<string, unknown>;
  const keys = Object.keys(parsed);
  // The adapter writes `{ _raw }` holding the exact text it could not parse, so
  // a lone `_raw` whose value IS that text is the marker and replaying it would
  // send a parameter no tool declares. `_raw` is not reserved, though, and an
  // MCP server's schema is its own, so a tool that really takes one keeps it.
  //
  // Only when the surviving text proves it. A thread stored before `argsText`
  // was kept carries the marker with nothing to compare it to, and guessing
  // from the shape of the value -- a run of whole JSON objects -- would also
  // discard the argument of a tool that really takes one, since a leading
  // underscore is a legal property name and MCP reserves nothing. Guessing
  // buys little either way: the wrapped form is one JSON object, so replaying
  // it does not raise the `Extra data` this file exists to prevent.
  if (
    keys.length === 1 &&
    keys[0] === "_raw" &&
    typeof parsed._raw === "string" &&
    // Non-empty, or the equality proves nothing: the adapter only writes the
    // marker for text it tried to parse, so `{ _raw: "" }` is an argument a
    // tool was really given rather than anything this file produced.
    parsed._raw.length > 0 &&
    parsed._raw === argsText
  ) {
    return "{}";
  }
  return serialized;
}
