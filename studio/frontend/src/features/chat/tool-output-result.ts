// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { stringifyToolResult, stripAnsi } from "../../lib/strip-ansi";

// Footer the backend appends when it truncates a result to protect the context window (see
// backend tools._truncate). Marks where the result stops being a copy of the stream.
const TRUNCATION_FOOTER_MARKER = "\n\n... (truncated";

// Written only by the backend, after the footer, so swapping in the stream would drop it.
const TIMEOUT_STATUS_TAIL = /\nExecution timed out after \d+ seconds\.$/;

// Backend `_defuse_sentinels` indents these lines in the result but not the stream, and stripping
// ANSI can expose one at a line start, so both sides are normalized.
const SENTINEL_LINE = /^(__FILES__:|__IMAGES__:|__RAG_SOURCES__:)/gm;

function defuseSentinels(text: string): string {
  return text.replace(SENTINEL_LINE, " $1");
}

/** Whether the live stdout holds more real output than the model-visible `result` and should be
 *  preserved for the finished card. Shared by writer and reader so they agree. True when the
 *  result is truncated, OR the stream is longer. Truncation cannot fall back to length: a
 *  truncated result may be longer once its footer or exit line is appended. Also true when a
 *  short stream is absent from the result, since a cancelled tool returns only a status line. */
export function shouldPreserveFullOutput(rawFull: string, rawResult: string): boolean {
  const full = defuseSentinels(rawFull);
  const result = defuseSentinels(rawResult);
  if (!full) {
    return false;
  }
  if (result.includes(TRUNCATION_FOOTER_MARKER)) {
    return true;
  }
  if (full.length > result.length) {
    return true;
  }
  // Stream no longer than the result, but a timed-out or cancelled tool's status line never
  // echoes the captured stdout: preserve the stream whenever its content is absent from the
  // result, trimmed to ignore trailing-newline drift.
  const core = full.trim();
  return core.length > 0 && !result.includes(core);
}

/** Pick what a finished python/terminal card shows. Prefer the fuller live stream over the
 *  truncated `result`, but the result can carry failure or exit text that never reached
 *  stdout, so show the stream when the result is just a truncated prefix of it, else append
 *  the result so its status survives. */
export function preferFullToolOutput(rawFull: string, rawResult: string): string {
  const full = defuseSentinels(rawFull);
  const result = defuseSentinels(rawResult);
  if (!shouldPreserveFullOutput(full, result)) {
    return result;
  }
  const marker = result.indexOf(TRUNCATION_FOOTER_MARKER);
  const core = marker === -1 ? result : result.slice(0, marker);
  if (!core || full === result || full.startsWith(core)) {
    // The stream lacks the status the backend put after the footer.
    const status = result.match(TIMEOUT_STATUS_TAIL)?.[0];
    return status ? `${full.replace(/\s+$/, "")}\n\n${status.trimStart()}` : full;
  }
  // Failed executions prefix the result, not the stream, with "Exit code N:", so
  // `full.startsWith(core)` misses and a plain append would duplicate the stdout. Re-attach
  // just the exit prefix to the fuller stream so the status survives.
  const exitMatch = core.match(/^(Exit code -?\d+:\n)([\s\S]*)$/);
  if (exitMatch && full.startsWith(exitMatch[2])) {
    const hint = result.match(/\nHint:[\s\S]*$/)?.[0] ?? "";
    return `${exitMatch[1]}${full}${hint}`;
  }
  return `${full.replace(/\s+$/, "")}\n\n${result}`;
}

/** A tool result as card text; strings stay raw for `preferSanitizedFullToolOutput` to strip. */
export function toolResultText(result: unknown): string {
  return typeof result === "string" ? result : stringifyToolResult(result);
}

/** Normalize both sources before deciding whether the live stream is fuller. */
export function preferSanitizedFullToolOutput(
  full: string,
  result: string,
): string {
  // Output, footer and status are stripped apart, or an escape left open would swallow the rest.
  const status = result.match(TIMEOUT_STATUS_TAIL)?.[0] ?? "";
  const body = status ? result.slice(0, result.length - status.length) : result;
  const footer = body.lastIndexOf(TRUNCATION_FOOTER_MARKER);
  const cut = footer === -1 ? body.length : footer;
  return preferFullToolOutput(
    stripAnsi(full),
    stripAnsi(body.slice(0, cut)) + stripAnsi(body.slice(cut)) + status,
  );
}
