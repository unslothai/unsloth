// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a finished tool card shows, decided from the captured stream and the model-visible result.
// Pure functions with no runtime imports on purpose: the recovery replay (chat-generation-replay)
// folds persisted `tool_output`/`tool_end` frames into parts under node --test, where importing
// the scope module would drag in @assistant-ui/react. `tool-output-scope` re-exports these for
// its React consumers so writer and reader keep reading one source.

import { stripAnsi } from "../../../lib/strip-ansi";

// Footer the backend appends when it truncates a result to protect the context window (see
// backend tools._truncate). Marks where the result stops being a copy of the stream.
const TRUNCATION_FOOTER_MARKER = "\n\n... (truncated";

/** What the backend appends AFTER the body (and its footer): the created-files/images envelope, appended last,
 *  so older clients slicing from a marker to end of string still parse it. The stream never carried one byte of
 *  it -- it exists only so the card can render downloads and images out of the same string -- so a selection
 *  that swaps the fuller body IN must carry it along, or the shaper has no envelope left to split and the
 *  reopened card loses its files and images. Anchored on exactly what `extractCreatedFiles` and the result
 *  shaper re-split on; output a tool printed itself never matches (the backend defuses stray markers). */
const ENVELOPE_MARKERS = ["\n__FILES__:", "\n__MCP_IMAGES__:", "\n__IMAGES__:"];

/** Whether the live stdout holds more real output than the model-visible `result` and should be
 *  preserved for the finished card. Shared by writer and reader so they agree. True when the
 *  result is truncated, OR the stream is longer. Truncation cannot fall back to length: a
 *  truncated result may be longer once its footer or exit line is appended. Also true when a
 *  short stream is absent from the result, since a cancelled tool returns only a status line. */
export function shouldPreserveFullOutput(full: string, result: string): boolean {
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
export function preferFullToolOutput(full: string, result: string): string {
  if (!shouldPreserveFullOutput(full, result)) {
    return result;
  }
  // Split the envelope off the body it rides on BEFORE deciding about the body: every branch below replaces
  // some part of the model-visible result with the fuller stream, and a suffix that only exists for the card's
  // sake must survive whichever one of them fires. The earliest marker wins; IMAGES sits last when both ride,
  // so this lands on FILES there, exactly where the shaper would split anyway.
  let envelopeAt = -1;
  for (const envelopeMarker of ENVELOPE_MARKERS) {
    const at = result.indexOf(envelopeMarker);
    if (at !== -1 && (envelopeAt === -1 || at < envelopeAt)) envelopeAt = at;
  }
  const body = envelopeAt === -1 ? result : result.slice(0, envelopeAt);
  const envelope = envelopeAt === -1 ? "" : result.slice(envelopeAt);
  const marker = body.indexOf(TRUNCATION_FOOTER_MARKER);
  const core = marker === -1 ? body : body.slice(0, marker);
  if (!core || full === result || full.startsWith(core)) {
    return envelope && !full.endsWith(envelope) ? `${full}${envelope}` : full;
  }
  // Failed executions prefix the result, not the stream, with "Exit code N:", so
  // `full.startsWith(core)` misses and a plain append would duplicate the stdout. Re-attach
  // just the exit prefix to the fuller stream so the status survives.
  const exitMatch = core.match(/^(Exit code -?\d+:\n)([\s\S]*)$/);
  if (exitMatch && full.startsWith(exitMatch[2])) {
    // Matched against the body, not the whole result: the hint ends where the envelope begins, and the
    // envelope rides along on its own now instead of riding inside the hint match to the end of string.
    const hint = body.match(/\nHint:[\s\S]*$/)?.[0] ?? "";
    return `${exitMatch[1]}${full}${hint}${envelope}`;
  }
  return `${full.replace(/\s+$/, "")}\n\n${result}`;
}

/** Normalize both sources before deciding whether the live stream is fuller. */
export function preferSanitizedFullToolOutput(
  full: string,
  result: string,
): string {
  return preferFullToolOutput(stripAnsi(full), stripAnsi(result));
}
