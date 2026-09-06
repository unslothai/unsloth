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
  const marker = result.indexOf(TRUNCATION_FOOTER_MARKER);
  const core = marker === -1 ? result : result.slice(0, marker);
  if (!core || full === result || full.startsWith(core)) {
    return full;
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

/** Normalize both sources before deciding whether the live stream is fuller. */
export function preferSanitizedFullToolOutput(
  full: string,
  result: string,
): string {
  return preferFullToolOutput(stripAnsi(full), stripAnsi(result));
}
