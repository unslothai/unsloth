// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { stringifyToolResult, stripAnsi } from "../../lib/strip-ansi";

// Footer added by backend tools._truncate; marks where the result stops copying the stream.
const TRUNCATION_FOOTER_MARKER = "\n\n... (truncated";

const TIMEOUT_STATUS_TAIL = /\nExecution timed out after \d+ seconds\.$/;

// Backend `_defuse_sentinels` indents these in the result but not the stream; normalize both.
const SENTINEL_LINE = /^(__FILES__:|__IMAGES__:|__RAG_SOURCES__:)/gm;

function defuseSentinels(text: string): string {
  return text.replace(SENTINEL_LINE, " $1");
}

/** Shared by writer and reader. Truncation cannot fall back to length; cancelled tools return
 *  only a status line, so an absent short stream also counts. */
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
  const core = full.trim();
  return core.length > 0 && !result.includes(core);
}

export function preferFullToolOutput(rawFull: string, rawResult: string): string {
  const full = defuseSentinels(rawFull);
  const result = defuseSentinels(rawResult);
  if (!shouldPreserveFullOutput(full, result)) {
    return result;
  }
  const marker = result.indexOf(TRUNCATION_FOOTER_MARKER);
  const core = marker === -1 ? result : result.slice(0, marker);
  if (!core || full === result || full.startsWith(core)) {
    const status = result.match(TIMEOUT_STATUS_TAIL)?.[0];
    return status ? `${full.replace(/\s+$/, "")}\n\n${status.trimStart()}` : full;
  }
  // Failed runs prefix only the result with "Exit code N:", so reattach that prefix to the stream.
  const exitMatch = core.match(/^(Exit code -?\d+:\n)([\s\S]*)$/);
  if (exitMatch && full.startsWith(exitMatch[2])) {
    const hint = result.match(/\nHint:[\s\S]*$/)?.[0] ?? "";
    return `${exitMatch[1]}${full}${hint}`;
  }
  return `${full.replace(/\s+$/, "")}\n\n${result}`;
}

export function toolResultText(result: unknown): string {
  return typeof result === "string" ? result : stringifyToolResult(result);
}

export function preferSanitizedFullToolOutput(
  full: string,
  result: string,
): string {
  // Strip output, footer and status separately, or an open escape swallows the rest.
  const status = result.match(TIMEOUT_STATUS_TAIL)?.[0] ?? "";
  const body = status ? result.slice(0, result.length - status.length) : result;
  const footer = body.lastIndexOf(TRUNCATION_FOOTER_MARKER);
  const cut = footer === -1 ? body.length : footer;
  return preferFullToolOutput(
    stripAnsi(full),
    stripAnsi(body.slice(0, cut)) + stripAnsi(body.slice(cut)) + status,
  );
}
