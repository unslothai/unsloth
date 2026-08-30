// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The `function.arguments` string to replay for a stored tool call.
 *
 * `argsText` is the text the provider streamed and is preferred so replay is
 * byte-exact. Text that does not parse would be replayed on every later request
 * in the thread, and strict chat templates reject the whole request rather than
 * one call, so it falls back to the structured args the part already carries.
 */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (typeof argsText === "string" && argsText.length > 0) {
    try {
      JSON.parse(argsText);
      return argsText;
    } catch {
      // unparsable, so the structured args below stand in for it
    }
  }
  return JSON.stringify(args ?? {});
}

/**
 * The text to render for a tool call's arguments.
 *
 * `JSON.parse` rounds an integer past 2**53 while reading the event, so re-encoding the
 * parsed arguments in the browser would show a value the tool is not being run with. The
 * backend sends its own encoding of the same arguments, which is preferred when present.
 */
export function toolCallArgumentsText(exactText: unknown, args: unknown): string {
  return typeof exactText === "string" && exactText.length > 0
    ? exactText
    : JSON.stringify(args ?? {});
}

/**
 * The text to render once a later event has had the chance to merge more arguments in.
 *
 * `merged` says whether it actually did. When nothing was merged the exact text still
 * describes the arguments and re-encoding would undo the precision it carries; once
 * something has been, it no longer describes them and the text has to be rebuilt.
 */
export function mergedToolCallArgumentsText(
  previousText: unknown,
  mergedArgs: unknown,
  merged: boolean,
): string {
  return merged
    ? JSON.stringify(mergedArgs ?? {})
    : toolCallArgumentsText(previousText, mergedArgs);
}
