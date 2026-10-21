// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The text of a `/chat?q=` link, the shape a browser search-engine shortcut sends. The router
 *  parses search values as JSON, so a query of only digits, or `true`, arrives as a number or a
 *  boolean and is turned back into its text. Anything else, or a blank query, is not a query. */
export function chatQueryParam(value: unknown): string | undefined {
  let text: string;
  if (typeof value === "string") {
    text = value;
  } else if (typeof value === "number" && Number.isFinite(value)) {
    text = String(value);
  } else if (typeof value === "boolean") {
    text = String(value);
  } else {
    return undefined;
  }
  return text.trim() ? text : undefined;
}
