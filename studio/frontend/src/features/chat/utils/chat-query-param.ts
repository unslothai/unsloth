// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Normalize URL text or typed router input; a blank value is not a query. */
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
