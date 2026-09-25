// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const BIDI_CONTROL_RE = /[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]/g;

/** Show bidi controls as `\uXXXX` so a url cannot display reordered. Display only. */
export function escapeBidiControls(value: string): string {
  return value.replace(
    BIDI_CONTROL_RE,
    (control) =>
      `\\u${(control.codePointAt(0) ?? 0).toString(16).padStart(4, "0")}`,
  );
}
