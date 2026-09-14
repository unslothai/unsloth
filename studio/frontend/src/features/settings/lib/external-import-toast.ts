// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the Settings import row should toast after a Cursor or Claude Code run. */

export type ExternalImportToastCopy = {
  none: string;
  upToDate: string;
  one: string;
  many: string;
  partial: string;
};

export type ExternalImportToastResult = {
  chats: number;
  newChats: number;
  warnings: string[];
};

export type ExternalImportToast = {
  kind: "success" | "warning";
  title: string;
  description?: string;
};

export function describeExternalImportToast(
  result: ExternalImportToastResult,
  copy: ExternalImportToastCopy,
): ExternalImportToast {
  if (result.warnings.length > 0) {
    return {
      kind: "warning",
      title: copy.partial,
      description: result.warnings.join("\n"),
    };
  }
  const title =
    result.chats === 0
      ? copy.none
      : result.newChats === 0
        ? copy.upToDate
        : result.newChats === 1
          ? copy.one
          : copy.many;
  return { kind: "success", title };
}
