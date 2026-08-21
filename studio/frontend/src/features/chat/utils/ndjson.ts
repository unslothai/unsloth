// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** JSONL body with every record newline-terminated, so exported files
 *  concatenate cleanly and line readers see a complete final record. */
export function ndjsonBody(records: readonly string[]): string {
  return records.length > 0 ? `${records.join("\n")}\n` : "";
}
