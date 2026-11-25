// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MODEL_ERROR_SUMMARY_LIMIT = 240;

// Labels the backend's _with_startup_diagnostics appends after the human-readable message.
const DIAGNOSTICS_BOUNDARY = /\r?\n\r?\n(?:llama-server output:|Full log: )/;
const NEWLINE = /\r?\n/;

export function modelErrorParts(message: string): {
  summary: string;
  details: string | null;
} {
  const text = message.trim();
  const boundary = text.search(DIAGNOSTICS_BOUNDARY);
  const visible = boundary < 0 ? text : text.slice(0, boundary).trim();
  const lines = visible.split(NEWLINE);
  const compact =
    lines.length > 3 ? `${lines.slice(0, 3).join("\n")}…` : visible;
  const summary =
    compact.length > MODEL_ERROR_SUMMARY_LIMIT
      ? `${compact.slice(0, MODEL_ERROR_SUMMARY_LIMIT - 1).trimEnd()}…`
      : compact;
  return {
    summary: summary || "Model operation failed",
    details: boundary >= 0 || summary !== visible ? message : null,
  };
}
