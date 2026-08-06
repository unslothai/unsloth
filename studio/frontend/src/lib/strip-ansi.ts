// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// CSI / Fe escape sequences as emitted by colourised CLI tools (ls, grep, npm,
// cargo, pytest, …). Built via fromCharCode so the source stays free of a
// literal ESC that some editors / log scrapers trip over.
const ANSI_ESCAPE_PATTERN = new RegExp(
  `${String.fromCharCode(27)}(?:[@-Z\\-_]|\\[[0-?]*[ -/]*[@-~])`,
  "g",
);

/** Strip SGR / CSI sequences so tool output is readable in a plain <pre>. */
export function stripAnsi(text: string): string {
  return text.replace(ANSI_ESCAPE_PATTERN, "");
}
