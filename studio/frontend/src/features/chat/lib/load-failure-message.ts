// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function loadFailureToastParts(message: string): {
  summary: string;
  detail: string;
} {
  let visible = message;
  for (const marker of ["\n\nllama-server output:", "\n\nFull log: "]) {
    const at = visible.indexOf(marker);
    if (at !== -1) {
      visible = visible.slice(0, at);
    }
  }
  const [summary, ...rest] = visible.trim().split("\n");
  return { summary, detail: rest.join("\n").trim() };
}
