// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ClaudeAgentStatus } from "./api/providers-api";

const SNAPSHOT_ID = /-\d{8}$/;
const OFF_MESSAGE = /^Claude Agent is off\b/i;

export function claudeAgentModelLabels(
  models: ClaudeAgentStatus["models"],
): string[] {
  const labels: string[] = [];
  for (const model of models) {
    const label =
      typeof model === "string"
        ? SNAPSHOT_ID.test(model)
          ? ""
          : model
        : SNAPSHOT_ID.test(model.id) && !model.display_name
          ? ""
          : (model.display_name || model.id || "").trim();
    if (label && !labels.includes(label)) {
      labels.push(label);
    }
  }
  return labels;
}

export function isClaudeAgentOffMessage(message: string): boolean {
  return OFF_MESSAGE.test(message.trim());
}
