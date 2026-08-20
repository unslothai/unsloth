// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ReadSkillArgs = {
  name?: unknown;
  resource?: unknown;
};

export function readSkillToolDisplay(args: unknown): {
  actionLabel: string;
  toolName: string;
} {
  const parsed =
    typeof args === "object" && args !== null ? (args as ReadSkillArgs) : {};
  const name =
    typeof parsed.name === "string" && parsed.name.trim()
      ? parsed.name.trim()
      : "Agent skill";
  const resource =
    typeof parsed.resource === "string" ? parsed.resource.trim() : "";

  if (!resource || resource === "SKILL.md") {
    return { actionLabel: "Read skill instructions", toolName: name };
  }

  return {
    actionLabel: "Read skill resource",
    toolName: `${name} · ${resource}`,
  };
}
