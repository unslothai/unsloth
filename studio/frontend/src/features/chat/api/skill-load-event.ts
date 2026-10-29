// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** UI-only card for a backend skill load; excluded from model history, provider frames sanitized. */
export function skillLoadCardEvent(
  event: Record<string, unknown>,
): Record<string, unknown> {
  const pending =
    event.status === "loading" || event.status === "awaiting_approval";
  return {
    type: pending ? "tool_start" : "tool_end",
    tool_name: "studio_load_skill",
    tool_call_id: event.load_id,
    arguments: {
      name: event.name,
      resource: "SKILL.md",
      _studio_skill_load: true,
    },
    ...(event.status === "awaiting_approval"
      ? { awaiting_confirmation: true, approval_id: event.approval_id }
      : {}),
    ...(pending
      ? {}
      : { result: event.detail ?? "Skill loading unavailable." }),
  };
}
