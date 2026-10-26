// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** No imports, so a stored value can be tested without the runtime store. */
export type ProjectAttachmentTarget = "project" | "thread";

export const CHAT_PROJECT_ATTACHMENT_TARGET_KEY =
  "unsloth_chat_project_attachment_target";

export const DEFAULT_PROJECT_ATTACHMENT_TARGET: ProjectAttachmentTarget = "project";

/** Unknown values from a later build map to the narrow scope, which is the safe reading. */
export function normalizeProjectAttachmentTarget(
  raw: string | null | undefined,
): ProjectAttachmentTarget {
  if (raw === "project" || raw === "thread") return raw;
  return raw ? "thread" : DEFAULT_PROJECT_ATTACHMENT_TARGET;
}
