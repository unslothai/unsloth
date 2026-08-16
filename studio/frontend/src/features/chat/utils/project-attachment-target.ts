// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Where a file attached in a project chat is indexed. `project` puts it into the
 * project's shared sources; `thread` keeps the file to the one chat. Ignored outside
 * a project.
 *
 * Its own module, with no imports, so the value read out of a browser profile can be
 * tested without pulling the whole runtime store in behind it. */
export type ProjectAttachmentTarget = "project" | "thread";

export const CHAT_PROJECT_ATTACHMENT_TARGET_KEY =
  "unsloth_chat_project_attachment_target";

// A project exists to share context, so its chats default to the whole project.
export const DEFAULT_PROJECT_ATTACHMENT_TARGET: ProjectAttachmentTarget = "project";

/** The target a stored value means, for any string a profile might hold.
 *
 * Nothing stored is an install that has never chosen, which gets the default. A value
 * this build does not know came from a later one, and the safe reading of an unknown
 * preference is the narrow scope: filing into the chat cannot put a file somewhere the
 * user did not mean, and filing into the project can. */
export function normalizeProjectAttachmentTarget(
  raw: string | null | undefined,
): ProjectAttachmentTarget {
  if (raw === "project" || raw === "thread") return raw;
  return raw ? "thread" : DEFAULT_PROJECT_ATTACHMENT_TARGET;
}
