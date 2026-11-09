// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A render prop returning one PROPLESS element lets React skip unchanged rows; the
 * `components={{...}}` form allocates props per row, so a delete re-renders every message.
 */

import { type ComponentType, type ReactElement, createElement } from "react";

export type ThreadMessageRole = "user" | "assistant" | "system";

export type ThreadMessageKind = "edit" | "user" | "assistant" | "none";

/**
 * Mirrors assistant-ui's getComponent fallback for the three components the thread supplies:
 * editing wins over role, and an unedited system message renders nothing.
 */
export function threadMessageKind(
  role: ThreadMessageRole,
  isEditing: boolean,
): ThreadMessageKind {
  if (isEditing) {
    return "edit";
  }
  if (role === "user") {
    return "user";
  }
  if (role === "assistant") {
    return "assistant";
  }
  return "none";
}

/** Derived from `threadMessageKind` so the two cannot drift; a "none" message has no height. */
export function rendersAsRow(
  role: ThreadMessageRole,
  isEditing: boolean,
): boolean {
  return threadMessageKind(role, isEditing) !== "none";
}

/** Built once: an identical element object is what lets React bail out. */
export function proplessSlot(Component: ComponentType): () => ReactElement {
  const element = createElement(Component);
  return () => element;
}
