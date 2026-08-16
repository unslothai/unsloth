// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How the thread hands one message to a component, and which component that is.
 *
 * ThreadPrimitive.Messages rebuilds its whole element array whenever the message COUNT changes,
 * so deleting one message re-renders the wrapper of every remaining message. assistant-ui absorbs
 * that with a bail-out in RenderChildrenWithAccessor: when the render prop returns an element
 * with no props, the same element object is handed back on every render and React skips
 * reconciling that subtree. The `components={{...}}` form never reaches the bail-out, because
 * what it returns is `<ThreadMessageComponent components={...} />`, whose props object is freshly
 * allocated every time -- so one delete re-renders every message body, action bar and tooltip in
 * the thread.
 */

import { type ComponentType, type ReactElement, createElement } from "react";

export type ThreadMessageRole = "user" | "assistant" | "system";

/** Which of the thread's message components renders a message in this state. */
export type ThreadMessageKind = "edit" | "user" | "assistant" | "none";

/**
 * Pick the component for a message.
 *
 * This is assistant-ui's own getComponent fallback chain, resolved for the three components the
 * thread supplies (UserMessage, AssistantMessage, EditComposer) and no others:
 *
 *   - editing wins over role. A user message falls back UserEditComposer -> EditComposer, an
 *     assistant message AssistantEditComposer -> EditComposer, a system message
 *     SystemEditComposer -> EditComposer, and only EditComposer is supplied.
 *   - a system message that is not being edited falls back SystemMessage -> Message -> a default
 *     that renders nothing, and neither SystemMessage nor Message is supplied.
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

/**
 * A render prop that always returns the SAME propless element for `Component`.
 *
 * Propless is what assistant-ui's bail-out requires; returning one shared instance also makes the
 * bail-out React's own, since React skips a child whose element is identical to the rendered one
 * regardless of what the library memoizes. Building the element once, here, is what makes that
 * true -- an arrow that calls createElement per render would hand back a new object every time.
 */
export function proplessSlot(Component: ComponentType): () => ReactElement {
  const element = createElement(Component);
  return () => element;
}
