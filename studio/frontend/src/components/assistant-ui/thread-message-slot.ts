// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How the thread hands one message to a component, and which component that is.
 *
 * ThreadPrimitive.Messages rebuilds its element array whenever the message COUNT changes, so one
 * delete re-renders every remaining message's wrapper. assistant-ui absorbs that in
 * RenderChildrenWithAccessor: a render prop returning a PROPLESS element gets the same element
 * object back every render, and React skips that subtree. The `components={{...}}` form never
 * reaches the bail-out -- it returns `<ThreadMessageComponent components={...} />`, whose props
 * object is freshly allocated each time, so a delete re-renders every body, action bar and
 * tooltip in the thread.
 */

import { type ComponentType, type ReactElement, createElement } from "react";

export type ThreadMessageRole = "user" | "assistant" | "system";

/** Which of the thread's message components renders a message in this state. */
export type ThreadMessageKind = "edit" | "user" | "assistant" | "none";

/**
 * Pick the component for a message.
 *
 * assistant-ui's own getComponent fallback chain, resolved for the three components the thread
 * supplies (UserMessage, AssistantMessage, EditComposer) and no others:
 *
 *   - editing wins over role: every role's *EditComposer falls back to EditComposer, the only
 *     one supplied.
 *   - an unedited system message falls back SystemMessage -> Message -> a default that renders
 *     nothing, and neither is supplied.
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
 * Propless is what assistant-ui's bail-out requires; one shared instance also makes the bail-out
 * React's own, since React skips a child whose element is identical to the one it rendered. That
 * needs the element built once, here -- createElement per render returns a new object each time.
 */
export function proplessSlot(Component: ComponentType): () => ReactElement {
  const element = createElement(Component);
  return () => element;
}
