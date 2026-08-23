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
 * Whether a message in this state paints anything.
 *
 * Derived from `threadMessageKind` rather than from its own list of roles: a second list would
 * drift from the renderer the first time a role is added, and the drift is silent in both
 * directions. Anything sizing itself on what the reader can SEE, rather than on how many messages
 * there are, has to ask this -- a "none" message occupies no height, so counting it as a row makes
 * a thread open on blanks. See progressive-mount-controller.ts.
 */
export function rendersAsRow(
  role: ThreadMessageRole,
  isEditing: boolean,
): boolean {
  return threadMessageKind(role, isEditing) !== "none";
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
