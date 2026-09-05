// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ToolExecutionMode } from "../tool-isolation";

/** The part of the isolation decision a queued send carries and the live store must still
 *  match right before dispatch. */
export type IsolationDecision = {
  toolExecutionMode: ToolExecutionMode;
  toolIsolationUiSessionId: string;
};

/** A send prepared under one decision goes out only if the live store still holds the same
 *  mode under the same UI session. The session id rotates on every authentication change
 *  (clearToolIsolationGrantForAuthSession), so a Full decision made before a rotation is not
 *  honoured because a later session independently chose Full again. */
export function queuedIsolationDecisionIsCurrent(
  snapshot: IsolationDecision,
  live: IsolationDecision,
): boolean {
  return (
    live.toolExecutionMode === snapshot.toolExecutionMode &&
    live.toolIsolationUiSessionId === snapshot.toolIsolationUiSessionId
  );
}
