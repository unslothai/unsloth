// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatArtifactsStore } from "../store";
import type { UnavailableReason } from "./compile-api";

export function needsNode(reason: UnavailableReason): boolean {
  return reason === "node_missing" || reason === "transform_missing";
}

/** Keeps the cards' "Needs Node.js" note in step with the last compile's answer. */
export function noteNodeAvailability(prepared: { status: string; reason?: UnavailableReason }): void {
  if (prepared.status === "unavailable" && !(prepared.reason && needsNode(prepared.reason))) return;
  useChatArtifactsStore.getState().setReactPreviewUnavailable(prepared.status === "unavailable");
}
