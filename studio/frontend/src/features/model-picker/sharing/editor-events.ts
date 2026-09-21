// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE = "data-shared-run-config";

export function keepSharedRunConfigOpen(content: ParentNode | null): boolean {
  return Boolean(
    content?.querySelector(`[${SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE}]`),
  );
}

export function isRunConfigEditorChange(event: {
  currentTarget: Node;
  target: EventTarget;
}): boolean {
  return (
    event.target instanceof Node && event.currentTarget.contains(event.target)
  );
}
