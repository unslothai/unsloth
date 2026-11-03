// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The reload hint reads Chat's runtime store, which only Chat's hooks fill; a flow opened directly
// (or a run that loaded another model) leaves it stale. Lazy so the editor's chunk stays small.
export function syncResidentModel(): void {
  void import("@/features/chat/hooks/use-chat-model-runtime")
    .then((m) =>
      m.syncInferenceStatusToStore({ includeLoras: false, quiet: true }),
    )
    .catch(() => undefined);
}
