// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat's runtime store is only filled by Chat's hooks, so sync it here; lazy to keep the editor chunk small.
export function syncResidentModel(): void {
  void import("@/features/chat/hooks/use-chat-model-runtime")
    .then((m) =>
      m.syncInferenceStatusToStore({ includeLoras: false, quiet: true }),
    )
    .catch(() => undefined);
}
