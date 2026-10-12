// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { decideAutoRead } from "@/features/chat";
import { useVoiceSettingsStore } from "@/features/settings";
import { useAui, useAuiEvent, useAuiState } from "@assistant-ui/react";
import { type FC, useCallback, useEffect, useRef } from "react";

/** Speaks each finished reply through the same path as the Read aloud button, when the
 *  Voice setting is on. Mount once per thread with a composer (not in Compare panes). */
export const AutoReadAloud: FC = () => {
  const aui = useAui();
  const pendingRef = useRef(false);
  const lastSpokenRef = useRef<string | null>(null);
  const lastId = useAuiState(({ thread }) => thread.messages.at(-1)?.id);
  const lastStatus = useAuiState(
    ({ thread }) => thread.messages.at(-1)?.status?.type,
  );

  // runEnd can fire before the final message state lands, so a still-running reply is retried
  // from the effect below once its status settles.
  const resolvePending = useCallback(() => {
    if (!pendingRef.current) return;
    const thread = aui.thread();
    const decision = decideAutoRead(thread.getState().messages);
    if (decision.kind === "wait") return;
    pendingRef.current = false;
    if (decision.kind !== "speak") return;
    if (!useVoiceSettingsStore.getState().autoReadAloud) return;
    if (lastSpokenRef.current === decision.messageId) return;
    lastSpokenRef.current = decision.messageId;
    try {
      thread.message({ id: decision.messageId }).speak();
    } catch {
      // No speech adapter in this runtime, or the message was removed meanwhile.
    }
  }, [aui]);

  useAuiEvent("thread.runStart", () => {
    pendingRef.current = false;
    if (!useVoiceSettingsStore.getState().autoReadAloud) return;
    // A new turn makes the previous reply stale: stop it rather than talk over the next one.
    const thread = aui.thread();
    if (!thread.getState().speech) return;
    try {
      thread.stopSpeaking();
    } catch {
      // Playback ended between reading the state and stopping it.
    }
  });

  useAuiEvent("thread.runEnd", () => {
    pendingRef.current = true;
    resolvePending();
  });

  useEffect(() => {
    resolvePending();
  }, [lastId, lastStatus, resolvePending]);

  return null;
};
