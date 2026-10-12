// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { decideAutoRead, useChatActive } from "@/features/chat";
import { useVoiceSettingsStore } from "@/features/settings";
import {
  type AssistantRuntime,
  useAui,
  useAuiEvent,
  useAuiState,
} from "@assistant-ui/react";
import { type FC, useCallback, useEffect, useRef } from "react";

/**
 * Speaks each finished reply through the same path as the Read aloud button, when the
 * Voice setting is on. Mount once per thread with a composer (not in Compare panes).
 */
export const AutoReadAloud: FC = () => {
  const aui = useAui();
  // False for the single chat kept mounted behind Compare or off-route.
  const active = useChatActive();
  const pendingRef = useRef(false);
  const lastSpokenRef = useRef<string | null>(null);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.id);
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
    if (decision.kind !== "speak" || !active) return;
    if (!useVoiceSettingsStore.getState().autoReadAloud) return;
    if (lastSpokenRef.current === decision.messageId) return;
    lastSpokenRef.current = decision.messageId;
    try {
      thread.message({ id: decision.messageId }).speak();
    } catch {
      // No speech adapter in this runtime, or the message was removed meanwhile.
    }
  }, [aui, active]);

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

  // Leaving the chat stops the reply this component started; the runtime keeps the old thread
  // alive, so it would otherwise play on under the next chat's reply.
  useEffect(() => {
    let bound: ReturnType<AssistantRuntime["threads"]["getById"]> | null = null;
    try {
      bound =
        aui
          .threads()
          .__internal_getAssistantRuntime?.()
          ?.threads.getById(threadId) ?? null;
    } catch {
      bound = null;
    }
    return () => {
      pendingRef.current = false;
      const spoken = lastSpokenRef.current;
      if (!bound || !spoken) return;
      try {
        if (bound.getState().speech?.messageId === spoken) bound.stopSpeaking();
      } catch {
        // The thread was removed, or playback already ended.
      }
    };
  }, [aui, threadId]);

  return null;
};
