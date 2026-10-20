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

  // runEnd can fire before the final status lands; the effect below retries then.
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
      // No speech adapter, or the message is gone.
    }
  }, [aui, active]);

  useAuiEvent("thread.runStart", () => {
    pendingRef.current = false;
    if (!useVoiceSettingsStore.getState().autoReadAloud) return;
    const thread = aui.thread();
    if (!thread.getState().speech) return;
    try {
      thread.stopSpeaking();
    } catch {
      // Already ended.
    }
  });

  useAuiEvent("thread.runEnd", () => {
    pendingRef.current = true;
    resolvePending();
  });

  useEffect(() => {
    resolvePending();
  }, [lastId, lastStatus, resolvePending]);

  // The old thread's runtime stays alive, so its playback would overlap the next chat's.
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
        // Thread removed or already ended.
      }
    };
  }, [aui, threadId]);

  return null;
};
