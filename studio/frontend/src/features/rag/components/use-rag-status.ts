// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { getRagStatus } from "@/features/rag/api/rag-api";

const RAG_STATUS_POLL_MS = 2000;

/**
 * Polls `GET /api/rag/status` into the store while `active` and RAG is not yet
 * available, surfacing a "setting up RAG" state; the poll itself triggers the
 * background install. Visibility-aware; stops once available.
 */
export function useRagStatus(active: boolean): void {
  const setRagStatus = useChatRuntimeStore((s) => s.setRagStatus);
  const available = useChatRuntimeStore((s) => s.ragAvailable);

  useEffect(() => {
    if (!active || available === true) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await getRagStatus();
        if (!cancelled) setRagStatus(s);
      } catch {
        // Transient network/auth error; the next tick retries.
      }
    };

    void poll();
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") void poll();
    }, RAG_STATUS_POLL_MS);
    const onVisibility = () => {
      if (document.visibilityState === "visible") void poll();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [active, available, setRagStatus]);
}
