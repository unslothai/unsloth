// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui, useAuiState } from "@assistant-ui/react";
import { useEffect } from "react";
import type { CompareHandle } from "../shared-composer";

/**
 * Registers a CompareHandle into a ref provided by the benchmark runner.
 * Must be rendered INSIDE the visible SingleContent's ChatRuntimeProvider so
 * that handle.append() drives the thread the user is actually watching,
 * causing messages to stream live in the UI.
 *
 * When `threadId` is provided the handle is NOT registered until:
 *   1. The AUI runtime has finished loading its thread list (!isLoading), AND
 *   2. AUI's mainThreadId matches threadId (ThreadAutoSwitch completed).
 *
 * Using AUI's own state avoids any race between the AUI thread switch and
 * the Zustand activeThreadId update that a Zustand subscription would have.
 */
export function RegisterBenchmarkHandle({
  setHandle,
  threadId,
}: {
  setHandle: (handle: CompareHandle | null) => void;
  threadId?: string;
}): null {
  const aui = useAui();
  // Mirror the same signals ThreadAutoSwitch uses so we know exactly when
  // AUI has finished switching to the target thread.
  const isLoading = useAuiState(({ threads }) => threads.isLoading);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);

  useEffect(() => {
    // Wait until the AUI runtime has loaded and switched to the target thread.
    if (isLoading) return;
    if (threadId && mainThreadId !== threadId) return;

    const handle: CompareHandle = {
      append: (content) =>
        aui.thread().append({ role: "user", content, createdAt: new Date() } as never),
      appendMessage: (content) =>
        aui.thread().append({ role: "user", content, createdAt: new Date(), startRun: false } as never),
      startRun: () => {
        const msgs = aui.thread().getState().messages;
        const lastId = msgs.length > 0 ? msgs[msgs.length - 1].id : null;
        aui.thread().startRun({ parentId: lastId });
      },
      cancel: () => aui.thread().cancelRun(),
      isRunning: () => aui.thread().getState().isRunning,
      /**
       * Polls AUI thread state directly to detect run completion.
       * Reliable because AUI state updates synchronously.
       */
      waitForRunEnd: () =>
        new Promise<void>((resolve) => {
          const startTime = Date.now();
          let wasRunning = false;
          const poll = () => {
            const running = aui.thread().getState().isRunning;
            if (running) wasRunning = true;
            if (wasRunning && !running) {
              resolve();
              return;
            }
            // Safety timeout: if the run never starts within 10s, continue
            if (!wasRunning && Date.now() - startTime > 10_000) {
              resolve();
              return;
            }
            setTimeout(poll, 100);
          };
          // Allow ~150ms for the AUI thread to start the run before polling
          setTimeout(poll, 150);
        }),
    };

    setHandle(handle);
    return () => setHandle(null);
  }, [aui, setHandle, threadId, mainThreadId, isLoading]);

  return null;
}
