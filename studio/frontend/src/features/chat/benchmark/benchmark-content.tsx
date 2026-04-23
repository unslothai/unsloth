// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
import { useEffect } from "react";
import type { CompareHandle } from "../shared-composer";

/**
 * Registers a CompareHandle into a ref provided by the benchmark runner.
 * Must be rendered INSIDE the visible SingleContent's ChatRuntimeProvider so
 * that handle.append() drives the thread the user is actually watching,
 * causing messages to stream live in the UI.
 */
export function RegisterBenchmarkHandle({
  setHandle,
}: {
  setHandle: (handle: CompareHandle | null) => void;
}): null {
  const aui = useAui();

  useEffect(() => {
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
       * Polls the AUI thread state directly to detect run completion.
       * This is more reliable than subscribing to useChatRuntimeStore because
       * the AUI thread state updates synchronously, eliminating race conditions
       * where the run completes before a Zustand subscription fires.
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
          // Allow ~150ms for the AUI thread to start the run before we begin polling
          setTimeout(poll, 150);
        }),
    };
    setHandle(handle);
    return () => setHandle(null);
  }, [aui, setHandle]);

  return null;
}
