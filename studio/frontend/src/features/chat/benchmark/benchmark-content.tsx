// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
import { useEffect } from "react";
import type { CompareHandle } from "../shared-composer";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/**
 * Registers a CompareHandle into a ref provided by the benchmark runner.
 * Must be rendered INSIDE the visible SingleContent's ChatRuntimeProvider so
 * that handle.append() drives the thread the user is actually watching,
 * causing messages to stream live in the UI.
 *
 * When `threadId` is provided the handle is NOT registered until
 * ThreadAutoSwitch has completed and `useChatRuntimeStore.activeThreadId`
 * matches. This prevents appending to a stale / wrong thread when the
 * runtime hasn't finished switching yet.
 */
export function RegisterBenchmarkHandle({
  setHandle,
  threadId,
}: {
  setHandle: (handle: CompareHandle | null) => void;
  threadId?: string;
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

    // If no specific thread is expected, register immediately (e.g. initial mount).
    if (!threadId) {
      setHandle(handle);
      return () => setHandle(null);
    }

    // Otherwise wait for ThreadAutoSwitch to activate the target thread before
    // handing off the handle. This ensures aui.thread().append() writes to the
    // correct pre-created benchmark thread, not a freshly-initialized one.
    let registered = false;

    const tryRegister = () => {
      if (useChatRuntimeStore.getState().activeThreadId === threadId && !registered) {
        registered = true;
        setHandle(handle);
        return true;
      }
      return false;
    };

    if (tryRegister()) {
      return () => setHandle(null);
    }

    const unsub = useChatRuntimeStore.subscribe((state) => {
      if (state.activeThreadId === threadId && !registered) {
        registered = true;
        unsub();
        setHandle(handle);
      }
    });

    // Safety: if the thread never activates within 10s, give up
    const timeout = setTimeout(() => {
      if (!registered) {
        unsub();
      }
    }, 10_000);

    return () => {
      clearTimeout(timeout);
      unsub();
      setHandle(null);
    };
  }, [aui, setHandle, threadId]);

  return null;
}
