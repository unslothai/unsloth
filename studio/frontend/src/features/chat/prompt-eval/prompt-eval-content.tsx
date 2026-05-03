// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import {
  CompareHandlesProvider,
  RegisterCompareHandle,
  type CompareHandle,
  type CompareHandles,
} from "../shared-composer";
import { ChatRuntimeProvider } from "../runtime-provider";

/**
 * Invisible orchestration component. Mounts a hidden ChatRuntimeProvider
 * connected to the prompt eval target thread and exposes a stable CompareHandle
 * getter to the runner. Completely independent of the visible SingleContent —
 * no race condition with navigation possible.
 *
 * When currentThreadId changes (next model), ThreadAutoSwitch inside
 * ChatRuntimeProvider switches the hidden runtime to the new thread.
 * RegisterCompareHandle re-registers, and waitForHandle() in the runner
 * picks up the fresh handle automatically via polling.
 */
export function PromptEvalOrchestrator({
  currentThreadId,
  onHandleReady,
}: {
  currentThreadId: string | undefined;
  onHandleReady: (getHandle: () => CompareHandle | null) => void;
}) {
  const handlesRef = useRef<Record<string, CompareHandle>>({}) as CompareHandles;

  // Give the runner a stable getter that always reads the latest handle.
  useEffect(() => {
    onHandleReady(() => handlesRef.current["eval"] ?? null);
  }, [onHandleReady]);

  if (!currentThreadId) return null;

  return (
    <div style={{ display: "none" }} aria-hidden="true">
      <CompareHandlesProvider handlesRef={handlesRef}>
        <ChatRuntimeProvider
          modelType="base"
          initialThreadId={currentThreadId}
          syncActiveThreadId={false}
        >
          <RegisterCompareHandle name="eval" />
        </ChatRuntimeProvider>
      </CompareHandlesProvider>
    </div>
  );
}

