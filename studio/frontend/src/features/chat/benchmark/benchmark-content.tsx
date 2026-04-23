// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef } from "react";
import {
  CompareHandlesProvider,
  RegisterCompareHandle,
  type CompareHandle,
  type CompareHandles,
} from "../shared-composer";
import { ChatRuntimeProvider } from "../runtime-provider";

/**
 * Invisible orchestration mount. Exists only while a benchmark run is active.
 * Registers a CompareHandle under the name "bench" so useBenchmarkRunner can
 * drive appendMessage / startRun / waitForRunEnd against the current thread.
 *
 * The user watches generation live in the normal SingleContent view.
 * This component renders nothing visible.
 */
export function BenchmarkOrchestrator({
  currentThreadId,
  onHandleReady,
}: {
  currentThreadId: string | undefined;
  onHandleReady: (getHandle: () => CompareHandle | null) => void;
}) {
  const handlesRef = useRef<Record<string, CompareHandle>>({}) as CompareHandles;

  // Stable callback — parent must wrap in useCallback to avoid re-firing
  useEffect(() => {
    onHandleReady(() => handlesRef.current["bench"] ?? null);
  }, [onHandleReady]);

  if (!currentThreadId) return null;

  return (
    // style= is more reliable than className="hidden" — avoids Tailwind purge edge cases
    <div style={{ display: "none" }} aria-hidden="true">
      <CompareHandlesProvider handlesRef={handlesRef}>
        <ChatRuntimeProvider
          modelType="base"
          initialThreadId={currentThreadId}
          syncActiveThreadId={false}
        >
          <RegisterCompareHandle name="bench" />
        </ChatRuntimeProvider>
      </CompareHandlesProvider>
    </div>
  );
}
