// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef, useState } from "react";
import { db } from "../db";
import { loadModel } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useBenchmarkStore } from "../stores/use-benchmark-store";
import type { CompareHandle } from "../shared-composer";
import type { BenchmarkProgress } from "./types";

/** Poll until the orchestrator registers its handle or timeout expires. */
async function waitForHandle(
  getHandle: () => CompareHandle | null,
  timeoutMs = 8000,
): Promise<CompareHandle | null> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const h = getHandle();
    if (h) return h;
    await new Promise<void>((r) => setTimeout(r, 50));
  }
  return null;
}

export function useBenchmarkRunner() {
  const [progress, setProgress] = useState<BenchmarkProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const cancelRef = useRef(false);
  const getterRef = useRef<(() => CompareHandle | null) | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Run one prompt against all selected models, creating or reusing threads.
   * Called by the shared benchmarkSendFn each time the user submits in
   * benchmark mode.
   */
  const run = useCallback(
    async (
      promptText: string,
      getHandle: () => CompareHandle | null,
      navigateToThread: (threadId: string) => void,
    ) => {
      const store = useBenchmarkStore.getState();
      const selectedIds = store.benchmarkSelectedModelIds;
      if (selectedIds.length === 0) return;

      // Clear any pending dismiss timer
      if (dismissTimerRef.current !== null) {
        clearTimeout(dismissTimerRef.current);
        dismissTimerRef.current = null;
      }
      cancelRef.current = false;
      getterRef.current = getHandle;
      setRunning(true);

      const maxSeqLength = useChatRuntimeStore.getState().params.maxSeqLength;

      // Determine or create the benchmark ID and per-model thread IDs
      let benchmarkId = store.activeBenchmarkId;
      let threadIds = { ...store.activeBenchmarkThreadIds };

      if (!benchmarkId) {
        // First message in this benchmark session — create a new run
        benchmarkId = crypto.randomUUID();
        const runtimeStore = useChatRuntimeStore.getState();
        const allModels = runtimeStore.models;
        const allLoras = runtimeStore.loras;

        for (let mi = 0; mi < selectedIds.length; mi++) {
          const modelId = selectedIds[mi];
          // Support "repoId::quantName" format for GGUF quant selections
          const sepIdx = modelId.indexOf("::");
          const baseModelId = sepIdx >= 0 ? modelId.slice(0, sepIdx) : modelId;
          const quantVariant = sepIdx >= 0 ? modelId.slice(sepIdx + 2) : null;

          const modelEntry =
            allModels.find((m) => m.id === baseModelId) ??
            allLoras.find((l) => l.id === baseModelId);
          const baseName =
            (modelEntry as { name?: string })?.name ??
            baseModelId.split("/").pop() ??
            baseModelId;
          const displayName = quantVariant ? `${baseName} ${quantVariant}` : baseName;

          const threadId = crypto.randomUUID();
          await db.threads.add({
            id: threadId,
            title: displayName,
            modelType: "base",
            modelId,
            benchmarkId,
            benchmarkName: store.benchmarkName,
            archived: false,
            createdAt: Date.now() + mi,
          });
          threadIds[modelId] = threadId;
        }

        useBenchmarkStore.getState().setActiveBenchmark(benchmarkId, threadIds);
      }

      // Run the prompt through each selected model
      for (let mi = 0; mi < selectedIds.length; mi++) {
        if (cancelRef.current) break;
        const modelId = selectedIds[mi];
        const threadId = threadIds[modelId];
        if (!threadId) continue;

        // Support "repoId::quantName" format for GGUF quant selections
        const sepIdx = modelId.indexOf("::");
        const baseModelId = sepIdx >= 0 ? modelId.slice(0, sepIdx) : modelId;
        const quantVariant = sepIdx >= 0 ? modelId.slice(sepIdx + 2) : null;

        const runtimeStore = useChatRuntimeStore.getState();
        const modelEntry =
          runtimeStore.models.find((m) => m.id === baseModelId) ??
          runtimeStore.loras.find((l) => l.id === baseModelId);
        const baseName =
          (modelEntry as { name?: string })?.name ??
          baseModelId.split("/").pop() ??
          baseModelId;
        const modelName = quantVariant ? `${baseName} ${quantVariant}` : baseName;
        const isLora = (modelEntry as { isLora?: boolean })?.isLora ?? false;
        const ggufVariant = quantVariant ?? (modelEntry as { ggufVariant?: string })?.ggufVariant ?? null;

        // 1. Load the model
        setProgress({
          promptIdx: 0,
          modelIdx: mi,
          totalPrompts: 1,
          totalModels: selectedIds.length,
          currentModelName: modelName,
          phase: "loading",
        });

        const runtimeState = useChatRuntimeStore.getState();
        const currentCheckpoint = runtimeState.params.checkpoint;
        const currentGgufVariant = runtimeState.activeGgufVariant;
        if (currentCheckpoint !== baseModelId || currentGgufVariant !== ggufVariant) {
          try {
            await loadModel({
              model_path: baseModelId,
              hf_token: useChatRuntimeStore.getState().hfToken ?? null,
              max_seq_length: maxSeqLength,
              load_in_4bit: false,
              is_lora: isLora,
              gguf_variant: ggufVariant,
            });
          } catch {
            continue; // Skip this model on load failure
          }
          if (cancelRef.current) break;
          useChatRuntimeStore
            .getState()
            .setCheckpoint(baseModelId, ggufVariant ?? null);
        }

        // 2. Navigate to the thread and wait for the orchestrator handle
        setActiveThreadId(threadId);
        navigateToThread(threadId);

        const handle = await waitForHandle(getHandle);
        if (!handle || cancelRef.current) break;

        // 3. Send the prompt
        setProgress({
          promptIdx: 0,
          modelIdx: mi,
          totalPrompts: 1,
          totalModels: selectedIds.length,
          currentModelName: modelName,
          phase: "generating",
        });

        // Use append (not appendMessage+startRun) so the user message and run
        // start atomically — startRun reads stale getState().messages otherwise.
        handle.append([{ type: "text", text: promptText }]);
        await handle.waitForRunEnd();
      }

      setProgress((p) =>
        p
          ? { ...p, phase: "done" }
          : {
              promptIdx: 0,
              modelIdx: 0,
              totalPrompts: 0,
              totalModels: 0,
              currentModelName: "",
              phase: "done",
            },
      );

      dismissTimerRef.current = setTimeout(() => {
        setProgress(null);
        setRunning(false);
        setActiveThreadId(null);
        dismissTimerRef.current = null;
      }, 3000);
    },
    [],
  );

  const cancel = useCallback(() => {
    cancelRef.current = true;
    getterRef.current?.()?.cancel();
    if (dismissTimerRef.current !== null) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
    setRunning(false);
    setProgress(null);
    setActiveThreadId(null);
  }, []);

  return { run, cancel, progress, running, activeThreadId };
}
