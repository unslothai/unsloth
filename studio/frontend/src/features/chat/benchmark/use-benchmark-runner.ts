// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef, useState } from "react";
import { db } from "../db";
import { loadModel } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { CompareHandle } from "../shared-composer";
import type { BenchmarkConfig, BenchmarkProgress } from "./types";

/** Poll until the orchestrator registers its handle or timeout expires. */
async function waitForHandle(
  getHandle: () => CompareHandle | null,
  timeoutMs = 5000,
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

  const run = useCallback(
    async (
      config: BenchmarkConfig,
      getHandle: () => CompareHandle | null,
      navigateToThread: (threadId: string) => void,
    ) => {
      // Clear any pending dismiss timer from a previous run
      if (dismissTimerRef.current !== null) {
        clearTimeout(dismissTimerRef.current);
        dismissTimerRef.current = null;
      }
      cancelRef.current = false;
      getterRef.current = getHandle;
      setRunning(true);

      const { prompts, models } = config;

      for (let mi = 0; mi < models.length; mi++) {
        if (cancelRef.current) break;
        const model = models[mi];
        const modelName = model.displayName ?? model.id;

        // 1. Load the model if not already loaded
        setProgress({
          promptIdx: 0,
          modelIdx: mi,
          totalPrompts: prompts.length,
          totalModels: models.length,
          currentModelName: modelName,
          phase: "loading",
        });

        const currentCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
        if (currentCheckpoint !== model.id) {
          try {
            await loadModel({
              model_path: model.id,
              hf_token: useChatRuntimeStore.getState().hfToken ?? null,
              max_seq_length: config.maxSeqLength,
              load_in_4bit: false,
              is_lora: model.isLora,
              gguf_variant: model.ggufVariant ?? null,
            });
          } catch {
            // If load fails, skip this model
            continue;
          }
          if (cancelRef.current) break;
          useChatRuntimeStore.getState().setCheckpoint(model.id, model.ggufVariant ?? null);
        }

        // 2. Create a DB thread for this model's run
        const threadId = crypto.randomUUID();
        await db.threads.add({
          id: threadId,
          title: modelName,
          modelType: "base",
          modelId: model.id,
          benchmarkId: config.id,
          benchmarkName: config.name,
          archived: false,
          // +mi offsets ensure stable insertion order even if timestamps collide
          createdAt: Date.now() + mi,
        });

        // 3. Navigate the visible view to this thread and wait for the orchestrator
        setActiveThreadId(threadId);
        navigateToThread(threadId);

        const handle = await waitForHandle(getHandle);
        if (!handle || cancelRef.current) break;

        // 4. Loop through all prompts on this thread
        for (let pi = 0; pi < prompts.length; pi++) {
          if (cancelRef.current) break;
          setProgress({
            promptIdx: pi,
            modelIdx: mi,
            totalPrompts: prompts.length,
            totalModels: models.length,
            currentModelName: modelName,
            phase: "generating",
          });
          handle.appendMessage([{ type: "text", text: prompts[pi].text }]);
          handle.startRun();
          await handle.waitForRunEnd();
        }
      }

      setProgress((p) =>
        p ? { ...p, phase: "done" } : { promptIdx: 0, modelIdx: 0, totalPrompts: 0, totalModels: 0, currentModelName: "", phase: "done" },
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
    // Abort any in-flight generation
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
