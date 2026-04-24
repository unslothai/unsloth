// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef, useState } from "react";
import { db } from "../db";
import { loadModel } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { usePromptEvalStore } from "../stores/use-prompt-eval-store";
import type { CompareHandle } from "../shared-composer";
import type { PromptEvalProgress } from "./types";

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

export function usePromptEvalRunner() {
  const [progress, setProgress] = useState<PromptEvalProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const cancelRef = useRef(false);
  const getterRef = useRef<(() => CompareHandle | null) | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Run one or more prompts against all selected models sequentially.
   * For each model: load it (if needed), then send each prompt in order
   * waiting for the response before sending the next.
   * Called by the shared promptEvalSendFn each time the user submits in
   * Prompt Eval mode, or by the prompt list runner with multiple texts.
   */
  const run = useCallback(
    async (
      promptTexts: string[],
      getHandle: () => CompareHandle | null,
      navigateToThread: (threadId: string) => void,
    ) => {
      const store = usePromptEvalStore.getState();
      const selectedIds = store.promptEvalSelectedModelIds;
      if (selectedIds.length === 0) return;
      if (promptTexts.length === 0) return;

      // Clear any pending dismiss timer
      if (dismissTimerRef.current !== null) {
        clearTimeout(dismissTimerRef.current);
        dismissTimerRef.current = null;
      }
      cancelRef.current = false;
      getterRef.current = getHandle;
      setRunning(true);

      const maxSeqLength = useChatRuntimeStore.getState().params.maxSeqLength;

      // Capture the user's current tool/reasoning settings so we can restore
      // them after each model load. The periodic refresh() poll in
      // use-chat-model-runtime.ts calls setReasoningEnabled(true) whenever a
      // reasoning-capable model is detected, and selectModel() auto-sets
      // toolsEnabled/codeToolsEnabled to true when a tool-capable model loads.
      // Restoring after each load guarantees all models run under the same
      // (user-intended) settings, preventing accidental web_search calls.
      const promptEvalSettings = {
        reasoningEnabled: useChatRuntimeStore.getState().reasoningEnabled,
        toolsEnabled: useChatRuntimeStore.getState().toolsEnabled,
        codeToolsEnabled: useChatRuntimeStore.getState().codeToolsEnabled,
      };

      // Determine or create the benchmark ID and per-model thread IDs
      let promptEvalId = store.activePromptEvalId;
      let threadIds = { ...store.activePromptEvalThreadIds };

      if (!promptEvalId) {
        // First message in this prompt eval session — create a new run
        promptEvalId = crypto.randomUUID();
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
            promptEvalId,
            promptEvalName: store.promptEvalName,
            archived: false,
            createdAt: Date.now() + mi,
          });
          threadIds[modelId] = threadId;
        }

        usePromptEvalStore.getState().setActiveBenchmark(promptEvalId, threadIds);
      }

      // Run the prompt(s) through each selected model
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

        // 1. Load the model (skip if already loaded)
        setProgress({
          promptIdx: 0,
          modelIdx: mi,
          totalPrompts: promptTexts.length,
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

        // Restore the user's tool/reasoning settings after each model load.
        // Model loading (via selectModel or the status poll) may auto-enable
        // toolsEnabled/codeToolsEnabled and setReasoningEnabled, overriding
        // the user's intent. We restore here so every model runs under identical
        // settings.
        useChatRuntimeStore.setState(promptEvalSettings);

        // 2. Navigate to this model's thread (user sees it live)
        setActiveThreadId(threadId);
        navigateToThread(threadId);

        // 3. Wait for the visible thread's handle to be ready after navigation
        const handle = await waitForHandle(getHandle);
        if (!handle || cancelRef.current) break;

        // 4. Send each prompt in order, waiting for each response to complete
        for (let pi = 0; pi < promptTexts.length; pi++) {
          if (cancelRef.current) break;
          const promptText = promptTexts[pi];

          setProgress({
            promptIdx: pi,
            modelIdx: mi,
            totalPrompts: promptTexts.length,
            totalModels: selectedIds.length,
            currentModelName: modelName,
            phase: "generating",
          });

          handle.append([{ type: "text", text: promptText }]);
          await handle.waitForRunEnd();
        }
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
