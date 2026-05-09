// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";
import { toast } from "sonner";
import { extractDocument } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ExtractedDocument } from "../types";
import { MAX_DOC_SIZE } from "../utils/document-extraction";
import { runWithTemporaryOcrModel } from "../utils/ocr-model-orchestrator";

// ---------------------------------------------------------------------------
// Non-React helper — usable outside component tree (e.g. async generators
// inside runtime-provider's adapter). The hook wraps this for convenience.
// ---------------------------------------------------------------------------

export interface DocumentExtractionRunnerOptions {
  onProgress?: (pct: number) => void;
}

export interface DocumentExtractionRunner {
  run: (
    file: File,
    options?: DocumentExtractionRunnerOptions,
  ) => Promise<ExtractedDocument>;
  abort: () => void;
}

/**
 * Creates a stateful extraction runner that owns its own AbortController.
 * Reads settings from the Zustand store at call time (not at creation time)
 * so changes to tokenBudget / describeImages take effect on the next call.
 *
 * This factory is intentionally framework-free so it can be used inside
 * async generator functions in runtime-provider.tsx without violating the
 * Rules of Hooks.
 */
export function createDocumentExtractionRunner(): DocumentExtractionRunner {
  let controller: AbortController | null = null;

  const run = async (
    file: File,
    options?: DocumentExtractionRunnerOptions,
  ): Promise<ExtractedDocument> => {
    // Read settings at call time so latest values are always used.
    const { docExtract } = useChatRuntimeStore.getState();

    if (!docExtract.enabled) {
      throw new Error("Document extraction is disabled in settings.");
    }

    if (file.size > MAX_DOC_SIZE) {
      throw new Error(
        `File "${file.name}" exceeds the 100 MB limit (${(file.size / 1024 / 1024).toFixed(1)} MB).`,
      );
    }

    // Abort any previous in-flight extraction before starting a new one.
    if (controller) {
      controller.abort();
    }
    controller = new AbortController();
    const signal = controller.signal;

    // Wrap extraction in the OCR-model orchestrator. When the user has
    // selected an OCR preset (or a custom OCR model), this temporarily
    // swaps the active chat model with the OCR model for the duration of
    // the extraction call, then restores the original chat model in
    // `finally`. With ocrModel === "default" or "none" the orchestrator is
    // a no-op pass-through and behaviour matches the loaded-model path.
    const result = await runWithTemporaryOcrModel({
      settings: docExtract,
      signal,
      run: () =>
        extractDocument(
          file,
          {
            describeImages: docExtract.describeImages,
            useVlmOcr: docExtract.useVlmOcr,
            maxFigures: docExtract.maxFigures,
            maxVisualPayloads: docExtract.maxVisualPayloads,
            tokenBudget: docExtract.tokenBudget,
          },
          signal,
          options?.onProgress,
        ),
    });

    if (result.describe_skipped_reason) {
      toast.warning("Figure descriptions were skipped", {
        description: result.describe_skipped_reason,
      });
    }

    return result;
  };

  const abort = () => {
    if (controller) {
      controller.abort();
      controller = null;
    }
  };

  return { run, abort };
}

// ---------------------------------------------------------------------------
// React hook — thin wrapper around createDocumentExtractionRunner that
// keeps the runner instance stable across renders via useRef.
// ---------------------------------------------------------------------------

export interface UseDocumentExtractionResult {
  extract: (
    file: File,
    options?: DocumentExtractionRunnerOptions,
  ) => Promise<ExtractedDocument>;
  abort: () => void;
}

/**
 * React hook for document extraction. Owns a single AbortController
 * per hook instance; calling `abort()` cancels any in-flight request.
 *
 * Settings (`tokenBudget`, `describeImages`, etc.) are read from the
 * Zustand store at extraction time — not at hook instantiation — so
 * settings changes are always reflected on the next extraction.
 *
 * For use outside React component trees (e.g. async generators), use
 * {@link createDocumentExtractionRunner} directly.
 */
export function useDocumentExtraction(): UseDocumentExtractionResult {
  const runnerRef = useRef<DocumentExtractionRunner | null>(null);
  if (runnerRef.current == null) {
    runnerRef.current = createDocumentExtractionRunner();
  }

  const extract = useCallback(
    (file: File, options?: DocumentExtractionRunnerOptions) => {
      return runnerRef.current!.run(file, options);
    },
    [],
  );

  const abort = useCallback(() => {
    runnerRef.current?.abort();
  }, []);

  return { extract, abort };
}
