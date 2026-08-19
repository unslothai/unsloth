// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

export type GeneratedImageOverlayState = {
  image: string;
  title: string;
  metadata: string;
  filename?: string;
  openaiImageGenerationCallId?: string;
  openaiResponseId?: string;
  openaiReasoningItem?: unknown;
  threadId?: string | null;
};

type GeneratedImageOverlayContextValue = {
  overlay: GeneratedImageOverlayState | null;
  openOverlay: (overlay: GeneratedImageOverlayState) => void;
  closeOverlay: () => void;
};

const GeneratedImageOverlayContext =
  createContext<GeneratedImageOverlayContextValue | null>(null);

export function GeneratedImageOverlayProvider({
  children,
  threadId = null,
}: {
  children: ReactNode;
  threadId?: string | null;
}) {
  const [overlay, setOverlay] = useState<GeneratedImageOverlayState | null>(
    null,
  );

  const openOverlay = useCallback(
    (nextOverlay: GeneratedImageOverlayState) => {
      setOverlay({ ...nextOverlay, threadId: nextOverlay.threadId ?? threadId });
    },
    [threadId],
  );

  const closeOverlay = useCallback(() => {
    setOverlay(null);
  }, []);

  const value = useMemo(
    () => ({ overlay, openOverlay, closeOverlay }),
    [closeOverlay, openOverlay, overlay],
  );

  return (
    <GeneratedImageOverlayContext.Provider value={value}>
      {children}
    </GeneratedImageOverlayContext.Provider>
  );
}

export function useGeneratedImageOverlay(): GeneratedImageOverlayContextValue {
  const context = useContext(GeneratedImageOverlayContext);
  if (!context) {
    throw new Error(
      "useGeneratedImageOverlay must be used within GeneratedImageOverlayProvider.",
    );
  }
  return context;
}
