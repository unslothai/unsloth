// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect } from "react";

import { useLatestRef } from "@/features/hub/hooks/use-latest-ref";
import { useRepoDownload } from "@/features/hub/download-manager/use-repo-download";
import type { DownloadJob } from "@/features/hub/download-manager/use-repo-download";

import { fetchGgufContextLength } from "../api/chat-api";
import {
  isPendingGguf,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";

/**
 * Drives the deferred ("Load on selection" off) staging flow for a GGUF:
 * download the file if needed (HF repo) or read it in place (native drag-drop /
 * picked file), then read its header context length so the settings sheet can
 * show the real context slider before the single GPU load. The staged context
 * lands on `pendingSelection.contextLength` (scoped to the staged model, never
 * the loaded model's `ggufContextLength`). Returns the live download job so the
 * sheet can render progress / cancel. Mount once on the chat page.
 */
export function useStagedModelPreparation(): DownloadJob {
  const pendingId = useChatRuntimeStore((s) => s.pendingSelection?.id ?? null);
  const pendingVariant = useChatRuntimeStore(
    (s) => s.pendingSelection?.ggufVariant ?? null,
  );
  const pendingNativeToken = useChatRuntimeStore(
    (s) => s.pendingSelection?.nativePathToken ?? null,
  );
  // Only GGUF picks (HF variant or native file) have a header worth reading.
  const pendingIsGguf = useChatRuntimeStore((s) =>
    isPendingGguf(s.pendingSelection),
  );
  const pendingDownloaded = useChatRuntimeStore(
    (s) => s.pendingSelection?.isDownloaded ?? false,
  );
  const pendingHasContext = useChatRuntimeStore(
    (s) => s.pendingSelection?.contextLength != null,
  );
  const setPendingSelection = useChatRuntimeStore((s) => s.setPendingSelection);

  const fetchContextMetadata = useCallback(async () => {
    const current = useChatRuntimeStore.getState().pendingSelection;
    if (!current?.id || !isPendingGguf(current)) return;
    const { id, ggufVariant, nativePathToken } = current;
    try {
      const contextLength = await fetchGgufContextLength({
        model_path: id,
        gguf_variant: ggufVariant,
        hf_token: useChatRuntimeStore.getState().hfToken || null,
        nativePathToken,
      });
      // Apply only if the same model is still staged (the user may have switched
      // picks or loaded/cancelled while the request was in flight). Native ids
      // are display labels, not paths, so two files can share an id -- compare
      // the path token too, or a stale response could land on the wrong pick.
      const latest = useChatRuntimeStore.getState().pendingSelection;
      if (
        latest?.id === id &&
        (latest.ggufVariant ?? null) === (ggufVariant ?? null) &&
        (latest.nativePathToken ?? null) === (nativePathToken ?? null) &&
        contextLength != null
      ) {
        setPendingSelection({ ...latest, contextLength });
      }
    } catch {
      // Leave contextLength null: the context slider stays hidden and the user
      // can still load (context fills in from the load response afterwards).
    }
  }, [setPendingSelection]);

  const job = useRepoDownload({
    kind: "model",
    // useRepoDownload must be called unconditionally; an idle repo id keeps it
    // inert until something is staged.
    repoId: pendingId ?? "__staged_idle__",
    activeVariant: pendingVariant,
    onComplete: () => {
      void fetchContextMetadata();
    },
  });

  // job.requestStartDownload's identity changes per render; hold it in a ref so
  // the staging effect re-runs only when the staged model itself changes.
  const startDownloadRef = useLatestRef(job.requestStartDownload);
  const fetchMetadataRef = useLatestRef(fetchContextMetadata);

  useEffect(() => {
    if (!pendingId || !pendingIsGguf || pendingHasContext) return;
    // Native files and already-downloaded HF files are local: read the header
    // now. Otherwise download first; onComplete then reads it.
    if (pendingNativeToken || pendingDownloaded) {
      void fetchMetadataRef.current();
    } else {
      const expectedBytes =
        useChatRuntimeStore.getState().pendingSelection?.expectedBytes ?? 0;
      void startDownloadRef.current(pendingVariant, expectedBytes);
    }
  }, [
    pendingId,
    pendingVariant,
    pendingNativeToken,
    pendingIsGguf,
    pendingDownloaded,
    pendingHasContext,
    startDownloadRef,
    fetchMetadataRef,
  ]);

  return job;
}
