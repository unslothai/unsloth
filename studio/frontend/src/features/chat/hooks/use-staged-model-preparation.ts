// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect } from "react";

import { useRepoDownload } from "@/features/hub/download-manager/use-repo-download";
import type { DownloadJob } from "@/features/hub/download-manager/use-repo-download";
import { useLatestRef } from "@/features/hub/hooks/use-latest-ref";

import { fetchGgufStagedMetadata } from "../api/chat-api";
import {
  isPendingGguf,
  pendingSelectionMatches,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import type { PendingModelSelection } from "../stores/chat-runtime-store";

/**
 * Drives the deferred ("Load on selection" off) staging flow for a GGUF:
 * download the file if needed (HF repo) or read it in place (native drag-drop /
 * picked file), then read its header context length so the settings sheet can
 * show the real context slider before the single GPU load. The staged context
 * lands on `pendingSelection.contextLength` (scoped to the staged model, never
 * the loaded model's `ggufContextLength`). Returns the live download job so the
 * sheet can render progress / cancel. Mount once on the chat page.
 */
export function useStagedModelPreparation(opts?: {
  /** Load the cached file once an autoLoad pick's download completes. */
  onAutoLoad?: (pending: PendingModelSelection) => void;
}): DownloadJob {
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
  // Non-GGUF HF repos download a full snapshot (variant null) but have no header.
  const pendingIsHubRepo = useChatRuntimeStore(
    (s) => s.pendingSelection?.isHubRepo ?? false,
  );
  const pendingDownloaded = useChatRuntimeStore(
    (s) => s.pendingSelection?.isDownloaded ?? false,
  );
  // "Already probed" must key off layerCount / moeLayerCount, which only the
  // full header probe fills (it sets all three together, so either is a
  // reliable marker). contextLength alone can be list-seeded from
  // /gguf-variants, which returns no layer/MoE counts -- treating it as
  // complete would skip the probe and leave the GPU Layers slider at its 256
  // fallback and the MoE slider hidden until the model loads.
  const pendingHasMetadata = useChatRuntimeStore(
    (s) =>
      s.pendingSelection?.layerCount != null ||
      s.pendingSelection?.moeLayerCount != null,
  );
  const setPendingSelection = useChatRuntimeStore((s) => s.setPendingSelection);
  const onAutoLoadRef = useLatestRef(opts?.onAutoLoad);

  // A failed or cancelled autoLoad download has no sheet to retry from, so drop
  // the staged pick rather than leave it waiting on a load that won't come.
  const handleAutoLoadAbort = useCallback((variant: string | null) => {
    const latest = useChatRuntimeStore.getState().pendingSelection;
    if (
      latest?.autoLoad &&
      (latest.ggufVariant ?? null) === (variant ?? null)
    ) {
      useChatRuntimeStore.getState().abandonStagedModel();
    }
  }, []);

  const fetchContextMetadata = useCallback(async () => {
    const current = useChatRuntimeStore.getState().pendingSelection;
    if (!current?.id || !isPendingGguf(current)) return;
    const { id, ggufVariant, nativePathToken } = current;
    try {
      const { contextLength, layerCount, moeLayerCount } =
        await fetchGgufStagedMetadata({
          model_path: id,
          gguf_variant: ggufVariant,
          hf_token: useChatRuntimeStore.getState().hfToken || null,
          nativePathToken,
        });
      // Apply only if the same model is still staged (the user may have switched
      // picks or loaded/cancelled while the request was in flight).
      const latest = useChatRuntimeStore.getState().pendingSelection;
      if (
        latest &&
        pendingSelectionMatches(latest, { id, ggufVariant, nativePathToken }) &&
        (contextLength != null || layerCount != null || moeLayerCount != null)
      ) {
        setPendingSelection({
          ...latest,
          contextLength,
          layerCount,
          moeLayerCount,
        });
      }
    } catch {
      // Leave metadata null: the context/MoE sliders stay hidden and the user
      // can still load (they fill in from the load response afterwards).
    }
  }, [setPendingSelection]);

  const job = useRepoDownload({
    kind: "model",
    // useRepoDownload must be called unconditionally; an idle repo id keeps it
    // inert until something is staged.
    repoId: pendingId ?? "__staged_idle__",
    activeVariant: pendingVariant,
    onComplete: (variant) => {
      // autoLoad picks load the cached file now; staged picks read the header so
      // the sheet's context slider can show before a manual load.
      const latest = useChatRuntimeStore.getState().pendingSelection;
      if (
        latest?.autoLoad &&
        (latest.ggufVariant ?? null) === (variant ?? null)
      ) {
        onAutoLoadRef.current?.(latest);
        return;
      }
      void fetchContextMetadata();
    },
    onError: handleAutoLoadAbort,
    onCancelled: handleAutoLoadAbort,
  });

  // job.requestStartDownload's identity changes per render; hold it in a ref so
  // the staging effect re-runs only when the staged model itself changes.
  const startDownloadRef = useLatestRef(job.requestStartDownload);
  const fetchMetadataRef = useLatestRef(fetchContextMetadata);

  useEffect(() => {
    // GGUF picks (header worth reading) and uncached non-GGUF hub repos (full
    // snapshot, no header) both run here; everything else is loaded directly.
    if (
      !pendingId ||
      (!pendingIsGguf && !pendingIsHubRepo) ||
      pendingHasMetadata
    ) {
      return;
    }
    // Native files and already-downloaded HF files are local: read the header
    // now. Otherwise download first (a GGUF variant, or a null-variant snapshot
    // for a hub repo); onComplete then reads the header or auto-loads.
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
    pendingIsHubRepo,
    pendingDownloaded,
    pendingHasMetadata,
    startDownloadRef,
    fetchMetadataRef,
  ]);

  return job;
}
