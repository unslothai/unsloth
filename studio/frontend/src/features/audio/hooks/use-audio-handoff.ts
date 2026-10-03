// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { translate } from "@/i18n";
import { loadGalleryUntil } from "@/lib/gallery-deep-link";
import { toast } from "@/lib/toast";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import {
  audioWorkflowForTask,
  clipWorkflow,
  isAudioWorkflowId,
} from "../workflows";
import type { AudioHostState } from "./audio-host-state";
import { type AudioGallery, galleryCache } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";

export function useAudioHandoff({
  active,
  busy,
  busyRef,
  mode,
  modeRef,
  handleModelSelect,
  transitionMode,
  transitionWorkflow,
  refreshGallery,
  loadMore,
  loadingMoreRef,
  selectClip,
}: Pick<AudioHostState, "active" | "busy" | "busyRef" | "mode" | "modeRef"> &
  Pick<
    AudioModelSlot,
    "handleModelSelect" | "transitionMode" | "transitionWorkflow"
  > &
  Pick<
    AudioGallery,
    "refreshGallery" | "loadMore" | "loadingMoreRef" | "selectClip"
  >) {
  const navigateSelf = useNavigate();
  const routeSearch = useSearch({ strict: false }) as {
    model?: string;
    quant?: string;
    ggufQuant?: string;
    task?: string;
    workflow?: string;
    audioType?: string;
    loadId?: string;
    item?: string;
  };
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      const routedWorkflow = routeSearch.workflow;
      if (isAudioWorkflowId(routedWorkflow)) {
        // Left in the URL when refused, so it retries once busy releases.
        if (!transitionWorkflow(routedWorkflow)) return;
        void navigateSelf({ to: "/audio", search: {}, replace: true });
        return;
      }
      const task = routeSearch.task;
      if (!task) return;
      const intended =
        task === "automatic-speech-recognition" ? "transcribe" : "speak";
      if (intended !== mode && !transitionMode(intended)) return;
      void navigateSelf({ to: "/audio", search: {}, replace: true });
      // text-to-audio is Music; the other tags name their page directly.
      useAudioWorkspaceStore
        .getState()
        .commitWorkflow(audioWorkflowForTask(task) ?? intended);
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}|${routeSearch.ggufQuant ?? ""}|${routeSearch.task ?? ""}|${routeSearch.audioType ?? ""}|${routeSearch.loadId ?? ""}|${routeSearch.workflow ?? ""}`;
    if (handledRouteModel.current === key) return;
    if (busyRef.current !== null) return;
    // Open the named page first: a staged or failed load otherwise left the user on another page.
    const routedWorkflow = routeSearch.workflow;
    if (
      isAudioWorkflowId(routedWorkflow) &&
      !transitionWorkflow(routedWorkflow)
    ) {
      return;
    }
    handledRouteModel.current = key;
    handleModelSelect(wanted, {
      source: "hub",
      isLora: false,
      ggufFilename: routeSearch.quant ?? undefined,
      ggufVariant: routeSearch.ggufQuant ?? undefined,
      loadId: routeSearch.loadId ?? undefined,
      audioType: routeSearch.audioType ?? undefined,
      // Chat-to-Audio routing drops the inventory flag, so stage the exact forwarded GGUF.
      isDownloaded: routeSearch.loadId
        ? true
        : routeSearch.quant
          ? false
          : undefined,
      pipelineTag: routeSearch.task ?? null,
    });
    void navigateSelf({ to: "/audio", search: {}, replace: true });
  }, [
    active,
    busy,
    mode,
    routeSearch.model,
    routeSearch.quant,
    routeSearch.ggufQuant,
    routeSearch.task,
    routeSearch.workflow,
    routeSearch.audioType,
    routeSearch.loadId,
    handleModelSelect,
    navigateSelf,
    transitionMode,
    transitionWorkflow,
  ]);

  // A counter, not effect cleanup, retires a lookup: clearing the query must not cancel its own.
  const routedItem = active ? routeSearch.item : undefined;
  const routedLookup = useRef(0);
  useEffect(() => {
    if (!active) routedLookup.current += 1;
  }, [active]);
  useEffect(() => {
    if (!routedItem) return;
    const lookup = ++routedLookup.current;
    void navigateSelf({
      to: "/audio",
      search: (prev) => ({ ...prev, item: undefined }),
      replace: true,
    });
    void loadGalleryUntil({
      has: () => galleryCache.clips.some((clip) => clip.id === routedItem),
      count: () => galleryCache.clips.length,
      hasMore: () => galleryCache.hasMore,
      refresh: () => refreshGallery(undefined, galleryCache.clips.length),
      loadMore,
      busy: () => loadingMoreRef.current,
      cancelled: () => lookup !== routedLookup.current,
    }).then((found) => {
      if (lookup !== routedLookup.current) return;
      if (found) {
        selectClip(routedItem);
        const clip = galleryCache.clips.find((c) => c.id === routedItem);
        if (clip && modeRef.current === "speak")
          transitionWorkflow(clipWorkflow(clip));
      } else {
        toast(translate("library.toast.clipNotFound"), {
          description: translate("library.toast.notFoundDescription"),
        });
      }
    });
  }, [
    routedItem,
    navigateSelf,
    refreshGallery,
    loadMore,
    selectClip,
    modeRef,
    transitionWorkflow,
  ]);

  return {
    navigateSelf,
  };
}

export type AudioHandoff = ReturnType<typeof useAudioHandoff>;
