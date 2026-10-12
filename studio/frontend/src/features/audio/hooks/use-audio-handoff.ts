// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { translate } from "@/i18n";
import { loadGalleryUntil } from "@/lib/gallery-deep-link";
import { toast } from "@/lib/toast";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { audioRouteIntent, audioWorkflowForPick } from "../route-search";
import { clipWorkflow, isAudioWorkflowId } from "../workflows";
import type { AudioHostState } from "./audio-host-state";
import { type AudioGallery, galleryCache } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";

export function useAudioHandoff({
  active,
  busy,
  busyRef,
  handleModelSelect,
  transitionWorkflow,
  refreshGallery,
  loadMore,
  loadingMoreRef,
  selectClip,
}: Pick<AudioHostState, "active" | "busy" | "busyRef"> &
  Pick<AudioModelSlot, "handleModelSelect" | "transitionWorkflow"> &
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
    gguf?: boolean;
    item?: string;
  };
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      // An explicit workflow, else the one the task names (text-to-audio is Music).
      const routedWorkflow = audioRouteIntent(routeSearch);
      if (routedWorkflow === null) return;
      // Left in the URL when refused, so it retries once busy releases.
      if (!transitionWorkflow(routedWorkflow)) return;
      void navigateSelf({ to: "/audio", search: {}, replace: true });
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}|${routeSearch.ggufQuant ?? ""}|${routeSearch.task ?? ""}|${routeSearch.audioType ?? ""}|${routeSearch.loadId ?? ""}|${routeSearch.gguf ? "gguf" : ""}|${routeSearch.workflow ?? ""}`;
    if (handledRouteModel.current === key) return;
    if (busyRef.current !== null) return;
    // Open the named page first: a staged or failed load otherwise left the user on another page.
    // Without an explicit workflow, the task (and audio type) name the page the load is for.
    const routedWorkflow = isAudioWorkflowId(routeSearch.workflow)
      ? routeSearch.workflow
      : audioWorkflowForPick({
          id: wanted,
          task: routeSearch.task,
          audioType: routeSearch.audioType,
        });
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
      isGguf: routeSearch.gguf ?? undefined,
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
    routeSearch.model,
    routeSearch.quant,
    routeSearch.ggufQuant,
    routeSearch.task,
    routeSearch.workflow,
    routeSearch.audioType,
    routeSearch.loadId,
    routeSearch.gguf,
    handleModelSelect,
    navigateSelf,
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
        // Through ?workflow=, so a switch refused while busy stays in the URL and retries.
        if (clip) {
          void navigateSelf({
            to: "/audio",
            search: (prev) => ({ ...prev, workflow: clipWorkflow(clip) }),
            replace: true,
          });
        }
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
  ]);

  return {
    navigateSelf,
  };
}
