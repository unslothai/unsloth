// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { translate } from "@/i18n";
import { loadGalleryUntil } from "@/lib/gallery-deep-link";
import { toast } from "@/lib/toast";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  audioRouteIntent,
  audioWorkflowForPick,
  audioWorkflowUrlAction,
} from "../route-search";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import {
  type AudioWorkflowId,
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
  const committedWorkflow = useAudioWorkspaceStore((state) => state.workflow);
  const handledRouteModel = useRef<string | null>(null);
  // The workflow the URL and the page last agreed on; cleared while hidden, so back/forward
  // into Audio reads the URL as a request.
  const lastSyncedWorkflow = useRef<AudioWorkflowId | null>(null);
  // A write the router has not rendered yet: until it lands, the old URL is not a request.
  const pendingWrite = useRef<{ from: AudioWorkflowId | null } | null>(null);
  useEffect(() => {
    if (!active) {
      lastSyncedWorkflow.current = null;
      pendingWrite.current = null;
      return;
    }
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      // An explicit workflow, else the one the task names (text-to-audio is Music).
      const urlWorkflow = audioRouteIntent(routeSearch);
      if (pendingWrite.current !== null) {
        if (urlWorkflow === pendingWrite.current.from) return;
        pendingWrite.current = null;
      }
      const action = audioWorkflowUrlAction({
        urlWorkflow,
        lastSynced: lastSyncedWorkflow.current,
        committed: committedWorkflow,
        hasModel: false,
        hasItem: Boolean(routeSearch.item),
      });
      if (action.kind === "none") {
        if (urlWorkflow === committedWorkflow) {
          lastSyncedWorkflow.current = urlWorkflow;
        }
        return;
      }
      if (action.kind === "intent") {
        // Left in the URL when refused, so it retries once busy releases.
        if (!transitionWorkflow(action.workflow)) return;
        lastSyncedWorkflow.current = action.workflow;
        // Only the page stays: ?task= and other leftovers go with the request.
        void navigateSelf({
          to: "/audio",
          search: { workflow: action.workflow },
          replace: true,
        });
        return;
      }
      lastSyncedWorkflow.current = action.workflow;
      pendingWrite.current = { from: urlWorkflow };
      void navigateSelf({
        to: "/audio",
        search: (prev) => ({ ...prev, workflow: action.workflow }),
        replace: true,
      });
      return;
    }
    pendingWrite.current = null;
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
    // The page stays in the URL, so a reload reopens it.
    const settledWorkflow = routedWorkflow ?? committedWorkflow;
    lastSyncedWorkflow.current = settledWorkflow;
    void navigateSelf({
      to: "/audio",
      search: { workflow: settledWorkflow },
      replace: true,
    });
  }, [
    active,
    busy,
    committedWorkflow,
    routeSearch.model,
    routeSearch.quant,
    routeSearch.ggufQuant,
    routeSearch.task,
    routeSearch.workflow,
    routeSearch.audioType,
    routeSearch.loadId,
    routeSearch.gguf,
    routeSearch.item,
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
