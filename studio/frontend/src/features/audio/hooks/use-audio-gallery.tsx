// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSettingsDialogStore } from "@/features/settings";
import { useStripReorder } from "@/hooks/use-strip-reorder";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  applyPin,
  moveGalleryItem,
  pinnedOrder,
  restorePinOrder,
  serializeById,
  sortGalleryItems,
  subscribeGalleryChanged,
} from "@/lib/gallery-flags";
import { toast } from "@/lib/toast";
import {
  type AudioGalleryClip,
  audioGalleryCursor,
  type AudioGalleryCursor,
  clearAudioGallery,
  deleteAudioClip,
  fetchClipObjectUrl,
  listAudioGallery,
  moveAudioClip,
  setAudioClipFlags,
} from "../api";
import { mergeGalleryPage } from "../audio-page-policy";
import {
  CLIP_BLOB_BUDGET_BYTES,
  MAX_PAGE_SIZE,
  PAGE_SIZE,
} from "../audio-workspace-constants";
import { clipWorkflow } from "../workflows";
import type { AudioHostState } from "./audio-host-state";

// Module scope so a tab switch re-renders the gallery instantly.
export const galleryCache: {
  clips: AudioGalleryClip[];
  hasMore: boolean;
  nextCursor: AudioGalleryCursor | null;
  selectedId: string | null;
  srcById: BlobUrlCache;
} = {
  clips: [],
  hasMore: false,
  nextCursor: null,
  selectedId: null,
  srcById: new BlobUrlCache(CLIP_BLOB_BUDGET_BYTES),
};

/** Speak and Music output: the clip gallery, its playback bytes, selection, ordering and history actions. */
export function useAudioGallery({
  active,
}: Pick<AudioHostState, "active">) {
  /** Audio the server produced that the gallery is not showing yet: either it could not be
   *  persisted, or this refresh missed it. Kept so the generation is playable either way. */
  const [fallbackClip, setFallbackClip] = useState<{
    url: string;
    prompt: string;
    model: string;
    saved: boolean;
  } | null>(null);
  const fallbackClipRef = useRef(fallbackClip);
  fallbackClipRef.current = fallbackClip;
  const loadingMoreRef = useRef(false);
  const galleryRefreshGeneration = useRef(0);
  // Pins and moves in flight. A refresh that overlaps one read the old order, so it is dropped and rerun after.
  const orderWrites = useRef({ inFlight: 0, epoch: 0, deferred: false });
  const [clips, setClips] = useState<AudioGalleryClip[]>(galleryCache.clips);
  const [hasMore, setHasMore] = useState(galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(
    galleryCache.selectedId,
  );
  const [srcById, setSrcById] = useState<Record<string, string>>(
    galleryCache.srcById.toRecord(),
  );
  const clipSrcLoads = useRef<Map<string, Promise<void>>>(new Map());

  const ensureClipSrc = useCallback(async (clip: AudioGalleryClip) => {
    const cached = galleryCache.srcById.get(clip.id);
    if (cached) {
      galleryCache.srcById.touch(clip.id);
      return;
    }
    const pending = clipSrcLoads.current.get(clip.id);
    if (pending) return pending;
    const load = (async () => {
      try {
        const fetched = await fetchClipObjectUrl(clip.url);
        // A delete can finish while protected bytes are in flight. Do not revive its cache entry after
        // the row is already gone.
        if (!galleryCache.clips.some((candidate) => candidate.id === clip.id)) {
          URL.revokeObjectURL(fetched.url);
          return;
        }
        galleryCache.srcById.set(clip.id, fetched.url, fetched.bytes);
        galleryCache.srcById.prune(
          galleryCache.selectedId ? [galleryCache.selectedId] : [],
        );
        setSrcById(galleryCache.srcById.toRecord());
      } catch {
        // Clip may have been deleted server-side; the next gallery refresh drops it.
        toast.error("Could not load this audio clip. Try selecting it again.");
      }
    })();
    clipSrcLoads.current.set(clip.id, load);
    try {
      await load;
    } finally {
      if (clipSrcLoads.current.get(clip.id) === load) {
        clipSrcLoads.current.delete(clip.id);
      }
    }
  }, []);

  const refreshGallery = useCallback(
    async (
      removedId?: string,
      windowSize = PAGE_SIZE,
    ): Promise<AudioGalleryClip[]> => {
      const generation = ++galleryRefreshGeneration.current;
      const writeEpoch = orderWrites.current.epoch;
      const wanted = Math.max(PAGE_SIZE, windowSize);
      const asked = Math.min(wanted, MAX_PAGE_SIZE);
      try {
        const page = await listAudioGallery(0, asked);
        // The caller's own fetch: a generation whose clip persisted must not be told otherwise.
        if (generation !== galleryRefreshGeneration.current) return page.audio;
        if (orderWrites.current.inFlight > 0 || orderWrites.current.epoch !== writeEpoch) {
          orderWrites.current.deferred = true;
          return page.audio;
        }
        // A window past the route's cap cannot be covered in one page, and stitching the old scrollback
        // back on keeps a cursor that starts BELOW it, stranding whatever was restored.
        const { clips: merged, stitched } =
          wanted > asked
            ? { clips: [...page.audio], stitched: false }
            : mergeGalleryPage(
                page.audio,
                galleryCache.clips,
                removedId,
                page.has_more,
              );
        galleryCache.clips = merged;
        // A clip record carries no mtime, so kept scrollback has no cursor; keep the deeper one.
        if (!stitched) {
          galleryCache.hasMore = page.has_more;
          galleryCache.nextCursor = audioGalleryCursor(page);
        }
        setClips(merged);
        setHasMore(galleryCache.hasMore);
        // The response audio was kept only until its record showed up: left mounted, deleting the
        // now-visible clip made the "saved, waiting for the gallery" copy reappear.
        if (
          fallbackClipRef.current &&
          galleryCache.selectedId &&
          merged.some((c) => c.id === galleryCache.selectedId)
        ) {
          setFallbackClip(null);
        }
        if (
          galleryCache.selectedId &&
          !merged.some((c) => c.id === galleryCache.selectedId)
        ) {
          galleryCache.selectedId = merged[0]?.id ?? null;
          setSelectedId(galleryCache.selectedId);
        }
        if (
          !galleryCache.selectedId &&
          !fallbackClipRef.current &&
          merged.length > 0
        ) {
          galleryCache.selectedId = merged[0].id;
          setSelectedId(galleryCache.selectedId);
        }
        const selected = merged.find(
          (clip) => clip.id === galleryCache.selectedId,
        );
        if (selected) void ensureClipSrc(selected);
        return merged;
      } catch {
        // Same recoverable-poll stance as status.
        return galleryCache.clips;
      }
    },
    [ensureClipSrc],
  );

  const loadMore = useCallback(async () => {
    // Repeated scroll events near the bottom would otherwise each fire with the same offset and
    // append the same page, duplicating clips and React keys.
    if (loadingMoreRef.current) return;
    loadingMoreRef.current = true;
    const refreshGeneration = galleryRefreshGeneration.current;
    const cursor = galleryCache.nextCursor;
    try {
      const page = await listAudioGallery(0, PAGE_SIZE, cursor);
      if (
        refreshGeneration !== galleryRefreshGeneration.current ||
        cursor !== galleryCache.nextCursor
      )
        return;
      galleryCache.nextCursor = audioGalleryCursor(page);
      const known = new Set(galleryCache.clips.map((clip) => clip.id));
      galleryCache.clips = [
        ...galleryCache.clips,
        ...page.audio.filter((clip) => !known.has(clip.id)),
      ];
      galleryCache.hasMore = page.has_more;
      setClips(galleryCache.clips);
      setHasMore(page.has_more);
    } catch {
      // Retry on the next scroll.
    } finally {
      loadingMoreRef.current = false;
    }
  }, []);


  useEffect(() => {
    if (!active) return;
    const refreshWhenVisible = () => {
      if (document.hidden) return;
      void refreshGallery(undefined, galleryCache.clips.length);
    };
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [active, refreshGallery]);

  useEffect(() => {
    const clip = clips.find((c) => c.id === selectedId);
    if (clip) void ensureClipSrc(clip);
  }, [clips, selectedId, ensureClipSrc]);

  /** `keepFallback` is for the one case where the id is not in `clips` yet: the server persisted the
   *  clip but this refresh missed it, so the response audio has to stay mounted or the player
   *  falls through to the empty state. */
  const selectClip = useCallback(
    (id: string, keepFallback = false) => {
      galleryCache.selectedId = id;
      setSelectedId(id);
      if (!keepFallback) setFallbackClip(null);
      const clip = galleryCache.clips.find((candidate) => candidate.id === id);
      if (clip) void ensureClipSrc(clip);
    },
    [ensureClipSrc],
  );

  const dropClip = useCallback((id: string) => {
    galleryCache.srcById.delete(id);
    setSrcById(galleryCache.srcById.toRecord());
    // Drop the row now, as the clear-all path does: refreshGallery swallows a failed GET and returns
    // the cache without setClips, leaving the row up against an already-revoked URL.
    galleryCache.clips = galleryCache.clips.filter((clip) => clip.id !== id);
    setClips(galleryCache.clips);
    if (galleryCache.selectedId === id) {
      galleryCache.selectedId = null;
      setSelectedId(null);
    }
  }, []);

  const handleDeleteClip = useCallback(
    async (id: string) => {
      try {
        await deleteAudioClip(id);
        dropClip(id);
        await refreshGallery(id);
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not delete the clip.",
        );
      }
    },
    [dropClip, refreshGallery],
  );

  const handleArchiveClip = useCallback(
    async (id: string) => {
      try {
        await setAudioClipFlags(id, { archived: true });
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not archive the clip.",
        );
        return;
      }
      dropClip(id);
      await refreshGallery(id);
      const toastId = toast(
        <button
          type="button"
          onClick={() => {
            toast.dismiss(toastId);
            useSettingsDialogStore.getState().openArchivedMedia("audio");
          }}
          className="w-full cursor-pointer text-left"
        >
          You can view archived audio in Settings
        </button>,
        { closeButton: true },
      );
    },
    [dropClip, refreshGallery],
  );

  // The pin state each id was last clicked or dragged into, so a stale response cannot undo a later one.
  const pinAttempt = useRef(new Map<string, number>());
  const pinSeq = useRef(0);

  const beginOrderWrite = useCallback(() => {
    orderWrites.current.inFlight += 1;
    orderWrites.current.epoch += 1;
  }, []);
  const endOrderWrite = useCallback(() => {
    const writes = orderWrites.current;
    writes.inFlight -= 1;
    writes.epoch += 1;
    if (writes.inFlight === 0 && writes.deferred) {
      writes.deferred = false;
      void refreshGallery(undefined, galleryCache.clips.length);
    }
  }, [refreshGallery]);

  const handleTogglePin = useCallback(async (id: string, pinned: boolean) => {
    // The pinned order before the click, so a failed unpin goes back where it was.
    const orderBefore = pinnedOrder(galleryCache.clips);
    const attempt = (pinSeq.current += 1);
    pinAttempt.current.set(id, attempt);
    // Optimistic. Records carry the server's sort key, so the local re-sort matches it.
    galleryCache.clips = applyPin(galleryCache.clips, id, pinned);
    setClips(galleryCache.clips);
    beginOrderWrite();
    try {
      // One queue for pins and moves: the server stamps pins in the order it runs them.
      await serializeById("audio-pin", () => setAudioClipFlags(id, { pinned }));
      // An unpinned clip can belong below the loaded window, so resync it once writes settle.
      if (!pinned && galleryCache.hasMore) orderWrites.current.deferred = true;
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not pin the clip.",
      );
      if (pinAttempt.current.get(id) === attempt) {
        galleryCache.clips = pinned
          ? applyPin(galleryCache.clips, id, false)
          : restorePinOrder(galleryCache.clips, id, orderBefore);
        setClips(galleryCache.clips);
      }
    } finally {
      if (pinAttempt.current.get(id) === attempt) pinAttempt.current.delete(id);
      endOrderWrite();
    }
  }, [beginOrderWrite, endOrderWrite]);

  // Drag to reorder: applied optimistically, then the server's record (key and pin) is adopted.
  const handleMoveClip = useCallback(
    async (id: string, afterId: string | null) => {
      const next = moveGalleryItem(galleryCache.clips, id, afterId);
      if (next === galleryCache.clips) return;
      const guessedPinned = Boolean(next.find((c) => c.id === id)?.pinned);
      // Takes a pin token too: a pin clicked after this drop must not be undone by its response.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      galleryCache.clips = next;
      setClips(next);
      beginOrderWrite();
      try {
        const record = await serializeById("audio-pin", () =>
          moveAudioClip(id, afterId),
        );
        if (pinAttempt.current.get(id) !== attempt) return;
        pinAttempt.current.delete(id);
        const patched = galleryCache.clips.map((c) =>
          c.id === id
            ? { ...c, pinned: record.pinned, order_at: record.order_at }
            : c,
        );
        // Re-sort only if the local pin guess was wrong.
        galleryCache.clips =
          Boolean(record.pinned) === guessedPinned
            ? patched
            : sortGalleryItems(patched);
        setClips(galleryCache.clips);
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not move the clip.",
        );
        if (pinAttempt.current.get(id) === attempt) pinAttempt.current.delete(id);
        // Put the server's order back once no other write is in flight.
        orderWrites.current.deferred = true;
      } finally {
        endOrderWrite();
      }
    },
    [beginOrderWrite, endOrderWrite],
  );
  const historyReorder = useStripReorder(
    (id, afterId) => void handleMoveClip(id, afterId),
    { axis: "y" },
  );

  // This page stays mounted across route changes, so a restore from the Settings archive would not reach
  // History until a reload. Refresh the loaded window, not just the first page: a clip re-enters at its own age.
  useEffect(
    () =>
      subscribeGalleryChanged("audio", () => {
        void refreshGallery(undefined, galleryCache.clips.length);
      }),
    [refreshGallery],
  );

  /** With a workflow, clears only that page's clips and leaves the other page's history alone. */
  const handleClearGallery = useCallback(async (workflow?: "speak" | "clone" | "convert" | "music") => {
    try {
      await clearAudioGallery(workflow);
      if (workflow) {
        const cleared = new Set(
          galleryCache.clips
            .filter((clip) => clipWorkflow(clip) === workflow)
            .map((clip) => clip.id),
        );
        for (const id of cleared) galleryCache.srcById.delete(id);
        if (galleryCache.selectedId && cleared.has(galleryCache.selectedId)) {
          galleryCache.selectedId = null;
        }
        galleryCache.clips = galleryCache.clips.filter(
          (clip) => !cleared.has(clip.id),
        );
        setClips(galleryCache.clips);
        setSrcById(galleryCache.srcById.toRecord());
        setSelectedId(galleryCache.selectedId);
        await refreshGallery();
        return;
      }
      galleryCache.srcById.clear();
      galleryCache.selectedId = null;
      // Drop the cached list first: refreshGallery merges the fetched page into it, so an empty page
      // would leave every cleared row on screen.
      galleryCache.clips = [];
      // React state too, not just the cache: refreshGallery swallows a failed GET and returns the cache
      // without calling setClips, which left cleared rows rendered against a revoked URL.
      setClips([]);
      setSrcById({});
      setSelectedId(null);
      await refreshGallery();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not clear the gallery.",
      );
    }
  }, [refreshGallery]);

  const handleDownloadClip = useCallback(
    (clip: AudioGalleryClip) => {
      const src = srcById[clip.id];
      if (!src) return;
      const anchor = document.createElement("a");
      anchor.href = src;
      anchor.download = `${clip.id}.wav`;
      anchor.click();
    },
    [srcById],
  );

  const handleDownloadFallbackClip = useCallback(() => {
    if (!fallbackClip) return;
    const anchor = document.createElement("a");
    anchor.href = fallbackClip.url;
    anchor.download = "generated-audio.wav";
    anchor.click();
  }, [fallbackClip]);

  /** Download from a history row, whose bytes are only fetched once selected. */
  const handleDownloadClipById = useCallback(async (clip: AudioGalleryClip) => {
    let temporaryUrl: string | null = null;
    try {
      let src = galleryCache.srcById.get(clip.id);
      if (!src) {
        const fetched = await fetchClipObjectUrl(clip.url);
        src = fetched.url;
        temporaryUrl = fetched.url;
      }
      const anchor = document.createElement("a");
      anchor.href = src;
      anchor.download = `${clip.id}.wav`;
      anchor.click();
    } catch {
      toast.error("Could not download the clip.");
    } finally {
      // A history-row download does not need to become resident playback state, so revoke it after the
      // browser has consumed the synthetic click rather than bypassing the 64 MB cache budget.
      if (temporaryUrl) {
        const url = temporaryUrl;
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    }
  }, []);

  const handleCopyPrompt = useCallback(async (text: string) => {
    if (await copyToClipboard(text)) {
      toast.success("Text copied");
    } else {
      toast.error("Could not copy the text.");
    }
  }, []);

  return {
    fallbackClip,
    setFallbackClip,
    loadingMoreRef,
    clips,
    hasMore,
    selectedId,
    setSelectedId,
    srcById,
    ensureClipSrc,
    refreshGallery,
    loadMore,
    selectClip,
    handleDeleteClip,
    handleArchiveClip,
    handleTogglePin,
    historyReorder,
    handleClearGallery,
    handleDownloadClip,
    handleDownloadFallbackClip,
    handleDownloadClipById,
    handleCopyPrompt,
  };
}

export type AudioGallery = ReturnType<typeof useAudioGallery>;

/** Below this many clips, a page tops its history up from the next gallery page. */
const HISTORY_MIN_VISIBLE = 8;

/** One page's slice of the shared gallery: its clips, its selection, and enough of them loaded
 *  to fill the list when the other page made most of the recent ones. */
export function useWorkflowHistory({
  workflow,
  enabled,
  clips,
  hasMore,
  selectedId,
  srcById,
  fallbackClip,
  loadMore,
  loadingMoreRef,
  selectClip,
}: { workflow: "speak" | "clone" | "convert" | "music"; enabled: boolean } & Pick<
  AudioGallery,
  | "clips"
  | "hasMore"
  | "selectedId"
  | "srcById"
  | "fallbackClip"
  | "loadMore"
  | "loadingMoreRef"
  | "selectClip"
>) {
  const visibleClips = useMemo(
    () => clips.filter((clip) => clipWorkflow(clip) === workflow),
    [clips, workflow],
  );
  const selectedClip =
    visibleClips.find((clip) => clip.id === selectedId) ?? null;
  const selectedClipSrc = selectedClip ? srcById[selectedClip.id] : undefined;

  // Arriving on a page whose clip is not the selected one selects its newest, as the gallery does on load.
  useEffect(() => {
    if (!enabled || selectedClip || fallbackClip) return;
    const first = visibleClips[0];
    if (first) selectClip(first.id);
  }, [enabled, selectedClip, fallbackClip, visibleClips, selectClip]);

  useEffect(() => {
    if (
      !enabled ||
      visibleClips.length >= HISTORY_MIN_VISIBLE ||
      !hasMore ||
      loadingMoreRef.current
    )
      return;
    void loadMore();
  }, [enabled, visibleClips, hasMore, loadMore, loadingMoreRef]);

  return { visibleClips, selectedClip, selectedClipSrc };
}
