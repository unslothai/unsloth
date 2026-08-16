// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveRestoreIcon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  getGallery,
  setGalleryImageFlags,
} from "@/features/images/api";
import {
  deleteGalleryVideo,
  fetchGalleryVideoSignedUrl,
  getVideoGallery,
  setGalleryVideoFlags,
} from "@/features/video/api";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { notifyGalleryChanged } from "@/lib/gallery-flags";
import { toast } from "@/lib/toast";

/** Archived items shown per page; "Show more" pulls the next page. Matches ArchivedChatsView. */
const ARCHIVED_PAGE_SIZE = 20;

// Blob budget for archived thumbnails. Far smaller than the gallery strip's 192 MB: these are 40px
// rows in a settings list, and only the loaded pages are ever on screen.
const ARCHIVED_THUMB_BUDGET_BYTES = 32 * 1024 * 1024;

// Retries for a thumbnail that failed to load, and the step between them. Capped so a row whose
// file is genuinely gone stops asking instead of retrying for as long as the dialog is open.
const THUMB_RETRY_LIMIT = 2;
const THUMB_RETRY_DELAY_MS = 750;

export type ArchivedMediaKind = "images" | "videos";

/** The shape both galleries share, once flattened for this list. */
interface ArchivedRow {
  id: string;
  prompt: string;
  /** Epoch ms, so images (epoch seconds) and videos (ISO 8601) render the same way. */
  createdAtMs: number;
  /** Relative, auth-protected URL of the underlying file. */
  url: string;
}

function formatCreatedAt(ms: number): string {
  if (!Number.isFinite(ms)) return "";
  return new Date(ms).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

/**
 * The archived shelf for one media gallery, modelled on ArchivedChatsView: rows with restore and
 * delete, revealed a page at a time. Unlike chats there is nothing readable to identify a result
 * by, so each row carries a thumbnail alongside its prompt.
 */
export function ArchivedMediaView({ kind }: { kind: ArchivedMediaKind }) {
  const isImages = kind === "images";
  const noun = isImages ? "image" : "video";
  const [rows, setRows] = useState<ArchivedRow[]>([]);
  // `showMore` reads the row count and the drop count from refs, not state: both can change while
  // its request is in flight, and a stale closure is exactly what makes it skip a row. The ref is
  // written with every list change rather than during render, so it is current the moment a drop
  // lands instead of one render later.
  const rowsRef = useRef<ArchivedRow[]>([]);
  const mutations = useRef(0);
  // Restores and deletes in flight. The counter above is an EDGE, so a page starting after it moves
  // and landing before the row is dropped sees it hold still. A page applies only while this is zero.
  const pendingMutations = useRef(0);
  const loadingMore = useRef(false);
  const putRows = useCallback((next: ArchivedRow[]) => {
    rowsRef.current = next;
    setRows(next);
  }, []);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  // Archived PNGs are full-size generated images, so "Show more" a few times would pin hundreds of
  // MB if every blob were held to unmount. Budget them like the main gallery does. Images only: a
  // clip uses a signed link, which is not an object URL and must not be revoked.
  const blobs = useRef(new BlobUrlCache(ARCHIVED_THUMB_BUDGET_BYTES));
  // Only rows on screen fetch a thumbnail, and only rows off screen are evicted. Together those
  // two rules keep memory bounded without ever blanking a row the user is looking at.
  const listRef = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState<ReadonlySet<string>>(new Set());

  const loadPage = useCallback(
    async (offset: number) => {
      if (isImages) {
        const page = await getGallery(offset, ARCHIVED_PAGE_SIZE, true);
        return {
          rows: page.images.map((i) => ({
            id: i.id,
            prompt: i.prompt,
            createdAtMs: i.created_at * 1000,
            url: i.url,
          })),
          hasMore: page.has_more,
        };
      }
      const page = await getVideoGallery(offset, ARCHIVED_PAGE_SIZE, true);
      return {
        rows: page.videos.map((v) => ({
          id: v.id,
          prompt: v.prompt,
          createdAtMs: Date.parse(v.created_at),
          url: v.url,
        })),
        hasMore: page.has_more,
      };
    },
    [isImages],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const page = await loadPage(0);
        if (cancelled) return;
        putRows(page.rows);
        setHasMore(page.hasMore);
      } catch (err) {
        if (!cancelled) {
          toast.error(`Failed to load archived ${kind}`, {
            description: err instanceof Error ? err.message : undefined,
          });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadPage, kind, putRows]);

  // Revoke everything still cached, once, on unmount.
  useEffect(() => {
    const cache = blobs.current;
    return () => cache.clear();
  }, []);

  // Track which rows are actually on screen. Without a visibility signal (jsdom, old webviews)
  // every row counts as visible, which is the old eager behaviour rather than a blank list.
  useEffect(() => {
    const root = listRef.current;
    if (!root) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(new Set(rows.map((r) => r.id)));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        setVisible((prev) => {
          const next = new Set(prev);
          for (const entry of entries) {
            const id = (entry.target as HTMLElement).dataset.archivedId;
            if (!id) continue;
            if (entry.isIntersecting) next.add(id);
            else next.delete(id);
          }
          return next;
        });
      },
      // A little margin so a row is fetched just before it scrolls into view.
      { rootMargin: "200px 0px" },
    );
    for (const el of root.querySelectorAll("[data-archived-id]")) io.observe(el);
    return () => io.disconnect();
  }, [rows]);

  // Thumbnails for VISIBLE rows that do not have one yet. `requested` is a ref, not state, so a
  // landing thumbnail cannot re-enter this effect and refetch the rest.
  const requested = useRef<Set<string>>(new Set());
  // The latest visibility set. A fetch that lands after the user scrolled would otherwise prune
  // against the set its own effect run closed over, protecting a row that has since gone off
  // screen and evicting one that is on it.
  const visibleRef = useRef<ReadonlySet<string>>(visible);
  useEffect(() => {
    visibleRef.current = visible;
  }, [visible]);
  // Prune on VISIBILITY, not only after a fetch. Rows leaving the viewport are what makes their
  // blobs evictable, and at the end of a shelf nothing fetches again, so the budget stopped binding.
  useEffect(() => {
    const evicted = blobs.current.prune(visible);
    if (evicted.length === 0) return;
    for (const id of evicted) requested.current.delete(id);
    setThumbs((prev) => {
      const next = { ...prev };
      for (const id of evicted) delete next[id];
      return next;
    });
  }, [visible]);
  // Failed attempts per row. Clearing `requested` on a failure changes nothing this effect
  // watches, so a visible row would stay blank until the user happened to scroll it away and
  // back. The tick schedules the retry; the count stops a permanently broken row from looping.
  const failures = useRef(new Map<string, number>());
  const [retryTick, setRetryTick] = useState(0);
  // Only an unmount has to discard a fetch that already completed. A plain effect re-run (a
  // scroll, another page) leaves that work perfectly usable, and throwing it away is what left
  // rows blank: `requested` outlives the effect, so nothing would ever fetch them again.
  const alive = useRef(true);
  useEffect(() => {
    // Set on the way in, not just cleared on the way out: StrictMode runs setup, cleanup, setup in
    // development, so a flag only cleared by the cleanup would stay false for the rest of the
    // dialog's life and discard every thumbnail that landed.
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      for (const row of rows) {
        if (cancelled) return;
        if (!visible.has(row.id)) continue;
        if (requested.current.has(row.id)) continue;
        // Checked here too, not only where the retry is scheduled: a failure clears `requested`, so
        // any later run of this effect would refetch a permanently missing file without limit.
        if ((failures.current.get(row.id) ?? 0) > THUMB_RETRY_LIMIT) continue;
        requested.current.add(row.id);
        try {
          if (isImages) {
            const { url, bytes } = await fetchGalleryObjectUrl(row.url);
            // Dropped from the list, or the dialog closed: there is no row left to show it on,
            // and caching it after the unmount sweep would leak the blob.
            if (!alive.current || !rowsRef.current.some((r) => r.id === row.id)) {
              URL.revokeObjectURL(url);
              requested.current.delete(row.id);
              return;
            }
            blobs.current.set(row.id, url, bytes);
            // Successful load: forget the earlier failures, or a single transient one after an
            // eviction would land past the cap and leave the row blank with no retry scheduled.
            failures.current.delete(row.id);
            // Evict the coldest thumbnails back within budget, never one that is on screen. Since
            // only visible rows are fetched, an evicted row is off screen by definition; clearing
            // it from `requested` lets it fetch again when it scrolls back and this effect re-runs.
            const evicted = blobs.current.prune(visibleRef.current);
            setThumbs((prev) => {
              const next = { ...prev, [row.id]: url };
              for (const id of evicted) {
                delete next[id];
                requested.current.delete(id);
              }
              return next;
            });
            // Stale generation: stop iterating, but keep what this fetch already paid for.
            if (cancelled) return;
            continue;
          }
          // A clip is a short-lived signed link, not a blob: nothing to budget or revoke.
          const src = await fetchGalleryVideoSignedUrl(row.id);
          if (!alive.current || !rowsRef.current.some((r) => r.id === row.id)) {
            requested.current.delete(row.id);
            return;
          }
          failures.current.delete(row.id);
          setThumbs((prev) => ({ ...prev, [row.id]: src }));
          if (cancelled) return;
        } catch {
          // A missing thumbnail still leaves a usable, actionable row, so a failure is not fatal.
          // Schedule the retry rather than only clearing the flag, which nothing would act on.
          requested.current.delete(row.id);
          const attempts = (failures.current.get(row.id) ?? 0) + 1;
          failures.current.set(row.id, attempts);
          if (attempts <= THUMB_RETRY_LIMIT) {
            setTimeout(() => {
              if (alive.current) setRetryTick((tick) => tick + 1);
            }, THUMB_RETRY_DELAY_MS * attempts);
          }
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [rows, isImages, visible, retryTick]);

  // Drop a row, then top the page back up if that emptied it while more remain, so the list never
  // dead-ends with rows still unreachable behind a hidden "Show more".
  const dropRow = useCallback(
    (id: string, backToGallery: boolean) => {
      putRows(rowsRef.current.filter((r) => r.id !== id));
      // Every drop shifts the rows behind it up by one, so an offset taken before this point is
      // now short. `showMore` uses the counter to notice and re-page instead of skipping a row.
      mutations.current += 1;
      // Release the thumbnail with the row. Its element unmounts without the observer reporting it,
      // so the id would sit in `visible` forever and permanently protect its blob from eviction,
      // walking the cache past its budget one restore at a time.
      blobs.current.delete(id);
      requested.current.delete(id);
      setVisible((prev) => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setThumbs((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
      // The page that owns this gallery is mounted persistently and only loads on mount, so a
      // restore has to be announced or the strip stays stale until a reload. A delete does not:
      // the item was archived, so it was never on that strip, and refetching would only cost the
      // user the pages they had scrolled to.
      if (backToGallery) notifyGalleryChanged(kind);
    },
    [kind, putRows],
  );

  async function handleRestore(row: ArchivedRow) {
    // Held for the whole round trip: the server shortens the shelf when it processes this, so a
    // page read inside that window sees it at the offset it captured with nothing to notice.
    mutations.current += 1;
    pendingMutations.current += 1;
    try {
      if (isImages) await setGalleryImageFlags(row.id, { archived: false });
      else await setGalleryVideoFlags(row.id, { archived: false });
      dropRow(row.id, true);
      pendingMutations.current -= 1;
      toast.success(`${isImages ? "Image" : "Video"} restored`);
    } catch (err) {
      pendingMutations.current -= 1;
      toast.error(`Failed to restore ${noun}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleDelete(row: ArchivedRow) {
    // Held for the whole round trip: the server shortens the shelf when it processes this, so a
    // page read inside that window sees it at the offset it captured with nothing to notice.
    mutations.current += 1;
    pendingMutations.current += 1;
    try {
      if (isImages) await deleteGalleryImage(row.id);
      else await deleteGalleryVideo(row.id);
      dropRow(row.id, false);
      pendingMutations.current -= 1;
      toast.success(`${isImages ? "Image" : "Video"} deleted`);
    } catch (err) {
      pendingMutations.current -= 1;
      toast.error(`Failed to delete ${noun}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function showMore() {
    if (loadingMore.current) return;
    loadingMore.current = true;
    try {
      // Offset paging over a list the user can shorten. A restore or delete landing while this
      // request is in flight pulls every later row up by one, so the row at the old offset is
      // never returned by any page and becomes unreachable. Retry at the corrected offset when
      // that happens; the bound is only there so a burst of clicks cannot spin here.
      for (let attempt = 0; attempt < 4; attempt += 1) {
        const before = mutations.current;
        const page = await loadPage(rowsRef.current.length);
        if (mutations.current !== before || pendingMutations.current > 0) continue;
        const seen = new Set(rowsRef.current.map((r) => r.id));
        putRows([...rowsRef.current, ...page.rows.filter((r) => !seen.has(r.id))]);
        setHasMore(page.hasMore);
        return;
      }
    } catch (err) {
      toast.error(`Failed to load more archived ${kind}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      loadingMore.current = false;
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <Spinner className="size-5 text-muted-foreground" />
      </div>
    );
  }

  // Only a genuinely empty shelf ends here. Emptying the LOADED page while more remain keeps the
  // list rendered so "Show more" survives, else the rest become unreachable without reopening.
  if (rows.length === 0 && !hasMore) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No archived {kind}.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div ref={listRef}>
        <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
          <span className="w-10 shrink-0" />
          <span className="flex-1">Prompt</span>
          <span className="w-32 shrink-0">Date created</span>
          <span className="w-16 shrink-0" />
        </div>
        {rows.map((row) => (
          <div
            key={row.id}
            data-archived-id={row.id}
            className="group flex items-center gap-4 border-b border-border/40 px-1 py-2.5 text-sm last:border-0"
          >
            <span className="size-10 shrink-0 overflow-hidden rounded-md bg-muted/40">
              {thumbs[row.id] ? (
                isImages ? (
                  <img src={thumbs[row.id]} alt="" className="size-full object-cover" />
                ) : (
                  // Muted metadata-only poster, same treatment as the filmstrip cards.
                  <video
                    src={thumbs[row.id]}
                    muted={true}
                    playsInline={true}
                    preload="metadata"
                    className="size-full object-cover"
                  />
                )
              ) : null}
            </span>
            <span className="min-w-0 flex-1 truncate" title={row.prompt}>
              {row.prompt}
            </span>
            <span className="w-32 shrink-0 text-muted-foreground tabular-nums">
              {formatCreatedAt(row.createdAtMs)}
            </span>
            <span className="flex w-16 shrink-0 items-center justify-end gap-1">
              <button
                type="button"
                onClick={() => void handleRestore(row)}
                aria-label={`Restore ${noun}`}
                title="Restore"
                className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <HugeiconsIcon icon={ArchiveRestoreIcon} strokeWidth={1.75} className="size-4" />
              </button>
              <button
                type="button"
                onClick={() => void handleDelete(row)}
                aria-label={`Delete ${noun}`}
                title="Delete"
                className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              >
                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-4" />
              </button>
            </span>
          </div>
        ))}
        {hasMore ? (
          <div className="flex justify-center pt-3">
            <Button variant="outline" size="sm" onClick={() => void showMore()}>
              Show more
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
