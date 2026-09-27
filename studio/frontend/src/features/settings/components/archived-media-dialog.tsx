// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AudioWave01Icon,
  Delete02Icon,
  Image03Icon,
  FlimSlateIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DEFAULT_LIBRARY_FILTERS,
  filterLibraryItems,
  type LibraryFilters,
} from "./data-library";
import { LibraryRow, LibraryToolbar } from "./data-library-controls";
import { Spinner } from "@/components/ui/spinner";
import {
  type AudioGalleryCursor,
  audioGalleryCursor,
  deleteAudioClip,
  listAudioGallery,
  setAudioClipFlags,
} from "@/features/audio/api";
import {
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  getGallery,
  setGalleryImageFlags,
} from "@/features/images/api";
import {
  deleteGalleryVideo,
  fetchGalleryVideoThumbnail,
  getVideoGallery,
  setGalleryVideoFlags,
} from "@/features/video/api";
import { videoThumbnailQueue } from "@/features/video/thumbnail-request-queue";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { notifyGalleryChanged } from "@/lib/gallery-flags";
import { translate, useLocale, useT } from "@/i18n";
import { toast } from "@/lib/toast";

/** Archived items shown per page; "Show more" pulls the next page. Matches ArchivedChatsView. */
const ARCHIVED_PAGE_SIZE = 20;
const SEARCH_PAGE_SIZE = 200;

// Blob budget for archived thumbnails. Far smaller than the gallery strip's 192 MB: these are 40px
// rows in a settings list, and only the loaded pages are ever on screen.
const ARCHIVED_THUMB_BUDGET_BYTES = 32 * 1024 * 1024;

// Retries for a thumbnail that failed to load, and the step between them. Capped so a row whose
// file is genuinely gone stops asking instead of retrying for as long as the dialog is open.
const THUMB_RETRY_LIMIT = 2;
const THUMB_RETRY_DELAY_MS = 750;

export type ArchivedMediaKind = "images" | "videos" | "audio";

/** The shape both galleries share, once flattened for this list. */
interface ArchivedRow {
  id: string;
  title: string;
  /** Epoch ms, so images (epoch seconds) and videos (ISO 8601) render the same way. */
  createdAt: number;
  /** Relative, auth-protected URL of the underlying file. */
  url: string;
}

interface ArchivedPage {
  rows: ArchivedRow[];
  hasMore: boolean;
  nextAudioCursor: AudioGalleryCursor | null;
}

/**
 * The archived shelf for one media gallery, modelled on ArchivedChatsView: rows with restore and
 * delete, revealed a page at a time. Unlike chats there is nothing readable to identify a result
 * by, so each row carries a thumbnail alongside its prompt.
 */
export function ArchivedMediaView({ kind }: { kind: ArchivedMediaKind }) {
  const t = useT();
  const locale = useLocale();
  const isImages = kind === "images";
  const isAudio = kind === "audio";
  const [rows, setRows] = useState<ArchivedRow[]>([]);
  // `showMore` reads the row count and the drop count from refs, not state: both can change while
  // its request is in flight, and a stale closure is exactly what makes it skip a row. The ref is
  // written with every list change rather than during render, so it is current the moment a drop
  // lands instead of one render later.
  const rowsRef = useRef<ArchivedRow[]>([]);
  const audioCursor = useRef<AudioGalleryCursor | null>(null);
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
  const [filters, setFilters] = useState<LibraryFilters>({
    ...DEFAULT_LIBRARY_FILTERS,
    sort: "default",
  });
  const [visibleCount, setVisibleCount] = useState(ARCHIVED_PAGE_SIZE);
  const [paging, setPaging] = useState(false);
  const [pageError, setPageError] = useState(false);
  const [bulkIntent, setBulkIntent] = useState<"delete" | "restore" | null>(
    null,
  );
  const [confirming, setConfirming] = useState<{
    rows: ArchivedRow[];
    action: "delete" | "restore";
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const running = useRef(false);
  const filtered = useMemo(
    () => filterLibraryItems(rows, filters, undefined, undefined, locale),
    [rows, filters, locale],
  );
  const displayed = useMemo(
    () => filtered.slice(0, visibleCount),
    [filtered, visibleCount],
  );
  const scanAll =
    filters.query.trim() !== "" ||
    filters.sort !== "default" ||
    bulkIntent !== null;

  function changeFilters(next: LibraryFilters) {
    setFilters(next);
    setVisibleCount(ARCHIVED_PAGE_SIZE);
    setPageError(false);
    setBulkIntent(null);
  }
  const [loading, setLoading] = useState(true);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  // Archived images and video posters are object URLs, so "Show more" a few times would otherwise
  // pin their bytes until unmount. Budget them like the main galleries do.
  const blobs = useRef(new BlobUrlCache(ARCHIVED_THUMB_BUDGET_BYTES));
  // Only rows on screen fetch a thumbnail, and only rows off screen are evicted. Together those
  // two rules keep memory bounded without ever blanking a row the user is looking at.
  const listRef = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState<ReadonlySet<string>>(new Set());

  const loadPage = useCallback(
    async (
      offset: number,
      before: AudioGalleryCursor | null = null,
      pageSize = ARCHIVED_PAGE_SIZE,
    ): Promise<ArchivedPage> => {
      if (isImages) {
        const page = await getGallery(offset, pageSize, true);
        return {
          rows: page.images.map((i) => ({
            id: i.id,
            title: i.prompt,
            createdAt: i.created_at * 1000,
            url: i.url,
          })),
          hasMore: page.has_more,
          nextAudioCursor: null,
        };
      }
      if (isAudio) {
        const page = await listAudioGallery(0, pageSize, before, true);
        return {
          rows: page.audio.map((a) => ({
            id: a.id,
            title: a.prompt,
            createdAt: Date.parse(a.created_at),
            url: a.url,
          })),
          hasMore: page.has_more,
          nextAudioCursor: audioGalleryCursor(page),
        };
      }
      const page = await getVideoGallery(offset, pageSize, true);
      return {
        rows: page.videos.map((v) => ({
          id: v.id,
          title: v.prompt,
          createdAt: Date.parse(v.created_at),
          url: v.url,
        })),
        hasMore: page.has_more,
        nextAudioCursor: null,
      };
    },
    [isImages, isAudio],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const page = await loadPage(0);
        if (cancelled) return;
        putRows(page.rows);
        audioCursor.current = page.nextAudioCursor;
        setHasMore(page.hasMore);
      } catch (err) {
        if (!cancelled) {
          setPageError(true);
          setHasMore(true);
          toast.error(translate("settings.data.library.loadFailed"), {
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
    setVisible((previous) =>
      new Set(
        displayed.filter((row) => previous.has(row.id)).map((row) => row.id),
      ),
    );
    let observing = true;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(new Set(displayed.map((r) => r.id)));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (!observing) return;
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
    for (const el of root.querySelectorAll("[data-archived-id]"))
      io.observe(el);
    return () => {
      observing = false;
      io.disconnect();
    };
  }, [displayed, loading]);

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
    if (isAudio) return;
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
          if (isImages || kind === "videos") {
            const { url, bytes } = isImages
              ? await fetchGalleryObjectUrl(row.url)
              : await videoThumbnailQueue.run(() =>
                  fetchGalleryVideoThumbnail(row.id),
                );
            // Dropped from the list, or the dialog closed: there is no row left to show it on,
            // and caching it after the unmount sweep would leak the blob.
            if (
              !alive.current ||
              !rowsRef.current.some((r) => r.id === row.id)
            ) {
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
  }, [rows, isImages, isAudio, kind, visible, retryTick]);

  // Drop a row, then top the page back up if that emptied it while more remain, so the list never
  // dead-ends with rows still unreachable behind a hidden "Show more".
  const dropRow = useCallback(
    (id: string) => {
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
    },
    [putRows],
  );

  async function handleRestore(row: ArchivedRow) {
    // Held for the whole round trip: the server shortens the shelf when it processes this, so a
    // page read inside that window sees it at the offset it captured with nothing to notice.
    mutations.current += 1;
    pendingMutations.current += 1;
    try {
      if (isImages) await setGalleryImageFlags(row.id, { archived: false });
      else if (isAudio) await setAudioClipFlags(row.id, { archived: false });
      else await setGalleryVideoFlags(row.id, { archived: false });
      dropRow(row.id);
      pendingMutations.current -= 1;
      return true;
    } catch (err) {
      pendingMutations.current -= 1;
      toast.error(t("settings.data.library.restoreFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
      return false;
    }
  }

  async function handleDelete(row: ArchivedRow) {
    // Held for the whole round trip: the server shortens the shelf when it processes this, so a
    // page read inside that window sees it at the offset it captured with nothing to notice.
    mutations.current += 1;
    pendingMutations.current += 1;
    try {
      if (isImages) await deleteGalleryImage(row.id);
      else if (isAudio) await deleteAudioClip(row.id);
      else await deleteGalleryVideo(row.id);
      dropRow(row.id);
      pendingMutations.current -= 1;
      return true;
    } catch (err) {
      pendingMutations.current -= 1;
      toast.error(t("settings.data.library.deleteFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
      return false;
    }
  }

  const showMore = useCallback(async () => {
    if (loadingMore.current || running.current) return;
    loadingMore.current = true;
    setPaging(true);
    setPageError(false);
    try {
      for (let attempt = 0; attempt < 4; attempt += 1) {
        const before = mutations.current;
        const page = await loadPage(
          rowsRef.current.length,
          audioCursor.current,
          scanAll ? SEARCH_PAGE_SIZE : ARCHIVED_PAGE_SIZE,
        );
        if (!alive.current) return;
        if (mutations.current !== before || pendingMutations.current > 0)
          continue;
        const seen = new Set(rowsRef.current.map((r) => r.id));
        const added = page.rows.filter((r) => !seen.has(r.id));
        if (page.hasMore && added.length === 0)
          throw new Error(t("settings.data.library.pageStalled"));
        putRows([...rowsRef.current, ...added]);
        audioCursor.current = page.nextAudioCursor;
        setHasMore(page.hasMore);
        return;
      }
      throw new Error(t("settings.data.library.pageChanged"));
    } catch (err) {
      if (!alive.current) return;
      setPageError(true);
      toast.error(t("settings.data.library.loadMoreFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      loadingMore.current = false;
      if (alive.current) setPaging(false);
    }
  }, [t, loadPage, putRows, scanAll]);

  useEffect(() => {
    if (!scanAll || loading || paging || pageError || busy || !hasMore) return;
    const timer = setTimeout(
      () => void showMore(),
      rows.length <= ARCHIVED_PAGE_SIZE ? 250 : 0,
    );
    return () => clearTimeout(timer);
  }, [
    scanAll,
    loading,
    paging,
    pageError,
    busy,
    hasMore,
    showMore,
    rows.length,
  ]);

  useEffect(() => {
    if (!bulkIntent || hasMore || loading || paging || pageError) return;
    setConfirming({ rows: [...filtered], action: bulkIntent });
    setBulkIntent(null);
  }, [bulkIntent, hasMore, loading, paging, pageError, filtered]);

  async function run(targets: ArchivedRow[], action: "delete" | "restore") {
    if (running.current) return;
    running.current = true;
    setBusy(true);
    let restored = false;
    try {
      for (const row of targets) {
        const succeeded =
          action === "delete"
            ? await handleDelete(row)
            : await handleRestore(row);
        if (!succeeded) break;
        if (action === "restore") restored = true;
      }
    } finally {
      // refresh the persistent gallery once, including after a partially successful batch.
      if (restored) notifyGalleryChanged(kind);
      running.current = false;
      setBusy(false);
      setConfirming(null);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <Spinner className="size-5 text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <LibraryToolbar
        filters={filters}
        onChange={changeFilters}
        placeholder={t(
          isImages
            ? "settings.data.library.searchImages"
            : isAudio
              ? "settings.data.library.searchAudio"
              : "settings.data.library.searchVideos",
        )}
        disabled={busy || bulkIntent !== null}
      />
      <div className="flex flex-wrap items-center gap-2">
        <span role="status" className="flex-1 text-xs text-muted-foreground">
          {pageError
            ? t("settings.data.library.incompleteSearch")
            : scanAll && hasMore
              ? t("settings.data.library.searchingRemaining", {
                  count: rows.length,
                })
              : t("settings.data.library.itemCount", {
                  count: `${filtered.length}${hasMore ? "+" : ""}`,
                })}
        </span>
        {bulkIntent ? (
          <Button variant="ghost" size="sm" onClick={() => setBulkIntent(null)}>
            {t("common.cancel")}
          </Button>
        ) : (
          <>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy || (rows.length === 0 && !hasMore)}
              onClick={() => setBulkIntent("restore")}
            >
              {filters.query.trim()
                ? t("settings.data.library.unarchiveResults")
                : t("settings.data.library.unarchiveAll")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy || (rows.length === 0 && !hasMore)}
              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setBulkIntent("delete")}
            >
              <HugeiconsIcon icon={Delete02Icon} className="mr-1.5 size-4" />
              {filters.query.trim()
                ? t("settings.data.library.deleteResults")
                : t("settings.data.deleteAllAction")}
            </Button>
          </>
        )}
      </div>
      <div
        ref={listRef}
        className="divide-y divide-border/50 rounded-2xl border border-border/60 px-3 sm:px-4"
      >
        {displayed.map((row) => (
          <div key={row.id} data-archived-id={row.id}>
            <LibraryRow
              title={row.title}
              date={row.createdAt}
              leading={
                <span className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-muted/40">
                  {!isAudio && thumbs[row.id] ? (
                    <img
                      src={thumbs[row.id]}
                      alt=""
                      className="size-full object-cover"
                    />
                  ) : (
                    <HugeiconsIcon
                      icon={
                        isAudio
                          ? AudioWave01Icon
                          : isImages
                            ? Image03Icon
                            : FlimSlateIcon
                      }
                      className="size-5 text-muted-foreground"
                    />
                  )}
                </span>
              }
              actions={
                <>
                  <Button
                    variant="ghost"
                    size="icon"
                    disabled={busy || bulkIntent !== null}
                    aria-label={t("settings.data.library.deleteItem", {
                      title: row.title,
                    })}
                    title={t("common.delete")}
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() =>
                      setConfirming({ rows: [row], action: "delete" })
                    }
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy || bulkIntent !== null}
                    aria-label={t("settings.data.library.unarchiveItem", {
                      title: row.title,
                    })}
                    className="rounded-xl bg-muted/60 hover:bg-muted"
                    onClick={() => void run([row], "restore")}
                  >
                    {t("settings.data.library.unarchive")}
                  </Button>
                </>
              }
            />
          </div>
        ))}
        {displayed.length === 0 && (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {hasMore
              ? t("settings.data.library.noLoadedMatches")
              : filters.query.trim()
                ? t("settings.data.library.noMediaMatches")
                : t("settings.data.library.noMedia")}
          </p>
        )}
      </div>
      {hasMore || filtered.length > visibleCount ? (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            disabled={paging || busy}
            onClick={() => {
              if (pageError) void showMore();
              else if (filtered.length > visibleCount)
                setVisibleCount((count) => count + ARCHIVED_PAGE_SIZE);
              else
                void showMore().then(() =>
                  setVisibleCount((count) => count + ARCHIVED_PAGE_SIZE),
                );
            }}
          >
            {paging
              ? t("common.loading")
              : pageError
                ? t("picker.retry")
                : t("shell.navigation.showMore")}
          </Button>
        </div>
      ) : null}
      <AlertDialog
        open={confirming !== null}
        onOpenChange={(open) => {
          if (!open && !busy) setConfirming(null);
        }}
      >
        <AlertDialogContent
          onEscapeKeyDown={(event) => {
            if (busy) event.preventDefault();
          }}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t(
                confirming?.action === "delete"
                  ? "settings.data.library.deleteItemsTitle"
                  : "settings.data.library.unarchiveItemsTitle",
                { count: confirming?.rows.length ?? 0 },
              )}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirming?.rows.length === 1
                ? `"${confirming.rows[0].title}". `
                : ""}
              {confirming?.action === "delete"
                ? t("settings.data.library.deleteFilesWarning")
                : t("settings.data.library.restoreWarning")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("common.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={busy || !confirming?.rows.length}
              variant={
                confirming?.action === "delete" ? "destructive" : "default"
              }
              onClick={(event) => {
                event.preventDefault();
                if (confirming) void run(confirming.rows, confirming.action);
              }}
            >
              {busy
                ? t("settings.data.library.working")
                : confirming?.action === "delete"
                  ? t("common.delete")
                  : t("settings.data.library.unarchive")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
