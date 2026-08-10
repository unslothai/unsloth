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
import { toast } from "@/lib/toast";

/** Archived items shown per page; "Show more" pulls the next page. Matches ArchivedChatsView. */
const ARCHIVED_PAGE_SIZE = 20;

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
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  // Object URLs this view minted, revoked together on unmount. Images only: a clip uses a signed
  // link, which is not an object URL and must not be revoked.
  const objectUrls = useRef<string[]>([]);

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
        setRows(page.rows);
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
  }, [loadPage, kind]);

  // Revoke every object URL this view minted, once, on unmount.
  useEffect(() => {
    const minted = objectUrls.current;
    return () => {
      for (const url of minted) URL.revokeObjectURL(url);
      minted.length = 0;
    };
  }, []);

  // Thumbnails for rows that do not have one yet. Bounded by the page size, so this never holds
  // more than a screenful plus whatever "Show more" added. `requested` is a ref, not state, so a
  // landing thumbnail cannot re-enter this effect and refetch the rest.
  const requested = useRef<Set<string>>(new Set());
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      for (const row of rows) {
        if (cancelled) return;
        if (requested.current.has(row.id)) continue;
        requested.current.add(row.id);
        try {
          const src = isImages
            ? (await fetchGalleryObjectUrl(row.url)).url
            : await fetchGalleryVideoSignedUrl(row.id);
          if (cancelled) {
            if (isImages) URL.revokeObjectURL(src);
            return;
          }
          if (isImages) objectUrls.current.push(src);
          setThumbs((prev) => ({ ...prev, [row.id]: src }));
        } catch {
          // A missing thumbnail still leaves a usable, actionable row. Allow a retry on the next
          // pass rather than pinning the failure forever.
          requested.current.delete(row.id);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [rows, isImages]);

  async function handleRestore(row: ArchivedRow) {
    try {
      if (isImages) await setGalleryImageFlags(row.id, { archived: false });
      else await setGalleryVideoFlags(row.id, { archived: false });
      setRows((prev) => prev.filter((r) => r.id !== row.id));
      toast.success(`${isImages ? "Image" : "Video"} restored`);
    } catch (err) {
      toast.error(`Failed to restore ${noun}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleDelete(row: ArchivedRow) {
    try {
      if (isImages) await deleteGalleryImage(row.id);
      else await deleteGalleryVideo(row.id);
      setRows((prev) => prev.filter((r) => r.id !== row.id));
      toast.success(`${isImages ? "Image" : "Video"} deleted`);
    } catch (err) {
      toast.error(`Failed to delete ${noun}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function showMore() {
    try {
      const page = await loadPage(rows.length);
      setRows((prev) => {
        const seen = new Set(prev.map((r) => r.id));
        return [...prev, ...page.rows.filter((r) => !seen.has(r.id))];
      });
      setHasMore(page.hasMore);
    } catch (err) {
      toast.error(`Failed to load more archived ${kind}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <Spinner className="size-5 text-muted-foreground" />
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No archived {kind}.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
          <span className="w-10 shrink-0" />
          <span className="flex-1">Prompt</span>
          <span className="w-32 shrink-0">Date created</span>
          <span className="w-16 shrink-0" />
        </div>
        {rows.map((row) => (
          <div
            key={row.id}
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
