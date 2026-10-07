// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Netscape bookmark HTML, as Chrome, Edge, Firefox and Safari export it.
// Folders are <H3> headings, each followed by a <DL> of links.

import { downloadFile } from "@/lib/native-files";
import { isWebUrl } from "./address";
import { type BookmarkFolder, useBrowserBookmarksStore } from "./bookmarks-store";

// Toolbar folder names, for files that don't flag one.
const TOOLBAR_NAMES = /^(bookmarks bar|bookmarks toolbar|bookmark bar|favorites bar|favourites bar|favorites)$/i;
// Far past MAX_BOOKMARKS; anything bigger isn't a bookmarks file.
const MAX_FILE_BYTES = 20 * 1024 * 1024;

export class BookmarksFileError extends Error {}

function isToolbarHeading(heading: Element): boolean {
  return (
    heading.getAttribute("personal_toolbar_folder") === "true" ||
    TOOLBAR_NAMES.test(heading.textContent?.trim() ?? "")
  );
}

/** A bookmarks file's web links, in order; toolbar folder links go on the toolbar. */
export function parseBookmarksHtml(html: string): { url: string; title: string; folder: BookmarkFolder; addedAt?: number }[] {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const links = [...doc.querySelectorAll("a[href]")];
  if (links.length === 0 && !/<!doctype netscape-bookmark-file/i.test(html)) {
    throw new BookmarksFileError("not a bookmarks file");
  }
  return links.flatMap((link) => {
    const url = link.getAttribute("href")?.trim() ?? "";
    if (!isWebUrl(url)) return [];
    let toolbar = false;
    for (let list = link.closest("dl"); list; list = list.parentElement?.closest("dl") ?? null) {
      const heading = list.previousElementSibling;
      if (heading?.tagName === "H3" && isToolbarHeading(heading)) {
        toolbar = true;
        break;
      }
    }
    // ADD_DATE is in seconds.
    const added = Number(link.getAttribute("add_date"));
    return [
      {
        url,
        title: link.textContent?.trim() || url,
        folder: toolbar ? "toolbar" : "other",
        // Out of Date's range, it would throw when the Bookmarks page formats it.
        addedAt: added > 0 && !Number.isNaN(new Date(added * 1000).getTime()) ? added * 1000 : undefined,
      },
    ];
  });
}

/** Adds a bookmarks file's links; how many were added, and how many didn't fit. */
export async function importBookmarksFile(file: File): Promise<{ added: number; leftOut: number }> {
  if (file.size > MAX_FILE_BYTES) throw new BookmarksFileError("file too large");
  const items = parseBookmarksHtml(await file.text());
  return useBrowserBookmarksStore.getState().importBookmarks(items);
}

const escapeHtml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export function bookmarksHtml(): string {
  const { bookmarks } = useBrowserBookmarksStore.getState();
  const entries = (folder: BookmarkFolder) =>
    bookmarks
      .filter((bookmark) => bookmark.folder === folder)
      .map(
        (bookmark) =>
          `        <DT><A HREF="${escapeHtml(bookmark.url)}" ADD_DATE="${Math.floor(bookmark.addedAt / 1000)}">${escapeHtml(bookmark.title || bookmark.url)}</A>`,
      )
      .join("\n");
  const now = Math.floor(Date.now() / 1000);
  return `<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3 ADD_DATE="${now}" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks bar</H3>
    <DL><p>
${entries("toolbar")}
    </DL><p>
    <DT><H3 ADD_DATE="${now}">Other bookmarks</H3>
    <DL><p>
${entries("other")}
    </DL><p>
</DL><p>
`;
}

/** Saves the bookmarks as an HTML file other browsers can import. */
export function exportBookmarksFile(): Promise<void> {
  return downloadFile(bookmarksHtml(), "unsloth-bookmarks.html", "text/html");
}
