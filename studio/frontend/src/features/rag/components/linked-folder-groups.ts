// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { listLinkedFolders } from "../api/rag-api";
import type { RagDocument } from "../types/rag";

/** Display names of a project's linked folders, read once per set of folder ids. */
export function useLinkedFolderNames(
  projectId: string | null,
  folderIds: string,
): ReadonlyMap<string, string> {
  const [names, setNames] = useState<ReadonlyMap<string, string>>(new Map());
  useEffect(() => {
    if (!projectId || !folderIds) return;
    let cancelled = false;
    listLinkedFolders({ type: "project", id: projectId })
      .then((folders) => {
        if (cancelled) return;
        setNames(new Map(folders.map((f) => [f.id, f.displayName])));
      })
      .catch(() => {
        // The card falls back to a generic label.
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, folderIds]);
  return names;
}

/** Splits project documents into loose files and per-folder groups. A linked folder can
 * hold thousands of files, so it is drawn as one item rather than one per file. */
export function groupByLinkedFolder<T extends RagDocument>(
  docs: T[],
): {
  loose: T[];
  folders: [string, T[]][];
} {
  const loose: T[] = [];
  const folders = new Map<string, T[]>();
  for (const doc of docs) {
    const folderId = doc.linkedFolderId;
    if (!folderId) {
      loose.push(doc);
      continue;
    }
    const group = folders.get(folderId);
    if (group) group.push(doc);
    else folders.set(folderId, [doc]);
  }
  return { loose, folders: [...folders] };
}
