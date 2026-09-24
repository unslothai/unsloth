// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";
import type { LibraryFolder, LibraryItem } from "./api";

export type LibraryTarget =
  | { kind: "item"; item: LibraryItem }
  | { kind: "folder"; folder: LibraryFolder };

/** Everything a card, row or menu can ask the page to do. */
export interface LibraryActions {
  folders: LibraryFolder[];
  openItem: (item: LibraryItem) => void;
  openFolder: (folderId: string) => void;
  chatAbout: (target: LibraryTarget) => void;
  toggleFavorite: (item: LibraryItem) => void;
  download: (item: LibraryItem) => void;
  rename: (target: LibraryTarget) => void;
  moveTo: (target: LibraryTarget, folderId: string | null) => void;
  moveToNewFolder: (target: LibraryTarget) => void;
  remove: (target: LibraryTarget) => void;
}

const LibraryActionsContext = createContext<LibraryActions | null>(null);

export const LibraryActionsProvider = LibraryActionsContext.Provider;

export function useLibraryActions(): LibraryActions {
  const actions = useContext(LibraryActionsContext);
  if (!actions) throw new Error("useLibraryActions outside LibraryActionsProvider");
  return actions;
}
