// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";

type Place = { id: string; name: string };

/** The open chat and what its menu can do to it. The sidebar holds the handlers (its dialogs,
 *  its router, the shortcuts that already act on the open chat), so it publishes them here for
 *  the chat header's menu rather than the header carrying a second copy of each. */
export interface ActiveChatMenu {
  item: SidebarItem;
  pinned: boolean;
  unread: boolean;
  /** Off for a comparison, while generating, or while another fork runs. */
  canFork: boolean;
  /** Projects and sections to move to, leaving out the one the chat is in. */
  projects: Place[];
  sections: Place[];
  /** Where the chat is now, offered as "Remove from"; null when it is in none. */
  project: Place | null;
  section: Place | null;
  rename: () => void;
  togglePin: () => void;
  toggleUnread: () => void;
  fork: () => void;
  moveToProject: (id: string | null) => void;
  newProject: () => void;
  moveToSection: (id: string | null) => void;
  newSection: () => void;
  copyMarkdown: () => void;
  copySessionId: () => void;
  archive: () => void;
  remove: () => void;
}

export const useActiveChatMenuStore = create<{ menu: ActiveChatMenu | null }>(() => ({
  menu: null,
}));
