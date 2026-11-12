// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SubagentRecord } from "../types/subagents";
import { mergeSubagentDetail, mergeSubagentList } from "../utils/subagents";

/** Subagents the open chats spawned, plus which one the side panel shows. */
export interface SubagentsState {
  /** Summaries per parent thread, oldest first. Absent until that thread was polled once. */
  byThread: Record<string, readonly SubagentRecord[]>;
  /** Full records (with transcript) of the subagents whose detail view was opened. */
  detailById: Record<string, SubagentRecord>;
  /** The thread whose panel is open; the panel closes by itself when the chat changes. */
  openThreadId: string | null;
  selectedId: string | null;
  /** Threads whose panel already opened by itself, so closing it is never undone. */
  autoOpenedThreadIds: Record<string, true>;

  setList: (threadId: string, records: readonly SubagentRecord[]) => void;
  setDetail: (record: SubagentRecord) => void;
  openPanel: (threadId: string, subagentId?: string | null) => void;
  closePanel: () => void;
  select: (subagentId: string | null) => void;
  markAutoOpened: (threadId: string) => void;
  reset: () => void;
}

const INITIAL: Pick<
  SubagentsState,
  | "byThread"
  | "detailById"
  | "openThreadId"
  | "selectedId"
  | "autoOpenedThreadIds"
> = {
  byThread: {},
  detailById: {},
  openThreadId: null,
  selectedId: null,
  autoOpenedThreadIds: {},
};

export const useSubagentsStore = create<SubagentsState>()((set) => ({
  ...INITIAL,
  setList: (threadId, records) =>
    set((state) => {
      const merged = mergeSubagentList(state.byThread[threadId], records);
      if (merged === state.byThread[threadId]) return state;
      return { byThread: { ...state.byThread, [threadId]: merged } };
    }),
  setDetail: (record) =>
    set((state) => {
      const merged = mergeSubagentDetail(state.detailById[record.id], record);
      if (merged === state.detailById[record.id]) return state;
      return { detailById: { ...state.detailById, [record.id]: merged } };
    }),
  openPanel: (threadId, subagentId) =>
    set((state) => ({
      openThreadId: threadId,
      // A card asks for one subagent; otherwise the view the user left stays. The panel falls back
      // to the list when that id does not belong to this thread.
      selectedId: subagentId === undefined ? state.selectedId : subagentId,
    })),
  closePanel: () =>
    set((state) =>
      state.openThreadId === null ? state : { openThreadId: null },
    ),
  select: (selectedId) => set({ selectedId }),
  markAutoOpened: (threadId) =>
    set((state) => ({
      autoOpenedThreadIds: { ...state.autoOpenedThreadIds, [threadId]: true },
    })),
  reset: () => set({ ...INITIAL }),
}));
