// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * The /context report shown under the last message. UI only: it is never a message, so it is
 * neither saved nor sent. `threadKey` is the assistant-ui thread list item id, so only the Thread
 * that asked shows it when compare panes mount several; `lastMessageId` is the message it was
 * counted under, and the report goes once the conversation moves past it. `nonce` recounts a
 * repeated /context.
 */
interface ContextReportRequest {
  threadKey: string;
  lastMessageId: string | null;
  nonce: number;
}

export const useContextReportStore = create<{
  report: ContextReportRequest | null;
  showContext: (threadKey: string, lastMessageId: string | null) => void;
  dismissContext: () => void;
}>((set) => ({
  report: null,
  showContext: (threadKey, lastMessageId) =>
    set((state) => ({
      report: {
        threadKey,
        lastMessageId,
        nonce: (state.report?.nonce ?? 0) + 1,
      },
    })),
  dismissContext: () => set({ report: null }),
}));

/** The report this Thread shows: its own, and only while nothing has landed after it. */
export function shownContextReport(
  report: ContextReportRequest | null,
  threadKey: string,
  lastMessageId: string | null,
): ContextReportRequest | null {
  return report?.threadKey === threadKey &&
    report.lastMessageId === lastMessageId
    ? report
    : null;
}
