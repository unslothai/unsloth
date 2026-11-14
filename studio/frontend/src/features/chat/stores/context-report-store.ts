// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { create } from "zustand";

/** Only message identities and edit revisions are needed to retire an open card.
 *  Keep media and prompt text out of the UI store. */
export function contextReportBranch(
  messages: readonly ThreadMessage[],
): string {
  return JSON.stringify(
    messages.map((m) => [m.id, m.metadata?.custom?.revision]),
  );
}

interface ContextReportRequest {
  threadKey: string;
  branch: string;
  checkpoint: string;
  nonce: number;
}

export const useContextReportStore = create<{
  report: ContextReportRequest | null;
  showContext: (threadKey: string, branch: string, checkpoint: string) => void;
  dismissContext: () => void;
}>((set) => ({
  report: null,
  showContext: (threadKey, branch, checkpoint) =>
    set((state) => ({
      report: {
        threadKey,
        branch,
        checkpoint,
        nonce: (state.report?.nonce ?? 0) + 1,
      },
    })),
  dismissContext: () => set({ report: null }),
}));

/** The report this Thread shows: its own, and only while nothing has landed after it. */
export function shownContextReport(
  report: ContextReportRequest | null,
  threadKey: string,
  branch: string,
  checkpoint: string,
): ContextReportRequest | null {
  return report?.threadKey === threadKey &&
    report.branch === branch &&
    report.checkpoint === checkpoint
    ? report
    : null;
}
