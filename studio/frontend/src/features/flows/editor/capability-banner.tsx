// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type CapabilityReport,
  approveCapabilities,
  getCapabilities,
} from "../api";
import { pendingItems } from "../capability-items";
import { useUiStore } from "../stores/ui-store";

/** Asks the owner to approve what the flow reaches outside itself (Python, hosts, MCP, triggers) before it runs. */
export function CapabilityBanner({ flowId }: { flowId: string }) {
  const [report, setReport] = useState<CapabilityReport | null>(null);
  const [busy, setBusy] = useState(false);
  const saveState = useUiStore((s) => s.saveState);

  // Only the answer for the flow on screen counts: a late one for the last flow must not be approved here.
  const current = useRef(flowId);
  current.current = flowId;
  const refresh = useCallback(() => {
    getCapabilities(flowId)
      .then((found) => current.current === flowId && setReport(found))
      .catch(() => current.current === flowId && setReport(null));
  }, [flowId]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: reset on identity change only
  useEffect(() => {
    setReport(null);
  }, [flowId]);

  // Edits change what is needed (new code, a new host), so read it again after every save.
  useEffect(() => {
    if (saveState === "saved") refresh();
  }, [saveState, refresh]);

  const items = pendingItems(report?.pending);
  if (!report || items.length === 0) return null;

  const approve = async () => {
    setBusy(true);
    try {
      const approved = await approveCapabilities(flowId, report.pending);
      // The user may have opened another flow meanwhile; its banner must keep its own report.
      if (current.current === flowId) setReport(approved);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not approve");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-[12px] text-sky-100"
      data-testid="flows-capabilities"
    >
      <span className="font-medium">This flow reaches outside Studio:</span>
      {items.map((item) => (
        <span
          key={item.key}
          className="rounded bg-white/10 px-1.5 py-0.5 text-sky-50"
        >
          {item.label}
        </span>
      ))}
      {report.can_approve ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => void approve()}
          className="ml-auto rounded-md bg-sky-500/80 px-2.5 py-1 text-[12px] font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          data-testid="flows-capabilities-approve"
        >
          Approve and allow
        </button>
      ) : (
        <span className="ml-auto text-sky-200/80">
          Only the Studio owner can approve these; until then those blocks do
          not run.
        </span>
      )}
    </div>
  );
}
