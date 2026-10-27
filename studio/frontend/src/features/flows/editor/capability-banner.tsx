// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type CapabilityReport,
  approveCapabilities,
  getCapabilities,
} from "../api";
import { pendingItems } from "../capability-items";
import { useUiStore } from "../stores/ui-store";

export function CapabilityBanner({ flowId }: { flowId: string }) {
  const [report, setReport] = useState<CapabilityReport | null>(null);
  const [busy, setBusy] = useState(false);
  const saveState = useUiStore((s) => s.saveState);

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

  useEffect(() => {
    if (saveState === "saved") refresh();
  }, [saveState, refresh]);

  const items = pendingItems(report?.pending);
  if (!report || items.length === 0) return null;

  const approve = async () => {
    setBusy(true);
    try {
      const approved = await approveCapabilities(flowId, report.pending);
      if (current.current === flowId) setReport(approved);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not approve");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-[12px] text-sky-900 dark:text-sky-100"
      data-testid="flows-capabilities"
    >
      <span className="font-medium">This flow reaches outside Studio:</span>
      {items.map((item) => (
        <span
          key={item.key}
          className="rounded bg-background/70 px-1.5 py-0.5 text-foreground"
        >
          {item.label}
        </span>
      ))}
      {report.can_approve ? (
        <Button
          type="button"
          variant="dark"
          size="xs"
          disabled={busy}
          onClick={() => void approve()}
          className="ml-auto"
          data-testid="flows-capabilities-approve"
        >
          Approve and allow
        </Button>
      ) : (
        <span className="ml-auto text-sky-800/80 dark:text-sky-200/80">
          Only the Studio owner can approve these; until then those blocks do
          not run.
        </span>
      )}
    </div>
  );
}
