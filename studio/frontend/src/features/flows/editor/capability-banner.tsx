// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type CapabilityReport,
  approveCapabilities,
  getCapabilities,
} from "../api";
import { pendingItems } from "../capability-items";
import { useUiStore } from "../stores/ui-store";
import { InformationCircleIcon } from "./icon-data";

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
    <Alert
      className="rounded-none border-x-0 border-t-0 py-2"
      data-testid="flows-capabilities"
    >
      <HugeiconsIcon icon={InformationCircleIcon} />
      <AlertTitle className="text-xs">This flow reaches outside Studio</AlertTitle>
      <AlertDescription className="flex flex-wrap items-center gap-1.5 text-xs">
        {items.map((item) => (
          <Badge key={item.key} variant="outline">
            {item.label}
          </Badge>
        ))}
        {!report.can_approve && (
          <span>
            Only the Studio owner can approve these; until then those blocks do
            not run.
          </span>
        )}
      </AlertDescription>
      {report.can_approve && (
        <AlertAction className="top-1/2 -translate-y-1/2">
          <Button
            type="button"
            variant="dark"
            size="xs"
            disabled={busy}
            onClick={() => void approve()}
            data-testid="flows-capabilities-approve"
          >
            Approve and allow
          </Button>
        </AlertAction>
      )}
    </Alert>
  );
}
