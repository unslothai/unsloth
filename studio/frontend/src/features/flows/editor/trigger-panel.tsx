// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { getApiBase } from "@/lib/api-base";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { type FlowTrigger, listTriggers, rotateTrigger } from "../api";
import { webhookUrl } from "../capability-items";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";

export function TriggerPanel({
  nodeId,
  kind,
}: { nodeId: string; kind: string }) {
  const flowId = useDocStore((s) => s.flowId);
  const saveState = useUiStore((s) => s.saveState);
  const [trigger, setTrigger] = useState<FlowTrigger | null>(null);
  const [ownerOnly, setOwnerOnly] = useState(false);

  // biome-ignore lint/correctness/useExhaustiveDependencies: reset on identity change only
  useEffect(() => {
    setTrigger(null);
  }, [flowId, nodeId]);

  useEffect(() => {
    if (!flowId || saveState !== "saved") return;
    let alive = true;
    listTriggers(flowId)
      .then((found) => {
        if (!alive) return;
        setOwnerOnly(found.owner_only);
        setTrigger(found.triggers.find((t) => t.node_id === nodeId) ?? null);
      })
      .catch(() => alive && setTrigger(null));
    return () => {
      alive = false;
    };
  }, [flowId, nodeId, saveState]);

  const url = trigger?.path
    ? webhookUrl(getApiBase() || window.location.origin, trigger.path)
    : null;
  return (
    <section
      className="flex flex-col gap-2 border-t border-border px-4 py-3 text-[12px] text-muted-foreground"
      data-testid="flows-trigger-panel"
    >
      <p>
        Runs only while Unsloth Studio is open; a time missed while it was
        closed is not made up. It starts on its own once you approve triggers
        for this flow.
      </p>
      {ownerOnly && <p>Only the Studio owner's flows can run on their own.</p>}
      {trigger?.cron_error && (
        <p className="text-amber-700 dark:text-amber-300">Cron: {trigger.cron_error}</p>
      )}
      {kind === "trigger.webhook" && url && flowId && (
        <div className="flex flex-col gap-1.5">
          <span className="font-medium text-foreground">
            POST a JSON body to:
          </span>
          <code className="break-all rounded bg-muted px-2 py-1 text-[11px] text-foreground">
            {url}
          </code>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() =>
                void navigator.clipboard
                  .writeText(url)
                  .then(() => toast("Webhook URL copied"))
              }
            >
              Copy URL
            </Button>
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() =>
                void rotateTrigger(flowId, nodeId)
                  .then((found) => {
                    setTrigger(
                      found.triggers.find((t) => t.node_id === nodeId) ?? null,
                    );
                    toast("New secret; the old URL no longer works");
                  })
                  .catch((error: unknown) =>
                    toast.error(
                      error instanceof Error
                        ? error.message
                        : "Could not rotate",
                    ),
                  )
              }
            >
              New secret
            </Button>
          </div>
        </div>
      )}
      {kind === "trigger.webhook" && !url && !ownerOnly && (
        <p>Save the flow to get its webhook URL.</p>
      )}
    </section>
  );
}
