// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { useMessage, useThreadRuntime } from "@assistant-ui/react";
import {
  messageCost,
  readCostReceipts,
  sumCostReceipts,
} from "../lib/cost-receipts";
import { formatUsd, nonnegativeDecimal } from "../lib/model-pricing";

export function MessageCost({ onClick }: { onClick: () => void }) {
  const message = useMessage();
  const cost = messageCost(message.metadata?.custom);
  if (!cost.relevant) return null;
  return (
    <button
      type="button"
      data-slot="message-cost-trigger"
      onClick={onClick}
      className="h-8 px-2 text-xs text-muted-foreground tabular-nums"
      aria-label="Recorded OpenRouter cost details"
    >
      {cost.known ? formatUsd(cost.total) : "Cost unknown"}
      {cost.incomplete && cost.known ? " + ?" : ""}
    </button>
  );
}

export function RecordedChatCost() {
  const runtime = useThreadRuntime();
  const [label, setLabel] = useState("");
  useEffect(() => {
    const update = () => {
      // Export includes saved siblings, whereas getState().messages is only the visible branch.
      const costs = runtime
        .export()
        .messages.filter(({ message }) => message.role === "assistant")
        .map(({ message }) => messageCost(message.metadata?.custom));
      if (!costs.some((cost) => cost.relevant)) {
        setLabel("");
        return;
      }
      const total = sumCostReceipts(costs.flatMap((cost) => cost.receipts));
      const incomplete =
        total.incomplete || costs.some((cost) => cost.historical);
      setLabel(
        `Recorded chat cost ${total.known ? formatUsd(total.total) : "unknown"}${incomplete ? " · Incomplete" : ""}`,
      );
    };
    update();
    return runtime.subscribe(update);
  }, [runtime]);
  return label ? (
    <p
      className="px-3 pt-2 text-center text-xs text-muted-foreground tabular-nums"
      title="OpenRouter account charges across saved branches. Missing final charges are unknown; upstream BYOK charges are separate."
    >
      {label}
    </p>
  ) : null;
}

export function CostReceiptDetails({ custom }: { custom: unknown }) {
  const cost = messageCost(custom);
  if (!cost.relevant) return null;
  const receipts = readCostReceipts(
    (custom as { costReceipts?: unknown })?.costReceipts,
  );
  return (
    <section className="space-y-3 rounded-xl border p-3 text-xs">
      <h3 className="text-sm font-medium">Recorded OpenRouter charge</h3>
      <p className="text-lg tabular-nums">
        {cost.known ? formatUsd(cost.total) : "Unknown"}
        {cost.incomplete || cost.historical ? " · Incomplete" : ""}
      </p>
      {sumCostReceipts(receipts).receipts.map((r) => (
        <details key={r.generationId ?? r.attemptId} className="border-t pt-2">
          <summary className="cursor-pointer break-all py-1">
            {formatUsd(r.cost)} · {r.servedModel ?? r.requestedModel}
          </summary>
          <dl className="space-y-1 break-all pt-2">
            <dt>Generation</dt>
            <dd>{r.generationId ?? "Not received"}</dd>
            <dt>Requested model</dt>
            <dd>{r.requestedModel}</dd>
            <dt>Served model</dt>
            <dd>{r.servedModel ?? "Not received"}</dd>
            {r.usage?.cost_details &&
            typeof r.usage.cost_details === "object" ? (
              <>
                <dt>Upstream BYOK charge (separate)</dt>
                <dd>
                  {formatUsd(
                    nonnegativeDecimal(
                      (r.usage.cost_details as Record<string, unknown>)
                        .upstream_inference_cost,
                    ),
                  )}
                </dd>
              </>
            ) : null}
            <dt>Reported usage and charge details</dt>
            <dd>
              <pre className="whitespace-pre-wrap break-all text-[11px]">
                {JSON.stringify(r.usage ?? {}, null, 2)}
              </pre>
            </dd>
          </dl>
        </details>
      ))}
      <p className="text-muted-foreground">
        Final provider usage supplies the account charge. Missing charges are
        unknown. Upstream BYOK charges are separate from this total.
      </p>
    </section>
  );
}
