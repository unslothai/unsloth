// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import {
  exactModelPricing,
  modelCatalogVersion,
  subscribeModelCatalog,
} from "../model-catalog";
import { formatUsd, nonnegativeDecimal, tokenRate } from "../lib/model-pricing";

function usePricing(modelId: string) {
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  return exactModelPricing("openrouter", modelId);
}

export function ModelPriceLine({ modelId }: { modelId: string }) {
  const pricing = usePricing(modelId);
  return (
    <span className="text-muted-foreground text-xs tabular-nums">
      Input {tokenRate(pricing?.rates.prompt)} · Output{" "}
      {tokenRate(pricing?.rates.completion)} / 1M tokens
      {pricing?.overrides?.length ? " · Variable" : ""}
      {pricing?.cached ? " · Cached" : ""}
    </span>
  );
}

const UNITS: Record<string, [string, number, string]> = {
  prompt: ["Input", 1e6, "1M tokens"],
  completion: ["Output", 1e6, "1M tokens"],
  input_cache_read: ["Cache read", 1e6, "1M tokens"],
  input_cache_write: ["Cache write", 1e6, "1M tokens"],
  internal_reasoning: ["Reasoning", 1e6, "1M tokens"],
  request: ["Request", 1, "request"],
  image: ["Image", 1, "image"],
  web_search: ["Web search", 1, "operation"],
};

function RateDetails({ rates }: { rates: Record<string, unknown> }) {
  return (
    <dl className="space-y-1">
      {Object.entries(rates)
        .filter(([key]) => key in UNITS)
        .map(([key, value]) => {
          const [label, multiplier, unit] = UNITS[key];
          const rate = nonnegativeDecimal(value);
          return (
            <div key={key} className="flex justify-between gap-2">
              <dt>{label}</dt>
              <dd className="tabular-nums">
                {formatUsd(rate == null ? null : rate * multiplier)} / {unit}
              </dd>
            </div>
          );
        })}
    </dl>
  );
}

export function ModelPricing({ modelId }: { modelId: string }) {
  const pricing = usePricing(modelId);
  return (
    <details className="rounded-xl border border-border/70 bg-muted/30 text-xs text-muted-foreground">
      <summary className="cursor-pointer list-none rounded-xl p-3 focus-visible:outline-2 focus-visible:outline-ring">
        <span className="mb-2 flex justify-between">
          Published rates{" "}
          <span>
            {pricing?.cached ? "Cached · " : ""}
            {pricing?.overrides?.length ? "Variable · " : ""}Details
          </span>
        </span>
        <span className="grid grid-cols-2 gap-3">
          <span>
            Input
            <span className="mt-0.5 block text-sm tabular-nums text-foreground">
              {tokenRate(pricing?.rates.prompt)}
            </span>
          </span>
          <span>
            Output
            <span className="mt-0.5 block text-sm tabular-nums text-foreground">
              {tokenRate(pricing?.rates.completion)}
            </span>
          </span>
        </span>
        <span className="mt-1 block">USD / 1M tokens</span>
      </summary>
      <div className="space-y-3 border-t p-3">
        {pricing ? (
          <>
            <RateDetails rates={pricing.rates} />
            {pricing.overrides?.map((override, index) => (
              <div key={index} className="space-y-1 border-t pt-2">
                <p className="font-medium">
                  Variable rates — condition {index + 1}
                </p>
                {Object.entries(override)
                  .filter(([key]) => !(key in UNITS))
                  .map(([key, value]) => (
                    <p className="break-words" key={key}>
                      {key.replaceAll("_", " ")}:{" "}
                      {typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value)}
                    </p>
                  ))}
                <RateDetails rates={override} />
              </div>
            ))}
            <p>Fetched {new Date(pricing.fetchedAt).toLocaleString()}</p>
          </>
        ) : (
          <p>Published prices are unavailable for this exact model.</p>
        )}
        <p>Routing and request conditions can affect the final charge.</p>
        <a
          href="https://openrouter.ai/docs/guides/overview/models"
          target="_blank"
          rel="noreferrer"
          className="underline"
        >
          Source: OpenRouter Models API
        </a>
      </div>
    </details>
  );
}
