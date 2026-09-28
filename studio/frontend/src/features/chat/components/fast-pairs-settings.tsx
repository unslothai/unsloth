// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useId, useState, useSyncExternalStore } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  fastCandidateModels,
  modelCatalogVersion,
  subscribeModelCatalog,
} from "../model-catalog";
import { discoverFastPairs, type FastPair } from "../lib/fast-variants";

export function FastPairsSettings({
  models,
  pairs,
  autoDetect,
  onPairsChange,
  onAutoDetectChange,
}: {
  models: string[];
  pairs: FastPair[];
  autoDetect: boolean;
  onPairsChange: (pairs: FastPair[]) => void;
  onAutoDetectChange: (enabled: boolean) => void;
}) {
  const id = useId();
  const [standard, setStandard] = useState("");
  const [fast, setFast] = useState("");
  const [error, setError] = useState("");
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  const detected = discoverFastPairs(fastCandidateModels()).filter(
    (pair) => models.includes(pair.standard) || models.includes(pair.fast),
  );
  const add = () => {
    const base = standard.trim(),
      target = fast.trim();
    if (!base || !target || base === target) {
      setError("Choose two different model IDs.");
      return;
    }
    if (
      pairs.some((pair) =>
        [pair.standard, pair.fast].some(
          (model) => model === base || model === target,
        ),
      )
    ) {
      setError(
        "A model can belong to one Fast pair. Remove its existing pair first.",
      );
      return;
    }
    onPairsChange([...pairs, { standard: base, fast: target, source: "user" }]);
    setStandard("");
    setFast("");
    setError("");
  };
  return (
    <section className="space-y-3 rounded-xl border p-4">
      <h3 className="text-sm font-medium">Fast models</h3>
      <p className="text-xs text-muted-foreground">
        Native Fast tiers are discovered from OpenRouter endpoints. Companion
        models are detected only when the catalog explicitly names the same
        checkpoint. You can override detection with your own pair.
      </p>
      <label className="flex min-h-9 items-center justify-between gap-3 text-sm">
        Detect companion models automatically
        <Switch
          aria-label="Detect Fast companions automatically"
          checked={autoDetect}
          onCheckedChange={onAutoDetectChange}
        />
      </label>
      {autoDetect &&
        detected.map((pair) => (
          <details
            key={pair.fast}
            className="rounded-lg bg-muted/40 p-2 text-xs"
          >
            <summary className="cursor-pointer break-all">
              Detected: {pair.standard} → {pair.fast}
            </summary>
            <p className="mt-2 text-muted-foreground">{pair.evidence}</p>
            <a
              className="underline"
              href={pair.verificationSource}
              target="_blank"
              rel="noreferrer"
            >
              Provider listing
            </a>
          </details>
        ))}
      {autoDetect && detected.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No unambiguous companion found. Add a pair below.
        </p>
      ) : null}
      {pairs.map((pair) => (
        <div
          key={pair.standard}
          className="flex items-center justify-between gap-2 rounded-lg border p-2 text-xs"
        >
          <span className="min-w-0 break-all">
            {pair.standard} → {pair.fast}
            <span className="block text-muted-foreground">
              Your pair · overrides detection
            </span>
          </span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => onPairsChange(pairs.filter((item) => item !== pair))}
            aria-label={`Remove Fast pair for ${pair.standard}`}
          >
            Remove
          </Button>
        </div>
      ))}
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs">
          Standard model
          <Input
            aria-label="Standard model ID"
            list={`${id}-models`}
            value={standard}
            onChange={(event) => setStandard(event.target.value)}
            placeholder="author/model"
          />
        </label>
        <label className="space-y-1 text-xs">
          Fast companion
          <Input
            aria-label="Fast companion model ID"
            list={`${id}-models`}
            value={fast}
            onChange={(event) => setFast(event.target.value)}
            placeholder="author/fast-model"
          />
        </label>
        <datalist id={`${id}-models`}>
          {[...new Set(models)].map((model) => (
            <option key={model} value={model} />
          ))}
        </datalist>
      </div>
      {error ? (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      ) : null}
      <Button type="button" size="sm" variant="outline" onClick={add}>
        Add Fast pair
      </Button>
      <p className="text-xs text-muted-foreground">
        Switching enables the companion automatically. Fast pairs and detection
        preferences are saved in this browser with this connection.
      </p>
    </section>
  );
}
