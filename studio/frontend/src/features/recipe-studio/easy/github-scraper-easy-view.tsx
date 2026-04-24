// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { GithubIcon, PlayCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FieldLabel } from "../dialogs/shared/field-label";
import { GithubRepoSeedForm } from "../dialogs/seed/seed-dialog";
import type { ModelConfig, NodeConfig, SeedConfig } from "../types";

type GithubScraperEasyViewProps = {
  configs: Record<string, NodeConfig>;
  rows: number;
  setRows: (rows: number) => void;
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  onRun: () => void;
  runLoading: boolean;
  runErrors: string[];
  onSwitchToAdvanced: () => void;
};

export function GithubScraperEasyView({
  configs,
  rows,
  setRows,
  updateConfig,
  onRun,
  runLoading,
  runErrors,
  onSwitchToAdvanced,
}: GithubScraperEasyViewProps): ReactElement {
  const seedConfig = useMemo(
    () =>
      Object.values(configs).find((c): c is SeedConfig => c.kind === "seed") ??
      null,
    [configs],
  );
  const modelConfig = useMemo(
    () =>
      Object.values(configs).find(
        (c): c is ModelConfig => c.kind === "model_config",
      ) ?? null,
    [configs],
  );

  const handleSeedUpdate = (patch: Partial<SeedConfig>): void => {
    if (!seedConfig) return;
    updateConfig(seedConfig.id, patch);
  };

  const handleModelChange = (value: string): void => {
    if (!modelConfig) return;
    updateConfig(modelConfig.id, { model: value });
  };

  if (!seedConfig) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <p className="text-sm text-muted-foreground">
          This recipe has no seed node. Switch to{" "}
          <button
            type="button"
            className="underline hover:text-foreground"
            onClick={onSwitchToAdvanced}
          >
            Advanced
          </button>{" "}
          to configure it.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-2xl flex-col gap-4 overflow-y-auto px-6 py-6">
      <div className="flex items-start gap-3">
        <div
          className="flex size-10 shrink-0 items-center justify-center rounded-lg corner-squircle border border-border/70 bg-muted/20"
          aria-hidden={true}
        >
          <HugeiconsIcon icon={GithubIcon} className="size-5" />
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-base font-semibold">GitHub Scraper</h2>
          <p className="text-xs text-muted-foreground">
            Scrape real GitHub issues and PRs and turn each thread into a{" "}
            <code>{"{user_request, grounded_response}"}</code> training pair.
            Defaults use the server's <code>GH_TOKEN</code> env var and the
            bundled local model.
          </p>
        </div>
      </div>

      <GithubRepoSeedForm config={seedConfig} onUpdate={handleSeedUpdate} />

      <section className="space-y-3 border-t border-border/60 pt-4">
        <h3 className="text-xs font-semibold uppercase text-muted-foreground">
          Run settings
        </h3>
        <div className="grid gap-3 sm:grid-cols-[minmax(0,10rem)_minmax(0,1fr)]">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Rows to generate"
              hint="How many training pairs the LLM should produce."
            />
            <Input
              type="number"
              className="nodrag"
              min={1}
              max={10000}
              value={rows}
              onChange={(event) => {
                const next = Number.parseInt(event.target.value, 10);
                setRows(Number.isFinite(next) && next > 0 ? next : 1);
              }}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Model"
              hint="OpenAI-compatible model id. Local GGUFs run on the bundled llama-server."
            />
            <Input
              className="nodrag font-mono text-xs"
              value={modelConfig?.model ?? ""}
              onChange={(event) => handleModelChange(event.target.value)}
              placeholder="unsloth/gemma-4-E2B-it-GGUF"
              disabled={!modelConfig}
            />
          </div>
        </div>
      </section>

      {runErrors.length > 0 && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          <p className="font-semibold">Cannot run:</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {runErrors.slice(0, 4).map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex items-center justify-between gap-3 pt-1">
        <button
          type="button"
          className="text-xs text-muted-foreground underline hover:text-foreground"
          onClick={onSwitchToAdvanced}
        >
          Advanced (drag and drop canvas)
        </button>
        <Button type="button" size="lg" onClick={onRun} disabled={runLoading}>
          <HugeiconsIcon icon={PlayCircleIcon} className="size-4" />
          {runLoading ? "Running..." : "Run"}
        </Button>
      </div>
    </div>
  );
}
