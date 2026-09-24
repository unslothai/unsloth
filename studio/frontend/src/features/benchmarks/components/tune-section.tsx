// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Auto-tune's verdict: the setting it picked, and the button that hands it to chat.

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo } from "react";
import {
  type BenchRun,
  aggregate,
  familyOf,
  fmtRate,
  modelShort,
  tuneVerdict,
} from "../lib/bench-math";
import { useBenchmarksStore } from "../stores/benchmarks-store";
import { useFamilyColors } from "./family-colors";

/** The verdict card above a finished auto-tune run. */
export function TuneVerdictCard({
  run,
}: {
  run: BenchRun;
}): ReactElement | null {
  const applying = useBenchmarksStore((s) => s.applying);
  const applyToChat = useBenchmarksStore((s) => s.applyToChat);
  const live = useBenchmarksStore((s) => s.live);
  const colors = useFamilyColors();
  const verdict = useMemo(
    () =>
      tuneVerdict(
        aggregate(run.results, run.config.variants, null),
        run.config.variants,
      ),
    [run],
  );
  if (!verdict) return null;
  const variant = run.config.variants.find(
    (v) => v.label === verdict.pick.label,
  );
  const color = colors[variant ? familyOf(variant.load) : "other"];
  const finished = run.finishedAt !== null;
  const busy = applying === run.id;
  const args = variant?.load.llama_extra_args ?? [];

  return (
    <section className="corner-squircle relative flex flex-col gap-5 rounded-3xl bg-card p-6 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
      <div className="relative flex flex-wrap items-end justify-between gap-5">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
            {finished ? "Auto-tune picked" : "Leading so far"}
          </span>
          <span className="flex items-center gap-2.5 font-heading text-ui-25 font-semibold tracking-[-0.02em] text-foreground">
            <span
              className="size-3 shrink-0 rounded-[4px]"
              style={{ background: color }}
              aria-hidden={true}
            />
            {verdict.pick.label}
          </span>
          <span className="text-ui-12p5 text-muted-foreground">
            <span className="font-semibold tabular-nums text-foreground">
              {fmtRate(verdict.pick.mean)}
            </span>
            {verdict.gain !== null && verdict.off && (
              <>
                {" "}
                · {verdict.gain >= 0 ? "+" : ""}
                {Math.round(verdict.gain)}% over speculation off (
                {fmtRate(verdict.off.mean)})
              </>
            )}
            {verdict.fastest && (
              <>
                {" "}
                · {verdict.fastest.label} was {fmtRate(verdict.fastest.mean)},
                inside the noise, so the simpler setting wins
              </>
            )}
          </span>
        </div>
        <Button
          onClick={() => void applyToChat(run)}
          disabled={!finished || busy || Boolean(live)}
          className={cn("h-10 gap-2 rounded-full px-5", busy && "opacity-80")}
        >
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-4" />
          {busy ? "Reloading chat…" : "Apply to chat"}
        </Button>
      </div>
      {variant && (
        <dl className="relative grid grid-cols-2 gap-x-6 gap-y-1.5 text-ui-12 sm:grid-cols-4">
          <Row
            label="Speculative decoding"
            value={String(variant.load.speculative_type ?? "auto")}
          />
          <Row
            label="Draft tokens"
            value={
              variant.load.spec_draft_n_max == null
                ? "auto"
                : String(variant.load.spec_draft_n_max)
            }
          />
          <Row
            label="Extra llama-server args"
            value={args.length ? args.join(" ") : "none"}
            mono={args.length > 0}
          />
          <Row
            label="Model"
            value={modelShort(run.config.tuneModel ?? run.model)}
          />
        </dl>
      )}
    </section>
  );
}

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col">
      <dt className="text-ui-10p5 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
      </dt>
      <dd
        className={cn(
          "truncate text-foreground/90",
          mono && "font-mono text-ui-11p5",
        )}
        title={value}
      >
        {value}
      </dd>
    </div>
  );
}
