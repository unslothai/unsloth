// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import type { ReactElement, ReactNode } from "react";
import {
  CUSTOM_PROMPT_ID,
  PROMPT_SETS,
  SWEEP_BLURB,
  SWEEP_KINDS,
  SWEEP_TITLE,
  familyOf,
} from "../lib/bench-math";
import { useBenchmarksStore } from "../stores/benchmarks-store";
import { useFamilyColors } from "./family-colors";

function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }): ReactElement {
  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}): ReactElement {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-ui-11 text-muted-foreground">
      {label}
      <input
        type="number"
        inputMode="numeric"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
        className="h-9 w-full rounded-lg border border-border/60 bg-background px-3 text-sm tabular-nums text-foreground outline-none transition-shadow focus:ring-2 focus:ring-ring/40 disabled:opacity-50"
      />
    </label>
  );
}

function ToggleRow({ label, hint, checked, disabled, onChange }: { label: string; hint: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }): ReactElement {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex min-w-0 flex-col">
        <span className="text-ui-13 text-foreground">{label}</span>
        <span className="text-ui-11 leading-snug text-muted-foreground">{hint}</span>
      </div>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onChange} className="mt-0.5" />
    </div>
  );
}

export function SetupPanel({ maxContext, locked }: { maxContext: number | null; locked: boolean }): ReactElement {
  const config = useBenchmarksStore((s) => s.config);
  const disabled = useBenchmarksStore((s) => s.disabled);
  const setConfig = useBenchmarksStore((s) => s.setConfig);
  const choosePreset = useBenchmarksStore((s) => s.choosePreset);
  const toggleVariant = useBenchmarksStore((s) => s.toggleVariant);
  const colors = useFamilyColors();
  const active = config.variants.filter((v) => !disabled.includes(v.label));
  const perRow = config.warmup + config.repetitions;

  return (
    <aside className="flex flex-col gap-6 rounded-2xl border border-border/60 bg-card p-5">
      <Section title="Sweep">
        <div className="grid grid-cols-2 gap-2">
          {SWEEP_KINDS.map((kind) => {
            const selected = config.sweep === kind;
            return (
              <button
                key={kind}
                type="button"
                disabled={locked}
                onClick={() => choosePreset(kind, maxContext)}
                title={SWEEP_BLURB[kind]}
                className={cn(
                  "flex flex-col items-start gap-0.5 rounded-xl border px-3 py-2.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                  selected
                    ? "border-primary/50 bg-primary/8 ring-1 ring-primary/30"
                    : "border-border/60 hover:border-border hover:bg-muted/40",
                )}
              >
                <span className="text-ui-13 font-medium text-foreground">{SWEEP_TITLE[kind]}</span>
              </button>
            );
          })}
        </div>
        <p className="text-ui-11 leading-snug text-muted-foreground">{SWEEP_BLURB[config.sweep]}</p>
      </Section>

      <Section
        title="Settings to compare"
        aside={
          <span className="text-ui-11 tabular-nums text-muted-foreground">
            {active.length} of {config.variants.length} · {active.length * perRow} generations
          </span>
        }
      >
        <ul className="-mx-1 flex max-h-72 flex-col overflow-y-auto">
          {config.variants.map((v) => {
            const on = !disabled.includes(v.label);
            const isBase = config.baseline === v.label;
            return (
              <li key={v.label} className="group flex items-center gap-2 rounded-lg px-1 py-1 hover:bg-muted/40">
                <Switch size="sm" checked={on} disabled={locked} onCheckedChange={() => toggleVariant(v.label)} aria-label={`Include ${v.label}`} />
                <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: colors[familyOf(v.load)] }} aria-hidden={true} />
                <span className={cn("min-w-0 flex-1 truncate text-ui-12", on ? "text-foreground" : "text-muted-foreground line-through decoration-muted-foreground/40")}>
                  {v.label}
                </span>
                <button
                  type="button"
                  disabled={locked}
                  onClick={() => setConfig({ baseline: isBase ? null : v.label })}
                  className={cn(
                    "shrink-0 rounded-full px-2 py-0.5 text-ui-10 font-medium transition-colors",
                    isBase ? "bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] text-foreground" : "text-transparent group-hover:text-muted-foreground hover:bg-muted",
                  )}
                  title={isBase ? "Percentages are measured against this row" : "Measure the others against this row"}
                >
                  baseline
                </button>
              </li>
            );
          })}
        </ul>
      </Section>

      <Section title="Prompt">
        <div className="flex flex-wrap gap-1.5">
          {[...PROMPT_SETS.map((s) => ({ id: s.id, name: s.name, hint: s.hint })), { id: CUSTOM_PROMPT_ID, name: "Your own", hint: "Separate prompts with a --- line" }].map((p) => (
            <button
              key={p.id}
              type="button"
              disabled={locked}
              title={p.hint}
              onClick={() => setConfig({ promptSet: p.id })}
              className={cn(
                "rounded-full border px-3 py-1 text-ui-12 transition-colors disabled:opacity-60",
                config.promptSet === p.id ? "border-transparent bg-foreground text-background" : "border-border/60 text-foreground hover:bg-muted/50",
              )}
            >
              {p.name}
            </button>
          ))}
        </div>
        {config.promptSet === CUSTOM_PROMPT_ID ? (
          <textarea
            value={config.customPrompt}
            disabled={locked}
            onChange={(e) => setConfig({ customPrompt: e.target.value })}
            rows={5}
            placeholder={"Write a prompt.\n---\nAdd more, one per --- line, so runs can rotate."}
            className="w-full resize-y rounded-lg border border-border/60 bg-background px-3 py-2 text-ui-12 text-foreground outline-none focus:ring-2 focus:ring-ring/40"
          />
        ) : (
          <p className="text-ui-11 text-muted-foreground">{PROMPT_SETS.find((s) => s.id === config.promptSet)?.hint}, five prompts.</p>
        )}
      </Section>

      <Section title="Runs">
        <div className="grid grid-cols-3 gap-2">
          <NumberField label="Measured" value={config.repetitions} min={1} max={50} disabled={locked} onChange={(repetitions) => setConfig({ repetitions })} />
          <NumberField label="Warm-up" value={config.warmup} min={0} max={10} disabled={locked} onChange={(warmup) => setConfig({ warmup })} />
          <NumberField label="Max tokens" value={config.maxTokens} min={16} max={8192} step={16} disabled={locked} onChange={(maxTokens) => setConfig({ maxTokens })} />
        </div>
        <div className="mt-1 flex flex-col gap-3">
          <ToggleRow
            label="Rotate prompts"
            hint="Ngram drafting learns a repeated answer, so one prompt sent over and over inflates it."
            checked={config.rotatePrompts}
            disabled={locked}
            onChange={(rotatePrompts) => setConfig({ rotatePrompts })}
          />
          <ToggleRow
            label="Restore my settings after"
            hint="Reload the model the way it was before the sweep."
            checked={config.restoreAfter}
            disabled={locked}
            onChange={(restoreAfter) => setConfig({ restoreAfter })}
          />
        </div>
      </Section>
    </aside>
  );
}
