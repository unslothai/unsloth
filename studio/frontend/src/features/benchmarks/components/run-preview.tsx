// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Train's run preview card: what Run will load and send, then the button.

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  type InferenceStatusResponse,
  useChatRuntimeStore,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon, Rocket01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement, ReactNode } from "react";
import { useChatSettings } from "../api/chat-base";
import {
  type BaseSetting,
  CUSTOM_PROMPT_ID,
  PROMPT_SETS,
  SWEEP_TITLE,
  modelShort,
  pinnedFields,
  variedFields,
} from "../lib/bench-math";
import { useBenchmarksStore } from "../stores/benchmarks-store";

function MetaRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
}): ReactElement {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="shrink-0 text-ui-11p5 text-muted-foreground/85">
        {label}
      </span>
      <span
        className={cn(
          "min-w-0 truncate text-ui-12p5 text-foreground/90",
          mono && "font-mono text-ui-12",
        )}
      >
        {value}
      </span>
    </div>
  );
}

/** Every Run Settings value chat will load with, the swept ones marked. */
export function SettingsPopover({
  settings,
  varied,
}: {
  settings: BaseSetting[];
  varied: Set<string>;
}): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          className="inline-flex items-center gap-1 text-ui-12p5 text-foreground/90 underline-offset-2 hover:underline"
        >
          All {settings.length}
          <HugeiconsIcon
            icon={ArrowRight01Icon}
            strokeWidth={1.75}
            className="size-3"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="max-h-[min(60vh,var(--radix-popover-content-available-height))] w-80 overflow-y-auto"
      >
        <p className="mb-2 text-xs font-semibold">From chat's Run Settings</p>
        <div className="flex flex-col gap-1">
          {settings.map((b) => (
            <div key={b.field} className="flex justify-between gap-4 text-xs">
              <span className="text-muted-foreground">{b.label}</span>
              <span
                className="truncate font-medium tabular-nums"
                title={b.value}
              >
                {varied.has(b.field) ? "swept" : b.value}
              </span>
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

const pinnedLabel = (v: string) =>
  /^\d+$/.test(v) ? Number(v).toLocaleString() : v;

const SHOWN_SETTINGS = ["max_seq_length", "cache_type_kv", "n_parallel"];

export function RunPreviewCard({
  status,
  onRun,
  onViewRun,
}: {
  status: InferenceStatusResponse | null;
  onRun: () => void;
  onViewRun: () => void;
}): ReactElement {
  const config = useBenchmarksStore((s) => s.config);
  const disabled = useBenchmarksStore((s) => s.disabled);
  const live = useBenchmarksStore((s) => s.live);
  const models = useChatRuntimeStore((s) => s.models);
  const chatSettings = useChatSettings();
  const settings = live?.run.base ?? chatSettings;
  const active = config.variants.filter((v) => !disabled.includes(v.label));
  const varied = variedFields(active);
  const pinned = pinnedFields(active);
  const tune = config.sweep === "tune";
  const tuneModel = config.tuneModel ?? null;

  const loaded = Boolean(status?.active_model) && status?.is_gguf !== false;
  const modelPath = live
    ? live.run.model
    : tuneModel
      ? (models.find((m) => m.id === tuneModel)?.name ?? tuneModel)
      : loaded
        ? status?.active_model
        : null;
  const variant = live
    ? live.run.ggufVariant
    : tuneModel
      ? (config.tuneVariant ?? null)
      : status?.gguf_variant;
  const hasModel = Boolean(modelPath);
  const ready = hasModel && active.length > 0;
  const perRow = config.warmup + config.repetitions;
  const prompts =
    config.promptSet === CUSTOM_PROMPT_ID
      ? "Your own"
      : (PROMPT_SETS.find((p) => p.id === config.promptSet)?.name ??
        config.promptSet);
  const shown = settings.filter((b) => SHOWN_SETTINGS.includes(b.field));

  return (
    <aside
      className={cn(
        "elevated-card elevated-card-dark-border flex flex-col gap-7 bg-[color-mix(in_oklab,var(--foreground)_calc(1.2%*var(--contrast-wash-gain,1)),transparent)] p-6",
        "dark:bg-[rgb(255_255_255_/_calc(0.018*var(--contrast-wash-gain,1)))]",
      )}
    >
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-ui-11 font-medium tracking-nav text-muted-foreground">
          Run preview
        </h2>
        <span
          className={cn(
            "inline-flex h-5 items-center rounded-full px-2 text-ui-10 font-medium tracking-nav",
            live || ready
              ? "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] text-foreground/90 dark:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]"
              : "bg-[color-mix(in_oklab,var(--foreground)_calc(3%*var(--contrast-wash-gain,1)),transparent)] text-muted-foreground/70 dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
          )}
        >
          {live ? "Running" : ready ? "Ready" : "Not ready"}
        </span>
      </header>

      <section className="flex flex-col gap-1">
        <p
          className={cn(
            "break-words font-heading text-xl font-semibold leading-[1.2] tracking-[-0.018em]",
            hasModel ? "text-foreground" : "text-muted-foreground/60",
          )}
          title={modelPath ?? undefined}
        >
          {hasModel
            ? modelShort(modelPath)
            : live
              ? "Reading the model"
              : "No GGUF loaded"}
        </p>
        <p className="truncate font-mono text-ui-12 text-muted-foreground">
          {hasModel
            ? `${variant ?? "GGUF"}${tuneModel && !live ? " · loads first" : ""}`
            : live
              ? " "
              : "Pick one under Setup"}
        </p>
      </section>

      <section className="flex flex-col gap-3">
        <MetaRow label="Sweep" value={SWEEP_TITLE[config.sweep]} />
        <MetaRow
          label="Rows"
          value={`${active.length} of ${config.variants.length}`}
        />
        <MetaRow
          label="Generations"
          value={
            <>
              <span className="font-mono">{active.length * perRow}</span>
              <span className="text-muted-foreground/70">
                {" "}
                · {config.maxTokens} tokens each
              </span>
            </>
          }
        />
        <MetaRow label="Baseline" value={config.baseline ?? "None"} />
        <MetaRow
          label="Prompts"
          value={`${prompts}${config.rotatePrompts ? " · rotated" : ""}`}
        />
      </section>

      <section className="flex flex-col gap-3">
        {shown.map((b) => (
          <MetaRow
            key={b.field}
            label={b.label}
            value={
              varied.has(b.field)
                ? "swept"
                : pinned.has(b.field)
                  ? `${pinnedLabel(pinned.get(b.field) ?? "")} (pinned)`
                  : b.value
            }
            mono={!varied.has(b.field)}
          />
        ))}
        <MetaRow
          label="Run Settings"
          value={<SettingsPopover settings={settings} varied={varied} />}
        />
        <MetaRow
          label="After the run"
          value={config.restoreAfter ? "Chat's settings restored" : "Left as the last row"}
        />
      </section>

      <div className="-mx-6 h-px bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]" />

      <Button
        size="lg"
        className={cn(
          "-mt-2 h-11 w-full justify-center rounded-xl text-ui-13p5 font-semibold tracking-tight",
          "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90",
          "disabled:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] disabled:text-muted-foreground disabled:shadow-none dark:disabled:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]",
          "transition-colors duration-200",
        )}
        onClick={live ? onViewRun : onRun}
        disabled={!live && !ready}
      >
        <HugeiconsIcon icon={Rocket01Icon} strokeWidth={1.75} className="size-4" />
        {live ? "View the run" : tune ? "Run auto-tune" : "Run benchmark"}
      </Button>
    </aside>
  );
}
