// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The setup side panel and the model strip, built from the pieces Train and the Hub use:
// Train's field labels and Selects, the Hub's stat pills, the shared Checkbox and Input.

import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  type InferenceStatusResponse,
  useChatRuntimeStore,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  Cancel01Icon,
  PlusSignIcon,
  CpuIcon,
  SidebarLeft01Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useMemo, useState } from "react";
import { useChatSettings } from "../api/chat-base";
import {
  type BenchRun,
  CUSTOM_PROMPT_ID,
  FAMILY_LABEL,
  type Family,
  PROMPT_SETS,
  SWEEP_BLURB,
  SWEEP_KINDS,
  SWEEP_TITLE,
  type SweepKind,
  type Variant,
  familyOf,
  modelShort,
  variedFields,
} from "../lib/bench-math";
import { type BenchKind, useBenchmarksStore } from "../stores/benchmarks-store";
import { useFamilyColors } from "./family-colors";

/** The Hub toolbar pill: icon, bold value, quiet label. */
export function StatPill({
  icon,
  value,
  label,
  title,
}: {
  icon?: IconSvgElement;
  value: ReactNode;
  label?: string;
  title?: string;
}): ReactElement {
  return (
    <span
      title={title}
      className="inline-flex h-8 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-[color-mix(in_srgb,var(--foreground)_calc(8%*var(--contrast-edge-gain,1)),transparent)] bg-[color-mix(in_srgb,var(--foreground)_calc(2.5%*var(--contrast-wash-gain,1)),transparent)] px-3 text-ui-12 leading-none text-muted-foreground dark:bg-[color-mix(in_srgb,var(--foreground)_calc(6.5%*var(--contrast-wash-gain,1)),transparent)]"
    >
      {icon && (
        <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
      )}
      <span className="font-semibold tabular-nums text-foreground">
        {value}
      </span>
      {label && <span>{label}</span>}
    </span>
  );
}

/** Train's field: small-caps label with an info hint, then the control. */
function Field({
  label,
  hint,
  aside,
  children,
}: {
  label: string;
  hint?: string;
  aside?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
          {label}
          {hint && <InfoHint>{hint}</InfoHint>}
        </span>
        {aside}
      </div>
      {children}
    </div>
  );
}

/** Model and chat's key Run Settings as Hub pills; the full list is a click away. */
export function ModelStrip({
  status,
  run,
}: {
  status: InferenceStatusResponse | null;
  /** A live run: show what it captured, not the status of whichever row is loading. */
  run?: BenchRun | null;
}): ReactElement {
  const config = useBenchmarksStore((s) => s.config);
  const disabled = useBenchmarksStore((s) => s.disabled);
  const chatSettings = useChatSettings();
  const settings = run?.base ?? chatSettings;
  const varied = variedFields(
    config.variants.filter((v) => !disabled.includes(v.label)),
  );
  const model = run ? run.model : status?.active_model;
  const variant = run ? run.ggufVariant : status?.gguf_variant;
  const ready = Boolean(model) && (run ? true : status?.is_gguf !== false);
  const pick = [
    "max_seq_length",
    "cache_type_kv",
    "speculative_type",
    "spec_draft_n_max",
    "n_parallel",
  ];
  const shown = settings.filter((b) => pick.includes(b.field));

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <StatPill
        icon={CpuIcon}
        value={
          ready
            ? modelShort(model)
            : run
              ? "Reading the model"
              : "No GGUF loaded"
        }
        label={
          ready ? (variant ?? undefined) : run ? undefined : "load one in chat"
        }
      />
      {ready &&
        shown.map((b) => (
          <StatPill
            key={b.field}
            value={varied.has(b.field) ? "swept" : b.value}
            label={b.label}
          />
        ))}
      {ready && (
        <Popover>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              className="inline-flex h-8 items-center gap-1 rounded-full px-2.5 text-ui-12 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              All {settings.length} settings
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            className="max-h-[60vh] w-80 overflow-y-auto"
          >
            <p className="mb-2 text-xs font-semibold">
              From chat's Run Settings
            </p>
            <div className="flex flex-col gap-1">
              {settings.map((b) => (
                <div
                  key={b.field}
                  className="flex justify-between gap-4 text-xs"
                >
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
      )}
    </div>
  );
}

/** Filled for on, a ring for off. The colour is the family's, so it doubles as the legend. */
function Dot({
  color,
  filled,
}: {
  color: string;
  filled: boolean;
}): ReactElement {
  return (
    <span
      className="size-2.5 shrink-0 rounded-full border-[1.5px] transition-colors"
      style={
        filled
          ? { background: color, borderColor: color }
          : { background: "transparent", borderColor: color, opacity: 0.6 }
      }
      aria-hidden={true}
    />
  );
}

/** The family prefix is already on the group header, so the row drops it. */
function shortLabel(label: string, family: Family): string {
  const prefix = `${FAMILY_LABEL[family]} \u00b7 `;
  return label.startsWith(prefix) ? label.slice(prefix.length) : label;
}

function VariantRow({
  label,
  title,
  color,
  active,
  isBase,
  onToggle,
  onBaseline,
}: {
  label: string;
  title: string;
  color: string;
  active: boolean;
  isBase: boolean;
  onToggle: () => void;
  onBaseline: () => void;
}): ReactElement {
  return (
    <li className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-muted/60">
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={active}
        aria-label={`Include ${title}`}
        className="grid size-5 shrink-0 place-items-center rounded-full"
      >
        <Dot color={color} filled={active} />
      </button>
      <button
        type="button"
        onClick={onToggle}
        className={cn(
          "min-w-0 flex-1 truncate text-left text-ui-12p5",
          active ? "text-foreground" : "text-muted-foreground",
        )}
        title={title}
      >
        {label}
      </button>
      <button
        type="button"
        disabled={!active}
        onClick={onBaseline}
        title={isBase ? "Baseline" : "Make this the baseline"}
        className={cn(
          "grid size-6 shrink-0 place-items-center rounded-full transition-colors",
          isBase
            ? "text-foreground"
            : "text-transparent group-hover:text-muted-foreground/60 hover:!text-foreground",
        )}
      >
        <HugeiconsIcon
          icon={StarIcon}
          strokeWidth={1.75}
          className={cn("size-3.5", isBase && "fill-current")}
        />
      </button>
    </li>
  );
}

const KIND_OPTIONS = [
  { value: "sweep", label: "Config sweep" },
  { value: "tune", label: "Auto-tune" },
] as const;

const PROMPT_SEP = "\n---\n";

/** Your own prompts, one field each. Stored as the --- separated text the runner reads. */
function PromptListEditor({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}): ReactElement {
  const list = value.split(/\r?\n\s*---\s*\r?\n/);
  const items = list.length ? list : [""];
  const write = (next: string[]) => onChange(next.join(PROMPT_SEP));
  return (
    <div className="flex flex-col gap-2">
      {items.map((text, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: prompts have no identity beyond their slot
        <div key={i} className="group relative">
          <Textarea
            value={text}
            rows={2}
            placeholder={
              i === 0 ? "Explain how a hash map works." : "Another prompt"
            }
            onChange={(e) =>
              write(items.map((t, j) => (j === i ? e.target.value : t)))
            }
            className="min-h-0 resize-y pr-8 text-ui-12p5"
          />
          {items.length > 1 && (
            <button
              type="button"
              onClick={() => write(items.filter((_, j) => j !== i))}
              aria-label={`Remove prompt ${i + 1}`}
              className="absolute right-1.5 top-1.5 grid size-6 place-items-center rounded-full text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100"
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          )}
        </div>
      ))}
      <button
        type="button"
        onClick={() => write([...items, ""])}
        disabled={items.length >= 12}
        className="inline-flex h-8 items-center justify-center gap-1.5 rounded-full border border-dashed border-border text-ui-12 text-muted-foreground transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:text-foreground disabled:opacity-50"
      >
        <HugeiconsIcon
          icon={PlusSignIcon}
          strokeWidth={1.75}
          className="size-3.5"
        />
        Add prompt
      </button>
      <p className="text-ui-11p5 text-muted-foreground">
        {items.filter((t) => t.trim()).length} prompt
        {items.filter((t) => t.trim()).length === 1 ? "" : "s"}, rotated across
        runs.
      </p>
    </div>
  );
}

const LOADED = "\u0000loaded";

export function SetupPanel({
  status,
  maxContext,
  locked,
  onCollapse,
}: {
  status: InferenceStatusResponse | null;
  maxContext: number | null;
  locked: boolean;
  onCollapse: () => void;
}): ReactElement {
  const config = useBenchmarksStore((s) => s.config);
  const disabled = useBenchmarksStore((s) => s.disabled);
  const setConfig = useBenchmarksStore((s) => s.setConfig);
  const chooseKind = useBenchmarksStore((s) => s.chooseKind);
  const choosePreset = useBenchmarksStore((s) => s.choosePreset);
  const toggleVariant = useBenchmarksStore((s) => s.toggleVariant);
  const models = useChatRuntimeStore((s) => s.models);
  const ggufs = useMemo(() => models.filter((m) => m.isGguf), [models]);
  const colors = useFamilyColors();
  const kind: BenchKind = config.sweep === "tune" ? "tune" : "sweep";
  const on = config.variants.filter((v) => !disabled.includes(v.label)).length;
  const perRow = config.warmup + config.repetitions;
  const allOn = on === config.variants.length;
  const groups = useMemo(() => {
    const out: { family: Family; rows: Variant[] }[] = [];
    for (const v of config.variants) {
      const family = familyOf(v.load);
      const group = out.find((g) => g.family === family);
      if (group) group.rows.push(v);
      else out.push({ family, rows: [v] });
    }
    return out;
  }, [config.variants]);
  // Collapsed by default: 22 rows of detail is not what you open the page for.
  const [opened, setOpened] = useState<Family[]>([]);
  const prompts = [
    ...PROMPT_SETS.map((p) => ({ id: p.id, name: p.name, hint: p.hint })),
    {
      id: CUSTOM_PROMPT_ID,
      name: "Your own",
      hint: "Put a --- line between prompts so runs can rotate.",
    },
  ];

  return (
    <aside className="corner-squircle flex max-h-[calc(100vh-3rem)] flex-col overflow-hidden rounded-3xl bg-card ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
      <header className="flex items-center justify-between gap-2 px-5 pb-1 pt-4">
        <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
          Setup
        </span>
        <button
          type="button"
          onClick={onCollapse}
          aria-label="Hide setup"
          title="Hide setup"
          className="grid size-7 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <HugeiconsIcon
            icon={SidebarLeft01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </header>
      <fieldset
        disabled={locked}
        className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-5 pb-5 pt-3 disabled:opacity-60"
      >
        <Tabs
          value={kind}
          onValueChange={(v) => chooseKind(v as BenchKind)}
          className="contents"
        >
          <SegmentedTabsList
            value={kind}
            options={KIND_OPTIONS}
            ariaLabel="Benchmark type"
            size="compact"
            className="w-full"
          />
        </Tabs>

        {kind === "sweep" ? (
          <Field
            label="Sweep"
            hint={`Which load setting changes from row to row. Everything else comes from chat's Run Settings. ${SWEEP_BLURB[config.sweep]}`}
          >
            <Select
              value={config.sweep}
              onValueChange={(v) => choosePreset(v as SweepKind, maxContext)}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SWEEP_KINDS.map((k) => (
                  <SelectItem key={k} value={k}>
                    {SWEEP_TITLE[k]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        ) : (
          <Field
            label="Model"
            hint="Tries every speculative mode with chat's Run Settings and picks the fastest. A simpler setting within 3% of it wins."
          >
            <Select
              value={config.tuneModel ?? LOADED}
              onValueChange={(v) =>
                setConfig({ tuneModel: v === LOADED ? null : v })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={LOADED}>
                  {status?.active_model
                    ? `${modelShort(status.active_model)} (loaded)`
                    : "Whatever chat has loaded"}
                </SelectItem>
                {ggufs.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        )}

        <Field
          label="Compare"
          hint="Each row is one model load. The star marks the baseline the others are measured against."
          aside={
            <button
              type="button"
              className="text-ui-11 tabular-nums text-muted-foreground hover:text-foreground"
              onClick={() => {
                for (const v of config.variants)
                  if (allOn || disabled.includes(v.label))
                    toggleVariant(v.label);
              }}
            >
              {on} of {config.variants.length} · {allOn ? "clear" : "all"}
            </button>
          }
        >
          <ul className="-mx-2 flex flex-col">
            {groups.map((g) => {
              const rows = g.rows.map((v) => (
                <VariantRow
                  key={v.label}
                  label={shortLabel(v.label, g.family)}
                  title={v.label}
                  color={colors[g.family]}
                  active={!disabled.includes(v.label)}
                  isBase={config.baseline === v.label}
                  onToggle={() => toggleVariant(v.label)}
                  onBaseline={() =>
                    setConfig({
                      baseline: config.baseline === v.label ? null : v.label,
                    })
                  }
                />
              ));
              if (g.rows.length === 1) return rows;
              const activeHere = g.rows.filter(
                (v) => !disabled.includes(v.label),
              ).length;
              const expanded = opened.includes(g.family);
              return (
                <li key={g.family} className="flex flex-col">
                  <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-muted/60">
                    <button
                      type="button"
                      onClick={() =>
                        setOpened((o) =>
                          o.includes(g.family)
                            ? o.filter((f) => f !== g.family)
                            : [...o, g.family],
                        )
                      }
                      aria-expanded={expanded}
                      aria-label={`${expanded ? "Collapse" : "Expand"} ${FAMILY_LABEL[g.family]}`}
                      className="grid size-5 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <HugeiconsIcon
                        icon={ArrowDown01Icon}
                        strokeWidth={1.75}
                        className={cn(
                          "size-3.5 transition-transform",
                          !expanded && "-rotate-90",
                        )}
                      />
                    </button>
                    <Dot color={colors[g.family]} filled={activeHere > 0} />
                    <button
                      type="button"
                      onClick={() => {
                        const allHere = activeHere === g.rows.length;
                        for (const v of g.rows)
                          if (allHere || disabled.includes(v.label))
                            toggleVariant(v.label);
                      }}
                      className={cn(
                        "min-w-0 flex-1 truncate text-left text-ui-12p5",
                        activeHere > 0
                          ? "text-foreground"
                          : "text-muted-foreground",
                      )}
                      title={`Turn ${activeHere === g.rows.length ? "off" : "on"} every ${FAMILY_LABEL[g.family]} row`}
                    >
                      {FAMILY_LABEL[g.family]}
                    </button>
                    <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
                      {activeHere} of {g.rows.length}
                    </span>
                  </div>
                  {expanded && <ul className="flex flex-col pl-5">{rows}</ul>}
                </li>
              );
            })}
          </ul>
        </Field>

        <Field
          label="Prompt"
          hint="Each measured run sends one of these. Rotating them keeps ngram drafting honest."
        >
          <Select
            value={config.promptSet}
            onValueChange={(promptSet) => setConfig({ promptSet })}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {prompts.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {config.promptSet === CUSTOM_PROMPT_ID ? (
            <PromptListEditor
              value={config.customPrompt}
              onChange={(customPrompt) => setConfig({ customPrompt })}
            />
          ) : (
            <p className="text-ui-11p5 text-muted-foreground">
              {prompts.find((p) => p.id === config.promptSet)?.hint}
            </p>
          )}
        </Field>

        <Field
          label="Runs"
          hint="Generations per row. Warm-ups run first and are left out of the numbers."
          aside={
            <span className="text-ui-11 tabular-nums text-muted-foreground">
              {on * perRow} total
            </span>
          }
        >
          <div className="grid grid-cols-3 gap-2">
            {(
              [
                ["Measured", "repetitions", 1, 50, 1],
                ["Warm-up", "warmup", 0, 10, 1],
                ["Max tokens", "maxTokens", 16, 8192, 16],
              ] as const
            ).map(([label, key, min, max, step]) => (
              <label key={key} className="flex flex-col gap-1">
                <span className="text-ui-11 text-muted-foreground">
                  {label}
                </span>
                <Input
                  type="number"
                  value={config[key]}
                  min={min}
                  max={max}
                  step={step}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    if (Number.isFinite(n))
                      setConfig({
                        [key]: Math.min(max, Math.max(min, Math.round(n))),
                      });
                  }}
                  className="text-center font-mono tabular-nums"
                />
              </label>
            ))}
          </div>
          <label className="flex items-center gap-2.5 text-ui-12p5 text-foreground">
            <Checkbox
              checked={config.rotatePrompts}
              onCheckedChange={(v) => setConfig({ rotatePrompts: v === true })}
            />
            Rotate prompts
          </label>
          <label className="flex items-center gap-2.5 text-ui-12p5 text-foreground">
            <Checkbox
              checked={config.restoreAfter}
              onCheckedChange={(v) => setConfig({ restoreAfter: v === true })}
            />
            Restore chat's settings after
          </label>
        </Field>
      </fieldset>
    </aside>
  );
}
