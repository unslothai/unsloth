// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
  deletePerModelConfigsForModel,
  hasPerModelConfig,
  isDefaultConfig,
  resolveInitialConfig,
} from "@/features/chat/model-config/per-model-config";
import { fetchModelDefaults } from "@/features/chat/model-config/model-defaults-fetch";
import {
  isAbortError,
  useModelDefaults,
} from "@/features/chat/model-config/use-model-defaults";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Delete02Icon,
  InformationCircleIcon,
  PlayIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import {
  DeleteFineTuneDialog,
  type DeleteFineTuneTarget,
} from "./delete-fine-tune-dialog";
import type { DeletedModelRef, ModelPickTarget } from "./types";

const DEFAULT_CONTEXT_FALLBACK = 131072;

interface ModelConfigPageProps {
  target: ModelPickTarget;
  onBack: () => void;
  onCancel: () => void;
  onRun: (config: PerModelConfig, remember: boolean) => void;
  onDeleted?: (deletedModel: DeletedModelRef) => void;
  deleteDisabled?: boolean;
}

function pickerSourceToDeleteSource(
  source: ModelPickTarget["meta"]["source"],
): "training" | "exported" | null {
  if (source === "lora") return "training";
  if (source === "exported") return "exported";
  return null;
}

function snapToStep(
  value: number,
  step: number,
  min?: number,
  max?: number,
): number {
  const lo = min ?? Number.NEGATIVE_INFINITY;
  const hi = max ?? Number.POSITIVE_INFINITY;
  const clamped = Math.min(Math.max(value, lo), hi);
  const stepStr = String(step);
  const decimals = stepStr.includes(".") ? stepStr.split(".")[1].length : 0;
  const base = Number.isFinite(lo) ? lo : 0;
  const snapped = base + Math.round((clamped - base) / step) * step;
  return Number(Math.min(Math.max(snapped, lo), hi).toFixed(decimals));
}

function NumericValueInput({
  value,
  min,
  max,
  step,
  onChange,
  ariaLabel,
  className,
}: {
  value: number;
  min?: number;
  max?: number;
  step: number;
  onChange: (v: number) => void;
  ariaLabel?: string;
  className?: string;
}) {
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState("");
  const cancelBlurCommitRef = useRef(false);

  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return;
    const final = snapToStep(parsed, step, min, max);
    if (final !== value) onChange(final);
  };

  return (
    <input
      type="text"
      inputMode="decimal"
      value={focused ? draft : value.toLocaleString()}
      aria-label={ariaLabel}
      onFocus={(e) => {
        cancelBlurCommitRef.current = false;
        setDraft(String(value));
        setFocused(true);
        const target = e.currentTarget;
        requestAnimationFrame(() => target.select());
      }}
      onBlur={() => {
        if (cancelBlurCommitRef.current) {
          cancelBlurCommitRef.current = false;
        } else {
          commit(draft);
        }
        setFocused(false);
      }}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          cancelBlurCommitRef.current = true;
          setDraft(String(value));
          e.currentTarget.blur();
        }
      }}
      className={cn(
        "panel-number-input min-w-0 flex-1 tabular-nums",
        className,
      )}
    />
  );
}

function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="More info"
          className="inline-flex size-4 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:text-foreground"
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="left"
        sideOffset={8}
        className="tooltip-compact max-w-64"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <span className="text-[12px] font-medium tracking-tight text-foreground">
      {children}
    </span>
  );
}

function FieldRow({
  label,
  info,
  children,
}: {
  label: string;
  info?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="text-[12px] font-medium tracking-tight text-foreground">
          {label}
        </span>
        {info && <InfoHint>{info}</InfoHint>}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">{children}</div>
    </div>
  );
}

function modelLeaf(id: string): string {
  const slash = id.lastIndexOf("/");
  if (slash > 0 && slash < id.length - 1) return id.slice(slash + 1);
  return id;
}

function clampLines(text: string, maxLines = 3, maxCharsPerLine = 88): string {
  if (!text) return "";
  const out: string[] = [];
  let truncated = false;
  for (const raw of text.split("\n")) {
    if (out.length >= maxLines) {
      truncated = true;
      break;
    }
    if (raw.length <= maxCharsPerLine) {
      out.push(raw);
      continue;
    }
    out.push(`${raw.slice(0, maxCharsPerLine - 1)}…`);
    truncated = true;
    break;
  }
  if (!truncated) return out.join("\n");
  return `${out.join("\n")}\n…`;
}

export function ModelConfigPage({
  target,
  onBack,
  onCancel,
  onRun,
  onDeleted,
  deleteDisabled = false,
}: ModelConfigPageProps) {
  const deleteSource = pickerSourceToDeleteSource(target.meta.source);
  const canDelete = !deleteDisabled && deleteSource !== null && !target.isGguf;
  const [deletePending, setDeletePending] = useState<DeleteFineTuneTarget | null>(
    null,
  );
  const initial = useMemo(
    () => resolveInitialConfig(target.id, target.meta.ggufVariant),
    [target.id, target.meta.ggufVariant],
  );
  const initialIsDefault = isDefaultConfig(initial.config);
  const [config, setConfig] = useState<PerModelConfig>(initial.config);
  const [remember, setRemember] = useState(initial.remembered);
  const [advancedOpen, setAdvancedOpen] = useState(!initialIsDefault);
  const [templateEditorOpen, setTemplateEditorOpen] = useState(false);
  const [templateDraft, setTemplateDraft] = useState("");
  const [templateInitialDraft, setTemplateInitialDraft] = useState("");
  const [templateLoading, setTemplateLoading] = useState(false);

  const defaultTemplateFromStore = useChatRuntimeStore(
    (s) => s.defaultChatTemplate,
  );
  const loadedCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const loadedVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const ggufNativeContextFromStore = useChatRuntimeStore(
    (s) => s.ggufNativeContextLength,
  );
  const ggufContextFromStore = useChatRuntimeStore(
    (s) => s.ggufContextLength,
  );
  const sameAsLoaded =
    loadedCheckpoint === target.id &&
    (loadedVariant ?? null) === (target.meta.ggufVariant ?? null);

  const {
    maxContext: fetchedMaxContext,
    chatTemplate: fetchedChatTemplate,
    setChatTemplate: setFetchedChatTemplate,
  } = useModelDefaults(target.id, { skip: sameAsLoaded });

  const defaultTemplate = sameAsLoaded
    ? defaultTemplateFromStore
    : fetchedChatTemplate;

  useEffect(() => {
    setConfig(initial.config);
    setRemember(initial.remembered);
    setAdvancedOpen(!initialIsDefault);
  }, [initial, initialIsDefault]);

  const isGguf = target.isGguf;
  const speculativeOn =
    config.speculativeType != null && config.speculativeType !== "off";
  const nativeMaxContext = sameAsLoaded
    ? (ggufNativeContextFromStore ?? ggufContextFromStore ?? null)
    : fetchedMaxContext;
  const ctxMax = nativeMaxContext ?? DEFAULT_CONTEXT_FALLBACK;
  const ctxValue = config.customContextLength ?? nativeMaxContext ?? ctxMax;
  const isCustomContext = config.customContextLength !== null;
  const ctxIsKnown = nativeMaxContext != null;
  const remembered = hasPerModelConfig(target.id, target.meta.ggufVariant);
  const isDefault = isDefaultConfig(config);

  function update<K extends keyof PerModelConfig>(
    key: K,
    value: PerModelConfig[K],
  ) {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  function resetToDefaults() {
    setConfig({ ...DEFAULT_PER_MODEL_CONFIG });
  }

  async function openTemplateEditor() {
    const initialSync =
      config.chatTemplateOverride ?? defaultTemplate ?? "";
    setTemplateDraft(initialSync);
    setTemplateInitialDraft(initialSync);
    setTemplateEditorOpen(true);

    if (
      config.chatTemplateOverride == null &&
      defaultTemplate == null &&
      !sameAsLoaded
    ) {
      setTemplateLoading(true);
      try {
        const defaults = await fetchModelDefaults(target.id);
        if (defaults.chatTemplate != null) {
          setFetchedChatTemplate(defaults.chatTemplate);
          setTemplateDraft((current) =>
            current.length === 0 ? defaults.chatTemplate ?? "" : current,
          );
          setTemplateInitialDraft((current) =>
            current.length === 0 ? defaults.chatTemplate ?? "" : current,
          );
        }
      } catch (err) {
        if (!isAbortError(err)) {
          toast.error("Couldn't load chat template", {
            description: err instanceof Error ? err.message : undefined,
          });
        }
      } finally {
        setTemplateLoading(false);
      }
    }
  }

  function saveTemplateEditor() {
    const trimmed = templateDraft.trim();
    const next =
      trimmed.length === 0 ||
      (defaultTemplate != null && templateDraft === defaultTemplate)
        ? null
        : templateDraft;
    update("chatTemplateOverride", next);
    setTemplateEditorOpen(false);
  }

  function resetTemplateDraftToDefault() {
    setTemplateDraft(defaultTemplate ?? "");
  }

  const templateDraftDirty = templateDraft !== templateInitialDraft;
  const draftIsDefault =
    defaultTemplate != null && templateDraft === defaultTemplate;
  const leaf = modelLeaf(target.displayName);

  const summaryParts: string[] = [];
  if (isCustomContext) {
    summaryParts.push(`Context ${ctxValue.toLocaleString()}`);
  }
  if (
    (config.kvCacheDtype ?? null) !==
    (DEFAULT_PER_MODEL_CONFIG.kvCacheDtype ?? null)
  ) {
    summaryParts.push(`KV ${config.kvCacheDtype ?? "f16"}`);
  }
  if (
    config.speculativeType !== DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.speculativeType !== null
  ) {
    summaryParts.push(
      config.speculativeType === "off"
        ? "Speculative off"
        : "Speculative on",
    );
  }
  if (config.chatTemplateOverride != null) {
    summaryParts.push("Custom chat template");
  }
  if (
    Boolean(config.trustRemoteCode) !==
    Boolean(DEFAULT_PER_MODEL_CONFIG.trustRemoteCode)
  ) {
    summaryParts.push("Custom code on");
  }
  const customizationSummary =
    summaryParts.length > 0 ? summaryParts.join(" · ") : null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-1.5 border-b border-border/40 pb-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={onBack}
              aria-label="Back to model list"
              className="flex size-7 shrink-0 items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-black/[0.04] hover:text-foreground dark:hover:bg-white/[0.05]"
            >
              <HugeiconsIcon
                icon={ArrowLeft01Icon}
                strokeWidth={2}
                className="size-4"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="tooltip-compact">
            Back to models
          </TooltipContent>
        </Tooltip>
        <div className="flex min-w-0 flex-1 flex-col leading-none">
          <span className="truncate text-[13px] font-medium tracking-tight text-foreground">
            {leaf}
          </span>
        </div>
        {target.meta.ggufVariant && (
          <span className="shrink-0 rounded-[7px] border border-format-gguf/40 bg-transparent px-1.5 py-0.5 font-mono text-[10.5px] leading-none text-format-gguf">
            {target.meta.ggufVariant}
          </span>
        )}
        {canDelete && deleteSource && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() =>
                  setDeletePending({
                    id: target.id,
                    displayName: target.displayName,
                    source: deleteSource,
                  })
                }
                aria-label="Delete fine-tuned model"
                className="flex size-7 shrink-0 items-center justify-center rounded-[8px] text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="tooltip-compact">
              Delete fine-tuned model
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      <div className="aui-thread-viewport flex max-h-[min(75vh,36rem)] min-h-0 flex-1 flex-col overflow-y-auto px-0.5 pt-3">
        <div className="flex items-start gap-2.5 rounded-[10px] border border-border/40 bg-foreground/[0.02] px-3 py-2 dark:bg-white/[0.02]">
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="text-[12px] font-medium tracking-tight leading-snug text-foreground">
              {isDefault
                ? "Running with model defaults"
                : "Running with custom settings"}
            </span>
            <span className="text-[11px] leading-snug tracking-tight text-muted-foreground/85">
              {isDefault
                ? "Optimal runtime, context and chat template applied automatically."
                : customizationSummary ?? "Custom configuration active."}
            </span>
          </div>
          {!isDefault && (
            <button
              type="button"
              onClick={resetToDefaults}
              className="hub-action-btn h-6 shrink-0 px-2 text-[11px]"
            >
              Reset
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={advancedOpen}
          className="mt-1.5 flex h-8 w-full shrink-0 items-center justify-between gap-2 rounded-[8px] bg-transparent px-2 text-left transition-colors hover:bg-foreground/[0.04] dark:hover:bg-white/[0.05]"
        >
          <span className="flex items-center gap-1.5">
            <HugeiconsIcon
              icon={Settings02Icon}
              strokeWidth={1.6}
              className="size-3.5 text-muted-foreground"
            />
            <span className="field-label">Advanced options</span>
          </span>
          <HugeiconsIcon
            icon={advancedOpen ? ArrowDown01Icon : ArrowRight01Icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0 text-muted-foreground"
          />
        </button>

        {advancedOpen && (
          <div className="flex min-w-0 flex-col gap-3 border-t border-border/40 pt-2.5">
        {isGguf && (
          <section className="flex flex-col gap-3">
            <div className="slider-field flex min-w-0 flex-col gap-2">
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="text-[12px] font-medium tracking-tight text-foreground">
                    Context Length
                  </span>
                  <InfoHint>
                    Maximum tokens the model can process. Larger contexts use
                    more VRAM.
                  </InfoHint>
                </div>
                {isCustomContext ? (
                  <NumericValueInput
                    value={ctxValue}
                    min={128}
                    max={ctxMax}
                    step={1}
                    onChange={(v) => update("customContextLength", v)}
                    ariaLabel="Context Length"
                    className="w-20 !flex-none text-right"
                  />
                ) : (
                  <button
                    type="button"
                    onClick={() =>
                      update(
                        "customContextLength",
                        nativeMaxContext ?? ctxMax,
                      )
                    }
                    className="hub-action-btn h-7 shrink-0 px-2.5 text-[11.5px] tabular-nums"
                  >
                    {ctxIsKnown
                      ? `Default · ${nativeMaxContext!.toLocaleString()}`
                      : "Default"}
                  </button>
                )}
              </div>
              {isCustomContext && (
                <>
                  <Slider
                    min={1024}
                    max={Math.max(1024, ctxMax)}
                    step={1024}
                    value={[Math.min(ctxValue, Math.max(1024, ctxMax))]}
                    onValueChange={([v]) =>
                      update("customContextLength", Math.round(v))
                    }
                    className="panel-slider"
                  />
                  <button
                    type="button"
                    onClick={() => update("customContextLength", null)}
                    className="self-start text-[11px] font-medium tracking-tight text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline"
                  >
                    Use model default
                  </button>
                </>
              )}
            </div>

            <FieldRow
              label="KV Cache Dtype"
              info={
                <>
                  Lower KV cache precision to save VRAM at the cost of some
                  quality. f16/bf16 are full precision; q8_0/q5_1/q4_1 are
                  quantized.
                </>
              }
            >
              <Select
                value={config.kvCacheDtype ?? "f16"}
                onValueChange={(v) =>
                  update("kvCacheDtype", v === "f16" ? null : v)
                }
              >
                <SelectTrigger
                  animateRadius={false}
                  icon={ArrowDown01Icon}
                  iconClassName="size-3.5"
                  className="grid h-7 w-[72px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[10px] border-transparent bg-black/[0.04] px-2 py-0 text-[13px]! font-medium hover:bg-black/[0.06] focus-visible:border-transparent focus-visible:ring-0 dark:bg-white/[0.05] dark:hover:bg-white/[0.07] [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="menu-soft-surface rounded-lg border-0 ring-0">
                  <SelectItem value="f16">f16</SelectItem>
                  <SelectItem value="bf16">bf16</SelectItem>
                  <SelectItem value="q8_0">q8_0</SelectItem>
                  <SelectItem value="q5_1">q5_1</SelectItem>
                  <SelectItem value="q4_1">q4_1</SelectItem>
                </SelectContent>
              </Select>
            </FieldRow>

            <FieldRow
              label="Speculative Decoding"
              info={
                <>
                  N-gram speculation. Faster generation with negligible VRAM
                  overhead. Text-only models.
                </>
              }
            >
              <Switch
                className="panel-switch shrink-0"
                checked={speculativeOn}
                onCheckedChange={(checked) =>
                  update("speculativeType", checked ? "default" : "off")
                }
              />
            </FieldRow>
          </section>
        )}

        {!isGguf && target.supportsTrustRemoteCode && (
          <section className="flex flex-col gap-2">
            <FieldRow
              label="Enable custom code"
              info={
                <>
                  Run custom Python from the model repo (e.g. Nemotron). Only
                  enable for trusted sources.
                </>
              }
            >
              <Switch
                className="panel-switch shrink-0"
                checked={config.trustRemoteCode ?? false}
                onCheckedChange={(v) => update("trustRemoteCode", v)}
              />
            </FieldRow>
          </section>
        )}

        <section className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-3">
            <SectionLabel>Chat template</SectionLabel>
            <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-muted-foreground/60">
              {config.chatTemplateOverride ? "Custom" : "Default"}
            </span>
          </div>

          <button
            type="button"
            onClick={() => {
              void openTemplateEditor();
            }}
            aria-label="Edit chat template"
            className="flex w-full cursor-pointer items-start gap-2 rounded-[10px] border border-border/40 bg-foreground/[0.025] px-2.5 py-2 text-left transition-colors hover:border-border/70 hover:bg-foreground/[0.04] dark:bg-white/[0.02] dark:hover:bg-white/[0.04]"
          >
            <span className="block min-w-0 flex-1">
              {config.chatTemplateOverride ? (
                <span className="block whitespace-pre-wrap break-words font-mono text-[11px] leading-snug text-foreground">
                  {clampLines(config.chatTemplateOverride)}
                </span>
              ) : defaultTemplate ? (
                <span className="block whitespace-pre-wrap break-words font-mono text-[11px] leading-snug text-foreground/75">
                  {clampLines(defaultTemplate)}
                </span>
              ) : (
                <span className="block text-[11.5px] leading-snug text-muted-foreground">
                  Click to add a custom chat template
                </span>
              )}
            </span>
          </button>
          {config.chatTemplateOverride != null && (
            <button
              type="button"
              onClick={() => update("chatTemplateOverride", null)}
              className="self-start text-[11px] font-medium tracking-tight text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline"
            >
              Reset to model default
            </button>
          )}
        </section>

        <label className="flex cursor-pointer items-start gap-2.5">
          <Checkbox
            checked={remember}
            onCheckedChange={(v) => setRemember(v === true)}
            className="mt-0.5 size-[18px] rounded-[5px] border-border bg-foreground/[0.06] dark:border-white/15 dark:bg-white/[0.08] [&_svg]:size-3!"
          />
          <div className="flex min-w-0 flex-1 flex-col leading-tight">
            <span className="text-[12px] font-medium tracking-tight text-foreground">
              Remember these settings
            </span>
            <span className="truncate text-[11px] tracking-tight text-muted-foreground/85">
              {remember
                ? `Save for ${leaf}${target.meta.ggufVariant ? ` · ${target.meta.ggufVariant}` : ""}`
                : remembered
                  ? "Previous save kept. This run uses your edits only."
                  : "Apply this run only."}
            </span>
          </div>
        </label>
          </div>
        )}
      </div>

      <div className="mt-2 flex shrink-0 items-center justify-end gap-2 border-t border-border/40 pt-2.5">
        <button
          type="button"
          onClick={onCancel}
          className="inline-flex h-9 items-center justify-center rounded-full px-4 text-[13px] font-medium tracking-tight text-muted-foreground transition-colors hover:bg-foreground/[0.04] hover:text-foreground dark:hover:bg-white/[0.05]"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={() => onRun(config, remember)}
          className="run-action-btn"
        >
          <HugeiconsIcon icon={PlayIcon} strokeWidth={2} />
          Run model
        </button>
      </div>

      <DeleteFineTuneDialog
        target={deletePending}
        onOpenChange={(open) => {
          if (!open) setDeletePending(null);
        }}
        onDeleted={(deleted) => {
          setDeletePending(null);
          deletePerModelConfigsForModel(deleted.id);
          onDeleted?.({ id: deleted.id });
          onBack();
        }}
      />

      <ChatTemplateEditorDialog
        open={templateEditorOpen}
        onOpenChange={setTemplateEditorOpen}
        hasOverride={config.chatTemplateOverride != null}
        defaultTemplate={defaultTemplate}
        draft={templateDraft}
        onDraftChange={setTemplateDraft}
        loading={templateLoading}
        draftDirty={templateDraftDirty}
        draftIsDefault={draftIsDefault}
        onResetToDefault={resetTemplateDraftToDefault}
        onSave={saveTemplateEditor}
      />
    </div>
  );
}
