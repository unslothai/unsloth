// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { InfoHint } from "@/components/ui/info-hint";
import { NumericValueInput } from "@/components/ui/numeric-value-input";
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
  KV_CACHE_DTYPES,
  MAX_CHAT_TEMPLATE_BYTES,
  MAX_CHAT_TEMPLATE_LENGTH,
  MTP_SPECULATIVE_TYPES,
  type PerModelConfig,
  SPECULATIVE_TYPES,
  chatTemplateByteLength,
  clampChatTemplateToByteLimit,
  fetchModelDefaults,
  ggufVariantsMatch,
  hasPerModelConfig,
  isAbortError,
  isChatTemplateWithinLimit,
  isDefaultConfig,
  modelIdsMatch,
  modelStorageKey,
  resolveInitialConfig,
  useChatRuntimeStore,
  useModelDefaults,
  validateChatTemplate,
} from "@/features/chat";
import {
  ArrowDown01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Delete02Icon,
  PlayIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import { clampLines } from "./chat-template-preview";
import {
  DeleteFineTuneDialog,
  type DeleteFineTuneTarget,
} from "./delete-fine-tune-dialog";
import type { DeletedModelRef, ModelPickTarget } from "./types";

const DEFAULT_CONTEXT_FALLBACK = 131072;
const MIN_CONTEXT_LENGTH = 1024;

interface ConfigState {
  key: string;
  config: PerModelConfig;
  remember: boolean;
  advancedOpen: boolean;
}

interface TemplateFetchRef {
  controller: AbortController;
  requestKey: string;
}

const SPECULATIVE_LABELS: Record<(typeof SPECULATIVE_TYPES)[number], string> = {
  auto: "Auto",
  mtp: "MTP",
  ngram: "Ngram",
  "ngram-simple": "Ngram Simple",
  "mtp+ngram": "MTP+Ngram",
  off: "Off",
};
const SPECULATIVE_OPTIONS = SPECULATIVE_TYPES.map((value) => ({
  value,
  label: SPECULATIVE_LABELS[value],
}));

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

export function ModelConfigPage({
  target,
  onBack,
  onCancel,
  onRun,
  onDeleted,
  deleteDisabled = false,
}: ModelConfigPageProps) {
  const deleteSource = pickerSourceToDeleteSource(target.meta.source);
  const canDeleteGgufExport =
    target.isGguf &&
    deleteSource === "exported" &&
    target.meta.exportType === "gguf" &&
    !!target.meta.ggufVariant;
  const canDelete =
    !deleteDisabled &&
    deleteSource !== null &&
    (!target.isGguf || canDeleteGgufExport);
  const [deletePending, setDeletePending] = useState<DeleteFineTuneTarget | null>(
    null,
  );
  const initial = useMemo(
    () => resolveInitialConfig(target.id, target.meta.ggufVariant),
    [target.id, target.meta.ggufVariant],
  );
  const initialIsDefault = isDefaultConfig(initial.config);
  const configKey = modelStorageKey(target.id, target.meta.ggufVariant);
  const initialConfigState = useMemo<ConfigState>(
    () => ({
      key: configKey,
      config: initial.config,
      remember: initial.remembered,
      advancedOpen: !initialIsDefault,
    }),
    [configKey, initial.config, initial.remembered, initialIsDefault],
  );
  const [configState, setConfigState] =
    useState<ConfigState>(() => initialConfigState);
  const activeConfigState =
    configState.key === configKey ? configState : initialConfigState;
  const { config, remember, advancedOpen } = activeConfigState;
  const rememberSettingsId = useId();
  const [contextEditingState, setContextEditingState] = useState<{
    key: string;
    value: boolean;
  }>({ key: configKey, value: false });
  const contextEditing =
    contextEditingState.key === configKey ? contextEditingState.value : false;
  const setContextEditing = useCallback((value: boolean) => {
    setContextEditingState({ key: configKey, value });
  }, [configKey]);
  const [templateEditorOpen, setTemplateEditorOpen] = useState(false);
  const [templateDraft, setTemplateDraft] = useState("");
  const [templateInitialDraft, setTemplateInitialDraft] = useState("");
  const [templateLoading, setTemplateLoading] = useState(false);
  const [templateValidating, setTemplateValidating] = useState(false);
  const templateDraftTouchedRef = useRef(false);
  const templateLimitToastShownRef = useRef(false);
  const templateFetchAbortRef = useRef<TemplateFetchRef | null>(null);

  useEffect(() => {
    setConfigState(initialConfigState);
    setContextEditingState({ key: configKey, value: false });
  }, [configKey, initialConfigState]);

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
    modelIdsMatch(loadedCheckpoint, target.id) &&
    ggufVariantsMatch(loadedVariant, target.meta.ggufVariant);
  const canUseVerifiedLocalDefaults =
    target.meta.source === "local" ||
    target.meta.source === "lora" ||
    target.meta.source === "exported" ||
    (target.meta.source === "hub" &&
      target.meta.preferLocalCache === true &&
      target.meta.isDownloaded === true &&
      target.meta.isPartial !== true);
  const defaultsFetchOptions = useMemo(
    () => ({
      preferLocalCache: canUseVerifiedLocalDefaults,
      localPath: canUseVerifiedLocalDefaults
        ? (target.meta.localPath ?? null)
        : null,
      modelFormat: target.meta.modelFormat ?? (target.isGguf ? "gguf" : null),
    }),
    [
      canUseVerifiedLocalDefaults,
      target.isGguf,
      target.meta.localPath,
      target.meta.modelFormat,
    ],
  );
  const defaultsFetchRequestKey = useMemo(
    () =>
      `${target.id}\0${defaultsFetchOptions.preferLocalCache ? "local" : "remote"}\0${defaultsFetchOptions.localPath ?? ""}\0${defaultsFetchOptions.modelFormat ?? ""}`,
    [
      target.id,
      defaultsFetchOptions.preferLocalCache,
      defaultsFetchOptions.localPath,
      defaultsFetchOptions.modelFormat,
    ],
  );

  const {
    maxContext: fetchedMaxContext,
    chatTemplate: fetchedChatTemplate,
    setChatTemplate: setFetchedChatTemplate,
  } = useModelDefaults(target.id, {
    skip: sameAsLoaded || (target.isGguf && !canUseVerifiedLocalDefaults),
    ...defaultsFetchOptions,
  });

  const defaultTemplate = sameAsLoaded
    ? defaultTemplateFromStore
    : fetchedChatTemplate;

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const abortTemplateFetch = useCallback(() => {
    templateFetchAbortRef.current?.controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      abortTemplateFetch();
    };
  }, [abortTemplateFetch]);

  useEffect(() => {
    const active = templateFetchAbortRef.current;
    if (active && active.requestKey !== defaultsFetchRequestKey) {
      active.controller.abort();
    }
  }, [defaultsFetchRequestKey]);

  const isGguf = target.isGguf;
  const nativeMaxContext = sameAsLoaded
    ? (ggufNativeContextFromStore ?? ggufContextFromStore ?? null)
    : fetchedMaxContext;
  const ctxMax = nativeMaxContext ?? DEFAULT_CONTEXT_FALLBACK;
  const ctxValue = config.customContextLength ?? nativeMaxContext ?? ctxMax;
  const isCustomContext = config.customContextLength !== null;
  const showContextEditor = isCustomContext || contextEditing;
  const remembered = hasPerModelConfig(target.id, target.meta.ggufVariant);
  const isDefault = isDefaultConfig(config);

  function update<K extends keyof PerModelConfig>(
    key: K,
    value: PerModelConfig[K],
  ) {
    setConfigState((prev) => {
      const base = prev.key === configKey ? prev : initialConfigState;
      return { ...base, config: { ...base.config, [key]: value } };
    });
  }

  function resetToDefaults() {
    setContextEditing(false);
    setConfigState((prev) => {
      const base = prev.key === configKey ? prev : initialConfigState;
      return { ...base, config: { ...DEFAULT_PER_MODEL_CONFIG } };
    });
  }

  async function openTemplateEditor() {
    abortTemplateFetch();
    setTemplateLoading(false);
    const initialSync =
      config.chatTemplateOverride ?? defaultTemplate ?? "";
    templateDraftTouchedRef.current = false;
    setTemplateDraft(initialSync);
    setTemplateInitialDraft(initialSync);
    setTemplateEditorOpen(true);

    if (
      config.chatTemplateOverride == null &&
      defaultTemplate == null &&
      !sameAsLoaded
    ) {
      const controller = new AbortController();
      templateFetchAbortRef.current = {
        controller,
        requestKey: defaultsFetchRequestKey,
      };
      setTemplateLoading(true);
      try {
        const defaults = await fetchModelDefaults(
          target.id,
          controller.signal,
          defaultsFetchOptions,
        );
        if (!mountedRef.current || controller.signal.aborted) return;
        if (defaults.chatTemplate != null) {
          setFetchedChatTemplate(defaults.chatTemplate);
          if (!templateDraftTouchedRef.current) {
            setTemplateDraft(defaults.chatTemplate);
            setTemplateInitialDraft(defaults.chatTemplate);
          }
        }
      } catch (err) {
        if (!mountedRef.current || controller.signal.aborted) return;
        if (!isAbortError(err)) {
          toast.error("Couldn't load chat template", {
            description: err instanceof Error ? err.message : undefined,
          });
        }
      } finally {
        if (templateFetchAbortRef.current?.controller === controller) {
          templateFetchAbortRef.current = null;
          if (mountedRef.current) setTemplateLoading(false);
        }
      }
    }
  }

  function handleTemplateEditorOpenChange(open: boolean) {
    setTemplateEditorOpen(open);
    if (open) {
      templateLimitToastShownRef.current = false;
    }
    if (!open) {
      abortTemplateFetch();
      setTemplateLoading(false);
    }
  }

  async function saveTemplateEditor() {
    const trimmed = templateDraft.trim();
    const next =
      trimmed.length === 0 ||
      (defaultTemplate != null && templateDraft === defaultTemplate)
        ? null
        : templateDraft;
    if (next != null && !isChatTemplateWithinLimit(next)) {
      toast.error("Chat template is too large", {
        description: `Keep it under ${MAX_CHAT_TEMPLATE_BYTES.toLocaleString()} bytes.`,
      });
      return;
    }
    if (next != null) {
      setTemplateValidating(true);
      try {
        const result = await validateChatTemplate(next);
        if (!mountedRef.current) return;
        if (!result.valid) {
          toast.error("Invalid chat template", {
            description:
              result.error ?? "Check the Jinja syntax and try again.",
          });
          return;
        }
      } catch (error) {
        if (!mountedRef.current) return;
        toast.error("Couldn't validate chat template", {
          description: error instanceof Error ? error.message : undefined,
        });
        return;
      } finally {
        if (mountedRef.current) setTemplateValidating(false);
      }
    }
    update("chatTemplateOverride", next);
    handleTemplateEditorOpenChange(false);
  }

  function resetTemplateDraftToDefault() {
    templateDraftTouchedRef.current = true;
    setTemplateDraft(defaultTemplate ?? "");
  }

  function updateTemplateDraft(value: string) {
    templateDraftTouchedRef.current = true;
    const next = clampChatTemplateToByteLimit(value);
    if (next !== value && !templateLimitToastShownRef.current) {
      templateLimitToastShownRef.current = true;
      toast.warning("Chat template limit reached", {
        description: `Keep it under ${MAX_CHAT_TEMPLATE_BYTES.toLocaleString()} bytes.`,
      });
    } else if (next === value) {
      templateLimitToastShownRef.current = false;
    }
    setTemplateDraft(next);
  }

  const templateDraftDirty = templateDraft !== templateInitialDraft;
  const templateDraftBytes = chatTemplateByteLength(templateDraft);
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
    const specLabel =
      SPECULATIVE_OPTIONS.find((o) => o.value === config.speculativeType)
        ?.label ?? config.speculativeType;
    summaryParts.push(`Speculative ${specLabel}`);
  }
  if (config.specDraftNMax != null) {
    summaryParts.push(`Draft ${config.specDraftNMax}`);
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
                    exportType: target.meta.exportType,
                    ggufVariant: target.meta.ggufVariant,
                  })
                }
                aria-label={
                  canDeleteGgufExport
                    ? "Delete GGUF quantization"
                    : "Delete fine-tuned model"
                }
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
          onClick={() =>
            setConfigState((prev) => {
              const base = prev.key === configKey ? prev : initialConfigState;
              return { ...base, advancedOpen: !base.advancedOpen };
            })
          }
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
                {showContextEditor ? (
                  <NumericValueInput
                    value={ctxValue}
                    min={MIN_CONTEXT_LENGTH}
                    max={Math.max(MIN_CONTEXT_LENGTH, ctxMax)}
                    step={1}
                    onChange={(v) => update("customContextLength", v)}
                    displayValue={ctxValue.toLocaleString()}
                    ariaLabel="Context Length"
                    className="min-w-0 flex-1 tabular-nums w-20 !flex-none text-right"
                  />
                ) : (
                  <button
                    type="button"
                    onClick={() => setContextEditing(true)}
                    className="hub-action-btn h-7 shrink-0 px-2.5 text-[11.5px] tabular-nums"
                  >
                    {nativeMaxContext != null
                      ? `Default · ${nativeMaxContext.toLocaleString()}`
                      : "Default"}
                  </button>
                )}
              </div>
              {showContextEditor && (
                <>
                  <Slider
                    min={MIN_CONTEXT_LENGTH}
                    max={Math.max(MIN_CONTEXT_LENGTH, ctxMax)}
                    step={1024}
                    value={[Math.min(ctxValue, Math.max(MIN_CONTEXT_LENGTH, ctxMax))]}
                    onValueChange={([v]) =>
                      update("customContextLength", Math.round(v))
                    }
                    className="panel-slider"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      update("customContextLength", null);
                      setContextEditing(false);
                    }}
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
                  quality. f16/bf16 are full precision; q8_0/q5_1/q4_1/q4_0 are
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
                  {KV_CACHE_DTYPES.map((dtype) => (
                    <SelectItem key={dtype} value={dtype}>
                      {dtype}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldRow>

            <FieldRow
              label="Speculative Decoding"
              info={
                <>
                  Faster generation with no accuracy hit. Auto picks MTP or
                  ngram based on the model and platform. Pick a mode to force
                  it on both GPU and CPU.
                </>
              }
            >
              <Select
                value={config.speculativeType ?? "auto"}
                onValueChange={(v) => {
                  update("speculativeType", v);
                  if (!MTP_SPECULATIVE_TYPES.has(v)) update("specDraftNMax", null);
                }}
              >
                <SelectTrigger
                  animateRadius={false}
                  icon={ArrowDown01Icon}
                  iconClassName="size-3.5"
                  className="grid h-7 w-[120px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[10px] border-transparent bg-black/[0.04] px-2 py-0 text-[13px]! font-medium hover:bg-black/[0.06] focus-visible:border-transparent focus-visible:ring-0 dark:bg-white/[0.05] dark:hover:bg-white/[0.07] [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="menu-soft-surface rounded-lg border-0 ring-0">
                  {SPECULATIVE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldRow>

            {MTP_SPECULATIVE_TYPES.has(config.speculativeType ?? "") && (
              <FieldRow
                label="Draft Tokens"
                info={
                  <>
                    Max MTP draft tokens per step (--spec-draft-n-max). Lower =
                    less wasted draft decode. Higher = bigger speedup when
                    acceptance stays high. Default: 2 on GPU, 3 on CPU/Mac.
                  </>
                }
              >
                <input
                  type="number"
                  min={1}
                  max={16}
                  step={1}
                  value={config.specDraftNMax ?? ""}
                  placeholder="auto"
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") {
                      update("specDraftNMax", null);
                      return;
                    }
                    const parsed = Number.parseInt(raw, 10);
                    if (Number.isFinite(parsed)) {
                      update("specDraftNMax", Math.max(1, Math.min(16, parsed)));
                    }
                  }}
                  aria-label="Speculative decoding draft tokens"
                  className="h-7 w-[72px] rounded-[10px] border-transparent bg-black/[0.04] px-2 py-0 text-[13px] font-medium text-foreground outline-none hover:bg-black/[0.06] focus-visible:ring-0 dark:bg-white/[0.05] dark:hover:bg-white/[0.07]"
                />
              </FieldRow>
            )}
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

        <label
          htmlFor={rememberSettingsId}
          className="flex cursor-pointer items-start gap-2.5"
        >
          <Checkbox
            id={rememberSettingsId}
            checked={remember}
            onCheckedChange={(v) =>
              setConfigState((prev) => {
                const base = prev.key === configKey ? prev : initialConfigState;
                return { ...base, remember: v === true };
              })
            }
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
        onDeleted={(deleted, deletedRunIds) => {
          setDeletePending(null);
          onDeleted?.({
            id: deleted.id,
            ...(deleted.ggufVariant !== undefined
              ? { ggufVariant: deleted.ggufVariant }
              : {}),
            ...(deletedRunIds.length > 0 ? { deletedRunIds } : {}),
          });
          onBack();
        }}
      />

      <ChatTemplateEditorDialog
        open={templateEditorOpen}
        onOpenChange={handleTemplateEditorOpenChange}
        hasOverride={config.chatTemplateOverride != null}
        defaultTemplate={defaultTemplate}
        draft={templateDraft}
        draftBytes={templateDraftBytes}
        draftByteLimit={MAX_CHAT_TEMPLATE_BYTES}
        maxLength={MAX_CHAT_TEMPLATE_LENGTH}
        onDraftChange={updateTemplateDraft}
        loading={templateLoading}
        draftDirty={templateDraftDirty}
        draftIsDefault={draftIsDefault}
        onResetToDefault={resetTemplateDraftToDefault}
        onSave={saveTemplateEditor}
        saving={templateValidating}
      />
    </div>
  );
}
