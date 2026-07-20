// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { InfoHint } from "@/components/ui/info-hint";
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
  GPU_LAYERS_AUTO,
  fetchGgufStagedMetadata,
  readPersistedSpeculativeType,
  useChatRuntimeStore,
} from "@/features/chat";
import { useGpuDevices } from "@/hooks/use-gpu-info";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useId, useState } from "react";
import {
  useDefaultChatTemplate,
  useModelMaxPositionEmbeddings,
} from "../hooks/use-model-defaults";
import { perModelConfigsEqual } from "../model-config/apply-per-model-config";
import {
  CONTEXT_LENGTH_MIN,
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  KV_CACHE_DTYPES,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MAX_SEQ_LENGTH_STEP,
  MTP_SPECULATIVE_TYPES,
  type PerModelConfig,
  SPECULATIVE_TYPES,
  deletePerModelConfig,
  floorMaxSeqLength,
  isDefaultConfig,
  normalizeMaxSeqLength,
  resolveInitialConfig,
  savePerModelConfig,
} from "../model-config/per-model-config";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import type { ModelPickTarget } from "./model-selector/types";
import { NumericValueInput } from "./numeric-value-input";

const ROW_CLASS = "flex min-h-8 items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 truncate text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg";
const LABEL_CLASS_WRAP =
  "min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg";
const CONTROL_SURFACE =
  "rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1]";
const SELECT_TRIGGER_CLASS = `grid h-8 min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0`;
const NUMBER_INPUT_CLASS = `h-8 w-[92px] ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-right text-[13px] font-medium text-nav-fg outline-none focus-visible:ring-0`;

const KV_CACHE_DTYPE_DEFAULT = "f16";
const SPECULATIVE_TYPE_LABELS: Record<(typeof SPECULATIVE_TYPES)[number], string> =
  {
    auto: "Auto",
    mtp: "MTP",
    ngram: "Ngram",
    "mtp+ngram": "MTP+Ngram",
    off: "Off",
  };

function hasNonDefaultAdvanced(config: PerModelConfig): boolean {
  return (
    config.kvCacheDtype != null ||
    (config.speculativeType ?? "auto") !== "auto" ||
    config.specDraftNMax != null ||
    config.tensorParallel ||
    config.chatTemplateOverride != null ||
    (config.gpuMemoryMode ?? "auto") !== "auto" ||
    (config.gpuLayers != null && config.gpuLayers >= 0) ||
    (config.nCpuMoe ?? 0) > 0 ||
    config.selectedGpuIds != null
  );
}

function ChatTemplateSetting({
  config,
  onEditTemplate,
  readOnly = false,
}: {
  config: PerModelConfig;
  onEditTemplate: () => void;
  readOnly?: boolean;
}) {
  return (
    <div className={ROW_CLASS}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={LABEL_CLASS}>Chat Template</span>
        <InfoHint>
          {readOnly
            ? "Preview the model's chat template. Custom overrides apply to GGUF models for now."
            : "Override the model's chat template with custom Jinja. Applies when the model loads."}
        </InfoHint>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {readOnly ? null : (
          <span className="text-[12px] text-muted-foreground">
            {config.chatTemplateOverride ? "Custom" : "Default"}
          </span>
        )}
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className={`h-8 px-3 text-[13px] ${CONTROL_SURFACE}`}
          onClick={onEditTemplate}
        >
          {readOnly ? "View" : "Edit"}
        </Button>
      </div>
    </div>
  );
}

function MaxSeqLengthSetting({
  value,
  max,
  inputMax,
  onChange,
}: {
  value: number;
  max: number;
  inputMax: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-3">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>Max Seq Length</span>
          <InfoHint>
            Maximum context window size in tokens. Applies when the model loads.
          </InfoHint>
        </div>
        <NumericValueInput
          value={value}
          min={MAX_SEQ_LENGTH_MIN}
          max={inputMax}
          step={MAX_SEQ_LENGTH_STEP}
          onChange={onChange}
          ariaLabel="Max Seq Length"
          className={NUMBER_INPUT_CLASS}
          size={8}
        />
      </div>
      <Slider
        min={MAX_SEQ_LENGTH_MIN}
        max={max}
        step={MAX_SEQ_LENGTH_STEP}
        value={[value]}
        onValueChange={([next]) => onChange(next)}
        className="panel-slider"
        aria-label="Max Seq Length"
      />
    </div>
  );
}

function clampMaxSeqLength(value: number, max: number): number {
  const normalized = normalizeMaxSeqLength(value) ?? MAX_SEQ_LENGTH_MIN;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(max, normalized));
}

function AdvancedGpuSlider({
  label,
  value,
  min,
  max,
  onChange,
  displayValue,
  info,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  displayValue?: string;
  info?: ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>{label}</span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <NumericValueInput
          value={value}
          min={min}
          max={max}
          step={1}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          className={NUMBER_INPUT_CLASS}
          size={8}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={1}
        value={[value]}
        onValueChange={([next]) => onChange(next)}
        className="panel-slider"
        aria-label={label}
      />
    </div>
  );
}

// GPU Memory placement controls (mode / GPU Layers / MoE offload / GPU picker),
// re-homed here from the chat settings sheet onto the per-model config. GGUF
// only; the layer/MoE slider ceilings come from the GGUF header dims and the GPU
// picker from the live device list. The per-GPU split ratio (--tensor-split) is
// intentionally not persisted per model, so it is not exposed here.
function GpuMemorySettings({
  config,
  update,
  layerCount,
  moeLayerCount,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  layerCount: number | null;
  moeLayerCount: number | null;
}) {
  const gpuDevices = useGpuDevices();
  const mode = config.gpuMemoryMode ?? "auto";
  const isManual = mode === "manual";
  const gpuLayers = config.gpuLayers ?? GPU_LAYERS_AUTO;
  // Manual with the slider at Auto (leftmost): llama.cpp --fit owns the whole
  // layout, so the MoE-offload knob doesn't apply.
  const autoLayers = isManual && gpuLayers < 0;
  // Ceiling = model layer count + 1 (llama.cpp counts the output layer as one
  // more offloadable layer past the repeating blocks), else a safe fallback.
  const gpuLayersMax = layerCount != null ? layerCount + 1 : 256;
  const nCpuMoe = config.nCpuMoe ?? 0;
  const moeLayersMax = moeLayerCount ?? 0;
  const showMoeSlider = isManual && !autoLayers && moeLayersMax > 0;
  const selectedGpuIds = config.selectedGpuIds ?? null;
  const singleGpuInUse =
    (selectedGpuIds ?? gpuDevices.map((device) => device.index)).length <= 1;
  // Only meaningful on multi-GPU, and only when the reported indices are
  // physical (relative ordinals from a parent CUDA_VISIBLE_DEVICES mask can't be
  // mapped back to pin a device). null = use all (auto).
  const showGpuPicker =
    gpuDevices.length > 1 && gpuDevices.every((d) => d.physicalIndex);
  const isGpuChecked = (index: number) =>
    selectedGpuIds === null || selectedGpuIds.includes(index);
  const toggleGpu = (index: number) => {
    const all = gpuDevices.map((d) => d.index);
    const current = selectedGpuIds ?? all;
    const next = current.includes(index)
      ? current.filter((i) => i !== index)
      : [...current, index].sort((a, b) => a - b);
    if (next.length === 0) return; // keep at least one GPU selected
    update({ selectedGpuIds: next.length === all.length ? null : next });
  };
  return (
    <>
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>GPU Memory</span>
          <InfoHint>
            <div className="flex flex-col gap-1.5">
              <div>
                <span className="font-medium">Default:</span> Unsloth fits the
                model and context to your GPUs.
              </div>
              <div>
                <span className="font-medium">Manual:</span> set GPU Layers
                yourself. Leave it on Auto to let llama.cpp size the context and
                offload overflow (including MoE experts) to RAM.
              </div>
            </div>
          </InfoHint>
        </div>
        <Select
          value={mode}
          onValueChange={(v) =>
            // Returning to Default must clear the Manual-only knobs; otherwise a
            // remembered config keeps stale gpuLayers/nCpuMoe/GPU pick that a
            // later load re-applies whenever the standing GPU preference is
            // Manual, despite the page showing Default.
            update(
              v === "manual"
                ? { gpuMemoryMode: "manual" }
                : {
                    gpuMemoryMode: "auto",
                    gpuLayers: undefined,
                    nCpuMoe: undefined,
                    selectedGpuIds: undefined,
                  },
            )
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={`w-[124px] shrink-0 ${SELECT_TRIGGER_CLASS}`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value="auto">Default</SelectItem>
            <SelectItem value="manual">Manual</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {isManual && (
        <>
          <AdvancedGpuSlider
            label="GPU Layers"
            value={Math.max(GPU_LAYERS_AUTO, Math.min(gpuLayers, gpuLayersMax))}
            min={GPU_LAYERS_AUTO}
            max={gpuLayersMax}
            onChange={(v) => update({ gpuLayers: v })}
            displayValue={autoLayers ? "Auto" : undefined}
            info={
              <>
                Layers to keep on the GPU (--gpu-layers); the rest run on CPU.
                Auto lets llama.cpp size the split (and the context) to fit VRAM.
                At the maximum, the whole model is on the GPU.
              </>
            }
          />
          {showMoeSlider && (
            <AdvancedGpuSlider
              label="MoE Layers on CPU"
              value={Math.min(nCpuMoe, moeLayersMax)}
              min={0}
              max={moeLayersMax}
              onChange={(v) => update({ nCpuMoe: v })}
              info={
                <>
                  Keep the experts of this many MoE layers on the CPU
                  (--n-cpu-moe) to save VRAM. 0 = all experts on the GPU; at the
                  maximum, all are on the CPU.
                </>
              }
            />
          )}
        </>
      )}
      {showGpuPicker && (
        <div className="space-y-2">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>GPUs</span>
            <InfoHint>
              Which GPUs this model may use. Unchecked GPUs are hidden from
              llama.cpp (CUDA_VISIBLE_DEVICES, or HIP_VISIBLE_DEVICES on ROCm).
              Leave all checked to use every GPU. At least one GPU must stay
              selected.
            </InfoHint>
          </div>
          <div className="flex flex-col gap-2">
            {gpuDevices.map((d) => (
              <div
                key={d.index}
                className="flex items-center justify-between gap-3"
              >
                <span className="min-w-0 truncate text-[12px] text-nav-fg/80">
                  GPU {d.index}: {d.name}
                  {d.memoryTotalGb
                    ? ` · ${Math.round(d.memoryTotalGb)} GB`
                    : ""}
                </span>
                <Switch
                  className="panel-switch shrink-0"
                  checked={isGpuChecked(d.index)}
                  onCheckedChange={() => toggleGpu(d.index)}
                  disabled={isGpuChecked(d.index) && singleGpuInUse}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function GgufAdvancedSettings({
  config,
  update,
  isMtp,
  speculativeFallback,
  onEditTemplate,
  layerCount,
  moeLayerCount,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  isMtp: boolean;
  speculativeFallback: string;
  onEditTemplate: () => void;
  layerCount: number | null;
  moeLayerCount: number | null;
}) {
  return (
    <>
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>KV Cache Dtype</span>
          <InfoHint>
            Lower KV cache precision to save VRAM at the cost of some quality.
            f16/bf16 are full precision; q8_0/q5_1/q4_1 are quantized.
          </InfoHint>
        </div>
        <Select
          value={config.kvCacheDtype ?? KV_CACHE_DTYPE_DEFAULT}
          onValueChange={(v) =>
            update({ kvCacheDtype: v === KV_CACHE_DTYPE_DEFAULT ? null : v })
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={`w-[92px] ${SELECT_TRIGGER_CLASS}`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value={KV_CACHE_DTYPE_DEFAULT}>
              {KV_CACHE_DTYPE_DEFAULT}
            </SelectItem>
            {KV_CACHE_DTYPES.map((dtype) => (
              <SelectItem key={dtype} value={dtype}>
                {dtype}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS_WRAP}>Speculative Decoding</span>
          <InfoHint>
            Faster generation with no accuracy hit. Auto picks MTP / ngram based
            on the model and platform. Pick a strategy to force it.
          </InfoHint>
        </div>
        <Select
          value={config.speculativeType ?? speculativeFallback}
          onValueChange={(v) =>
            update({
              speculativeType: v,
              specDraftNMax:
                v === "mtp" || v === "mtp+ngram" ? config.specDraftNMax : null,
            })
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={`w-[124px] shrink-0 ${SELECT_TRIGGER_CLASS}`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            {SPECULATIVE_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {SPECULATIVE_TYPE_LABELS[type]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isMtp && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Draft Tokens</span>
            <InfoHint>
              Max MTP draft tokens per step. Leave blank for the platform
              default (2 on GPU, 3 on CPU/Mac).
            </InfoHint>
          </div>
          <input
            type="number"
            min={1}
            max={16}
            step={1}
            value={config.specDraftNMax ?? ""}
            placeholder="auto"
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") {
                update({ specDraftNMax: null });
                return;
              }
              const parsed = Number.parseInt(raw, 10);
              if (Number.isFinite(parsed)) {
                update({ specDraftNMax: Math.max(1, Math.min(16, parsed)) });
              }
            }}
            aria-label="Speculative decoding draft tokens"
            className={NUMBER_INPUT_CLASS}
          />
        </div>
      )}

      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>Tensor Parallelism</span>
          <InfoHint>
            No effect on a single GPU. On multi-GPU setups, improves tokens/sec
            for dense models. MoE models don't benefit.
          </InfoHint>
        </div>
        <Switch
          className="panel-switch shrink-0"
          checked={config.tensorParallel}
          onCheckedChange={(checked) => update({ tensorParallel: checked })}
        />
      </div>

      <GpuMemorySettings
        config={config}
        update={update}
        layerCount={layerCount}
        moeLayerCount={moeLayerCount}
      />

      <ChatTemplateSetting config={config} onEditTemplate={onEditTemplate} />
    </>
  );
}

interface ModelConfigPageProps {
  target: ModelPickTarget;
  onBack?: () => void;
  onRun: (config: PerModelConfig) => void;
  loadedConfig?: PerModelConfig | null;
  loadedContextLength?: number | null;
  initialConfig?: PerModelConfig | null;
  variant?: "page" | "sidebar";
}

export function ModelConfigPage({
  target,
  onBack,
  onRun,
  loadedConfig = null,
  loadedContextLength = null,
  initialConfig = null,
  variant = "page",
}: ModelConfigPageProps) {
  const rememberId = useId();
  const isActiveModel = loadedConfig != null;
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const loadedDefaultChatTemplate = useChatRuntimeStore(
    (s) => s.defaultChatTemplate,
  );
  const loadedMaxContextLength = useChatRuntimeStore(
    (s) => s.ggufMaxContextLength,
  );
  const resolveInitial = () => {
    const resolved = resolveInitialConfig(target.id, target.ggufVariant);
    if (loadedConfig) {
      return { config: loadedConfig, remembered: resolved.remembered };
    }
    if (initialConfig) {
      return {
        config: initialConfig,
        remembered:
          resolved.remembered &&
          perModelConfigsEqual(initialConfig, resolved.config),
      };
    }
    return resolved;
  };
  const [initial] = useState(resolveInitial);
  const [config, setConfig] = useState<PerModelConfig>(() => initial.config);
  const [remember, setRemember] = useState(() => initial.remembered);
  const [savedRemember, setSavedRemember] = useState(() => initial.remembered);
  const [speculativeFallback] = useState(readPersistedSpeculativeType);
  const [templateOpen, setTemplateOpen] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(() =>
    hasNonDefaultAdvanced(config),
  );
  const nativePathToken =
    target.meta.nativePathToken ??
    (isActiveModel ? activeNativePathToken : null);
  const templateDefaults = useDefaultChatTemplate(
    target.id,
    target.ggufVariant,
    templateOpen,
    nativePathToken,
  );
  const modelMaxPosition = useModelMaxPositionEmbeddings(
    target.id,
    !target.isGguf,
  );
  const hasLoadedDefaultTemplate =
    isActiveModel && loadedDefaultChatTemplate != null;
  const resolvedDefaultTemplate = hasLoadedDefaultTemplate
    ? loadedDefaultChatTemplate
    : templateDefaults.template;
  const resolvedDefaultLoading = hasLoadedDefaultTemplate
    ? false
    : templateDefaults.loading;

  const update = (patch: Partial<PerModelConfig>) =>
    setConfig((current) => ({ ...current, ...patch }));

  // Fetch the GGUF header dims (context + layer/MoE counts) for any GGUF target
  // so the GPU Memory sliders can size themselves; the context is also used
  // below when target.meta doesn't already carry it.
  const contextFetchKey = target.isGguf
    ? `${target.id}\n${target.ggufVariant ?? ""}\n${hfToken || ""}\n${nativePathToken ?? ""}`
    : null;
  const [fetchedStagedDims, setFetchedStagedDims] = useState<{
    key: string;
    contextLength: number | null;
    layerCount: number | null;
    moeLayerCount: number | null;
  } | null>(null);
  useEffect(() => {
    if (contextFetchKey == null) {
      return;
    }
    let cancelled = false;
    void fetchGgufStagedMetadata({
      model_path: target.id,
      gguf_variant: target.ggufVariant ?? null,
      hf_token: hfToken || null,
      nativePathToken,
    })
      .then((dims) => {
        if (!cancelled) {
          setFetchedStagedDims({ key: contextFetchKey, ...dims });
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFetchedStagedDims({
            key: contextFetchKey,
            contextLength: null,
            layerCount: null,
            moeLayerCount: null,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    contextFetchKey,
    target.id,
    target.ggufVariant,
    hfToken,
    nativePathToken,
  ]);
  const stagedDims =
    fetchedStagedDims?.key === contextFetchKey ? fetchedStagedDims : null;

  const isMtp =
    config.speculativeType != null &&
    MTP_SPECULATIVE_TYPES.has(config.speculativeType);
  const nativeContextLength =
    target.meta.contextLength ?? stagedDims?.contextLength ?? null;
  const activeLoadedContext =
    isActiveModel && target.isGguf ? loadedContextLength : null;
  const minContext = CONTEXT_LENGTH_MIN;
  const maxContext = Math.max(
    minContext,
    Math.max(
      nativeContextLength ?? 0,
      activeLoadedContext ?? 0,
      config.customContextLength ?? 0,
    ) || 32768,
  );
  const contextValue = Math.min(
    Math.max(
      config.customContextLength ??
        activeLoadedContext ??
        nativeContextLength ??
        maxContext,
      minContext,
    ),
    maxContext,
  );
  const setContextLength = (v: number) =>
    update({ customContextLength: v });
  const baseline = loadedConfig ?? DEFAULT_PER_MODEL_CONFIG;
  const atBaseline = perModelConfigsEqual(config, baseline);
  // An explicit customContextLength that happens to equal the native ceiling is
  // still a user override (not a default), so Reset must stay enabled for it. It
  // only counts as "at default" when there is no override at all AND the shown
  // context matches native (or the model exposes no native context length).
  const contextAtDefault =
    !target.isGguf ||
    (config.customContextLength == null &&
      (nativeContextLength == null || contextValue === nativeContextLength));
  const atDefault =
    contextAtDefault &&
    perModelConfigsEqual(
      { ...config, customContextLength: null },
      DEFAULT_PER_MODEL_CONFIG,
    );
  const nativeMaxSeqLength =
    floorMaxSeqLength(modelMaxPosition.maxPositionEmbeddings) ??
    MAX_SEQ_LENGTH_MAX;
  // A non-GGUF active model seeds config.maxSeqLength from its loaded value, so
  // the initial view still shows the running context. Once that is cleared
  // (Reset sets it to null), fall back to the app default rather than the loaded
  // runtime value, otherwise a remembered/active override can never be cleared.
  const maxSeqLengthValue =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength);
  const maxSeqLengthMax = Math.max(nativeMaxSeqLength, maxSeqLengthValue);
  // An already-loaded GGUF that was auto-fit below native shows activeLoadedContext
  // while customContextLength stays null. If the user fixes GPU Layers (Manual)
  // and remembers, pin that shown context so a later fresh load keeps the fitted
  // placement instead of sending native/0 for fixed layers and recreating the OOM
  // the same-model reload workaround avoids.
  const pinFixedLayerContext =
    target.isGguf &&
    config.gpuMemoryMode === "manual" &&
    config.gpuLayers != null &&
    config.gpuLayers >= 0 &&
    config.customContextLength == null &&
    activeLoadedContext != null;
  // Persisted record: keep config as-is (a default / just-reset non-GGUF model
  // keeps maxSeqLength null) so isDefaultConfig can still recognise it and clear
  // a remembered override instead of pinning the app-default value.
  const runtimeConfig = target.isGguf
    ? pinFixedLayerContext
      ? { ...config, customContextLength: activeLoadedContext }
      : config
    : config;
  // Load request: the model needs a concrete max length, so substitute the
  // resolved fallback here only, never in the persisted runtimeConfig.
  const loadConfig = target.isGguf
    ? runtimeConfig
    : { ...runtimeConfig, maxSeqLength: maxSeqLengthValue };
  const rememberChanged = remember !== savedRemember;
  const persistenceOnly = isActiveModel && atBaseline && rememberChanged;
  const primaryActionLabel = persistenceOnly
    ? remember
      ? "Save settings"
      : "Forget settings"
    : isActiveModel
      ? "Reload model"
      : "Load model";

  const handleRun = () => {
    const defaultConfig = isDefaultConfig(runtimeConfig);
    let saveFailed = false;
    if (remember) {
      saveFailed = !savePerModelConfig(
        target.id,
        target.ggufVariant,
        runtimeConfig,
      );
    } else {
      saveFailed = !deletePerModelConfig(target.id, target.ggufVariant);
    }
    if (persistenceOnly) {
      if (saveFailed) {
        toast.error("Couldn't save settings for this model.");
        return;
      }
      const nextRemember = remember && !defaultConfig;
      setSavedRemember(nextRemember);
      setRemember(nextRemember);
      toast.success(
        nextRemember
          ? "Settings saved."
          : remember
            ? "Default settings kept."
            : "Settings forgotten.",
      );
      return;
    }
    if (saveFailed) {
      toast.error("Couldn't save these settings, loading with them anyway.");
    }
    onRun(loadConfig);
  };

  return (
    <div className="flex flex-col">
      {variant === "page" && (
        <div className="flex items-center gap-2.5 pb-4">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              className="nav-icon-btn shrink-0 text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
              aria-label="Back to model list"
            >
              <HugeiconsIcon
                icon={ArrowLeft01Icon}
                className="size-4"
                strokeWidth={1.75}
              />
            </button>
          )}
          <div className="min-w-0 flex-1">
            <div className="text-[10px] font-semibold uppercase leading-none tracking-wider text-muted-foreground">
              Run settings
            </div>
            <div className="mt-1.5 truncate text-[14px] font-semibold leading-tight text-nav-fg">
              {target.displayName}
            </div>
          </div>
        </div>
      )}

      <div className="space-y-3.5">
        {target.isGguf && (
          <>
            <div className="space-y-3">
              <div className={ROW_CLASS}>
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className={LABEL_CLASS}>Context Length</span>
                  <InfoHint>
                    Tokens of context to allocate. Higher uses more VRAM.
                    {nativeContextLength != null
                      ? ` This model's native context is ${nativeContextLength.toLocaleString()} tokens.`
                      : ""}
                  </InfoHint>
                </div>
                <NumericValueInput
                  value={contextValue}
                  min={minContext}
                  max={maxContext}
                  step={1}
                  onChange={setContextLength}
                  displayValue={
                    config.customContextLength == null &&
                    nativeContextLength == null &&
                    activeLoadedContext == null
                      ? "Auto"
                      : undefined
                  }
                  ariaLabel="Context Length"
                  className={NUMBER_INPUT_CLASS}
                  size={8}
                />
              </div>
              {nativeContextLength != null ? (
                <Slider
                  min={minContext}
                  max={maxContext}
                  step={128}
                  value={[contextValue]}
                  onValueChange={([v]) => setContextLength(v)}
                  className="panel-slider"
                  aria-label="Context Length"
                />
              ) : null}
              {isActiveModel &&
                loadedMaxContextLength != null &&
                contextValue > loadedMaxContextLength && (
                  <p className="text-[11px] text-amber-500">
                    Exceeds estimated VRAM capacity (
                    {loadedMaxContextLength.toLocaleString()} tokens). The model
                    may use system RAM.
                  </p>
                )}
            </div>

            {showAdvanced && (
              <GgufAdvancedSettings
                config={config}
                update={update}
                isMtp={isMtp}
                speculativeFallback={speculativeFallback}
                onEditTemplate={() => setTemplateOpen(true)}
                layerCount={stagedDims?.layerCount ?? null}
                moeLayerCount={stagedDims?.moeLayerCount ?? null}
              />
            )}

            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-muted-foreground">
                  Advanced settings
                </span>
                <InfoHint>
                  Extra options for how the model loads. Most setups don't need
                  these.
                </InfoHint>
              </div>
              <Switch
                className="panel-switch shrink-0"
                checked={showAdvanced}
                onCheckedChange={setShowAdvanced}
                aria-label="Show advanced settings"
              />
            </div>
          </>
        )}
        {!target.isGguf && (
          <>
            <MaxSeqLengthSetting
              value={maxSeqLengthValue}
              max={maxSeqLengthMax}
              inputMax={MAX_SEQ_LENGTH_MAX}
              onChange={(value) =>
                update({
                  maxSeqLength: clampMaxSeqLength(value, MAX_SEQ_LENGTH_MAX),
                })
              }
            />
            <ChatTemplateSetting
              config={config}
              onEditTemplate={() => setTemplateOpen(true)}
              readOnly={true}
            />
          </>
        )}
      </div>

      <div
        className={
          variant === "sidebar"
            ? "mt-4 flex flex-col gap-3 border-t border-border/60 pt-4"
            : "mt-4 flex items-center justify-between gap-3 border-t border-border/60 pt-4"
        }
      >
        <div className="flex min-w-0 items-center gap-2">
          <Checkbox
            id={rememberId}
            checked={remember}
            onCheckedChange={(checked) => setRemember(checked === true)}
          />
          <label
            htmlFor={rememberId}
            className="cursor-pointer select-none truncate text-[13px] text-nav-fg"
          >
            Remember for this model
          </label>
        </div>
        <div
          className={
            variant === "sidebar"
              ? "flex items-center justify-end gap-2"
              : "flex shrink-0 items-center gap-2"
          }
        >
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={atDefault}
            onClick={() => setConfig({ ...DEFAULT_PER_MODEL_CONFIG })}
          >
            Reset
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-8"
            disabled={isActiveModel && atBaseline && !rememberChanged}
            onClick={handleRun}
          >
            {primaryActionLabel}
          </Button>
        </div>
      </div>

      <ChatTemplateEditorDialog
        open={templateOpen}
        onOpenChange={setTemplateOpen}
        value={config.chatTemplateOverride}
        defaultTemplate={resolvedDefaultTemplate}
        defaultLoading={resolvedDefaultLoading}
        readOnly={!target.isGguf}
        onSave={(override) => update({ chatTemplateOverride: override })}
      />
    </div>
  );
}
