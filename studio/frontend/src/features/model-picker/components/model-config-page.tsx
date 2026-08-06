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
import { usePlatformStore } from "@/config/env";
import {
  GPU_LAYERS_AUTO,
  fetchGgufStagedMetadata,
  readPersistedSpeculativeType,
  resolveStagedDiffusionClassification,
  useChatRuntimeStore,
} from "@/features/chat";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import {
  type GpuIndexKind,
  type SystemGpuDevice,
  cachedPinnableGpuContext,
  pinnableGpuContext,
  reconcileGpuSelection,
  useGpuDevices,
} from "@/hooks/use-gpu-info";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  type Ref,
  useEffect,
  useId,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { syncModelOverride } from "../api/model-overrides";
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
  MLX_KV_BITS,
  MTP_SPECULATIVE_TYPES,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  type PerModelConfig,
  SPECULATIVE_TYPES,
  deletePerModelConfig,
  floorMaxSeqLength,
  isDefaultConfig,
  isServedByMlx,
  normalizeMaxSeqLength,
  normalizePerModelConfig,
  readAdvancedSettingsOpen,
  resolveInitialConfig,
  saveAdvancedSettingsOpen,
  savePerModelConfig,
  subscribeAdvancedSettingsOpen,
} from "../model-config/per-model-config";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import type { ModelPickTarget } from "./model-selector/types";
import {
  NumericValueInput,
  type NumericValueInputHandle,
} from "./numeric-value-input";

const ROW_CLASS = "flex min-h-8 items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 truncate text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg";
const LABEL_CLASS_WRAP =
  "min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg";
const CONTROL_SURFACE =
  "rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1]";
const SELECT_TRIGGER_CLASS = `grid h-8 min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-ui-13! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0`;
const NUMBER_INPUT_CLASS = `h-8 w-[92px] ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-right text-ui-13 font-medium text-nav-fg outline-none focus-visible:ring-0`;

const KV_CACHE_DTYPE_DEFAULT = "f16";
const SPECULATIVE_TYPE_LABELS: Record<
  (typeof SPECULATIVE_TYPES)[number],
  string
> = {
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
    config.nParallel != null ||
    config.tensorParallel ||
    config.chatTemplateOverride != null ||
    (config.gpuMemoryMode ?? "auto") !== "auto" ||
    (config.gpuLayers != null && config.gpuLayers >= 0) ||
    (config.nCpuMoe ?? 0) > 0 ||
    config.selectedGpuIds != null
  );
}

function withoutUnsupportedDiffusionSettings(
  config: PerModelConfig,
  currentGpuIndexKind: GpuIndexKind | null = null,
): PerModelConfig {
  const hasUnsupportedGpuPick =
    config.selectedGpuIds != null &&
    (config.selectedGpuIndexKind === "vulkan" ||
      currentGpuIndexKind === "vulkan");
  if (
    (config.gpuMemoryMode ?? "auto") === "auto" &&
    config.gpuLayers == null &&
    config.nCpuMoe == null &&
    !config.tensorParallel &&
    !hasUnsupportedGpuPick
  ) {
    return config;
  }
  return {
    ...config,
    gpuMemoryMode: "auto",
    gpuLayers: undefined,
    nCpuMoe: undefined,
    tensorParallel: false,
    ...(hasUnsupportedGpuPick
      ? {
          selectedGpuIds: undefined,
          selectedGpuIndexKind: undefined,
        }
      : {}),
  };
}

function reconcileConfigGpuSelection(
  config: PerModelConfig,
  isDiffusion: boolean,
  gpuDevices?: SystemGpuDevice[],
): PerModelConfig {
  const context = cachedPinnableGpuContext(isDiffusion, gpuDevices);
  const supported = isDiffusion
    ? withoutUnsupportedDiffusionSettings(config, context.indexKind ?? null)
    : config;
  if (supported.selectedGpuIds == null) {
    return supported;
  }
  const reconciled = reconcileGpuSelection(
    supported.selectedGpuIds,
    supported.selectedGpuIndexKind,
    context.indexKind,
    context.ids,
  );
  const next = {
    ...supported,
    selectedGpuIds: reconciled.ids ?? undefined,
    selectedGpuIndexKind:
      reconciled.ids === null ? undefined : reconciled.indexKind,
  };
  return perModelConfigsEqual(next, supported) ? supported : next;
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
            ? "Preview the model's chat template. This model's backend cannot take a custom one."
            : "Override the model's chat template with custom Jinja. Applies when the model loads."}
        </InfoHint>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {readOnly ? null : (
          <span className="text-ui-12 text-muted-foreground">
            {config.chatTemplateOverride ? "Custom" : "Default"}
          </span>
        )}
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className={`h-8 px-3 text-ui-13 ${CONTROL_SURFACE}`}
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
  inputRef,
}: {
  value: number;
  max: number;
  inputMax: number;
  onChange: (value: number) => void;
  inputRef?: Ref<NumericValueInputHandle>;
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
          ref={inputRef}
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
  inputRef,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  displayValue?: string;
  info?: ReactNode;
  inputRef?: Ref<NumericValueInputHandle>;
}) {
  return (
    <div className="space-y-3">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>{label}</span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <NumericValueInput
          ref={inputRef}
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
// GGUF only. Slider ceilings come from the GGUF header dims, the picker from the
// live device list. --tensor-split is not persisted per model, so not exposed here.
function GpuMemorySettings({
  config,
  update,
  layerCount,
  moeLayerCount,
  isDiffusion,
  gpuDevices,
  gpuLayersInputRef,
  moeLayersInputRef,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  layerCount: number | null;
  moeLayerCount: number | null;
  isDiffusion: boolean;
  gpuDevices: SystemGpuDevice[];
  gpuLayersInputRef?: Ref<NumericValueInputHandle>;
  moeLayersInputRef?: Ref<NumericValueInputHandle>;
}) {
  const mode = config.gpuMemoryMode ?? "auto";
  const isManual = mode === "manual";
  const gpuLayers = config.gpuLayers ?? GPU_LAYERS_AUTO;
  // Slider at Auto: llama.cpp --fit owns the layout, so MoE-offload doesn't apply.
  const autoLayers = isManual && gpuLayers < 0;
  // Ceiling = layer count + 1 (llama.cpp counts the output layer as offloadable),
  // else a safe fallback.
  const gpuLayersMax = layerCount != null ? layerCount + 1 : 256;
  const nCpuMoe = config.nCpuMoe ?? 0;
  const moeLayersMax = moeLayerCount ?? 0;
  const showMoeSlider = isManual && !autoLayers && moeLayersMax > 0;
  const selectedGpuIds = config.selectedGpuIds ?? null;
  const gpuContext = pinnableGpuContext(gpuDevices, isDiffusion);
  const pinnableDevices = gpuContext.devices ?? [];
  const gpuIndexKind = gpuContext.indexKind ?? null;
  const singleGpuInUse = (selectedGpuIds ?? gpuContext.ids ?? []).length <= 1;
  // Multi-GPU only, with one backend-declared index namespace. null = automatic.
  const showGpuPicker = (gpuContext.ids?.length ?? 0) > 1;
  const isGpuChecked = (index: number) =>
    selectedGpuIds === null || selectedGpuIds.includes(index);
  const toggleGpu = (index: number) => {
    const all = gpuContext.ids ?? [];
    const current = selectedGpuIds ?? all;
    const next = current.includes(index)
      ? current.filter((i) => i !== index)
      : [...current, index].sort((a, b) => a - b);
    if (next.length === 0) return; // keep at least one GPU selected
    update({
      selectedGpuIds: next,
      selectedGpuIndexKind: gpuIndexKind,
    });
  };
  return (
    <>
      <div className={isDiffusion ? "hidden" : ROW_CLASS}>
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
            // Default clears every Manual-only setting.
            update(
              v === "manual"
                ? { gpuMemoryMode: "manual" }
                : {
                    gpuMemoryMode: "auto",
                    gpuLayers: undefined,
                    nCpuMoe: undefined,
                    selectedGpuIds: undefined,
                    selectedGpuIndexKind: undefined,
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
      {!isDiffusion && isManual && (
        <>
          <AdvancedGpuSlider
            label="GPU Layers"
            inputRef={gpuLayersInputRef}
            value={Math.max(GPU_LAYERS_AUTO, Math.min(gpuLayers, gpuLayersMax))}
            min={GPU_LAYERS_AUTO}
            max={gpuLayersMax}
            onChange={(v) => update({ gpuLayers: v })}
            displayValue={autoLayers ? "Auto" : undefined}
            info={
              <>
                Layers to keep on the GPU (--gpu-layers); the rest run on CPU.
                Auto lets llama.cpp size the split (and the context) to fit
                VRAM. At the maximum, the whole model is on the GPU.
              </>
            }
          />
          {showMoeSlider && (
            <AdvancedGpuSlider
              label="MoE Layers on CPU"
              inputRef={moeLayersInputRef}
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
              By default, Unsloth chooses GPUs automatically. Editing this list
              makes the checked GPUs the explicit candidate pool. At least one
              GPU must stay selected.
            </InfoHint>
          </div>
          <div className="flex flex-col gap-2">
            {pinnableDevices.map((d) => (
              <div
                key={d.index}
                className="flex items-center justify-between gap-3"
              >
                <span className="min-w-0 truncate text-ui-12 text-nav-fg/80">
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

const MLX_KV_BITS_AUTO = "auto";

function AdvancedSettingsToggle({
  checked,
  onCheckedChange,
}: {
  checked: boolean;
  onCheckedChange: (next: boolean) => void;
}) {
  return (
    <div className={ROW_CLASS}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
          Advanced settings
        </span>
        <InfoHint>
          Extra options for how the model loads. Most setups don't need these.
        </InfoHint>
      </div>
      <Switch
        className="panel-switch shrink-0"
        checked={checked}
        onCheckedChange={onCheckedChange}
        aria-label="Show advanced settings"
      />
    </div>
  );
}

function MlxAdvancedSettings({
  config,
  update,
  outcome,
  servedByMlx,
  onEditTemplate,
  templateOutcome,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  /** What the backend reported for this exact setting on the loaded model. */
  outcome: string | null;
  /** KV quantization is MLX-only; a CUDA safetensors model has no such control. */
  servedByMlx: boolean;
  onEditTemplate: () => void;
  /** Why the loaded model could not take the override it was given. */
  templateOutcome: string | null;
}) {
  return (
    <div className="flex flex-col gap-1">
      {servedByMlx && (
        <>
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>KV Cache Dtype</span>
          <InfoHint>
            Lower KV cache precision to save memory at the cost of some
            quality. Auto keeps full precision; 8-bit is the safest reduction,
            and lower widths save more memory.
          </InfoHint>
        </div>
        <Select
          value={config.mlxKvBits ? String(config.mlxKvBits) : MLX_KV_BITS_AUTO}
          onValueChange={(v) =>
            update({ mlxKvBits: v === MLX_KV_BITS_AUTO ? null : Number(v) })
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
            <SelectItem value={MLX_KV_BITS_AUTO}>Auto</SelectItem>
            {MLX_KV_BITS.map((bits) => (
              <SelectItem key={bits} value={String(bits)}>
                {bits}-bit
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {outcome ? (
        <p className="text-[11px] leading-snug text-muted-foreground">
          {outcome}
        </p>
      ) : null}
        </>
      )}
      <ChatTemplateSetting
        config={config}
        onEditTemplate={onEditTemplate}
        readOnly={!servedByMlx}
      />
      {templateOutcome ? (
        <p className="text-[11px] leading-snug text-muted-foreground">
          {templateOutcome}
        </p>
      ) : null}
    </div>
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
  isDiffusion,
  gpuDevices,
  gpuLayersInputRef,
  moeLayersInputRef,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  isMtp: boolean;
  speculativeFallback: string;
  onEditTemplate: () => void;
  layerCount: number | null;
  moeLayerCount: number | null;
  isDiffusion: boolean;
  gpuDevices: SystemGpuDevice[];
  gpuLayersInputRef?: Ref<NumericValueInputHandle>;
  moeLayersInputRef?: Ref<NumericValueInputHandle>;
}) {
  return (
    <>
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>KV Cache Dtype</span>
          <InfoHint>
            Lower KV cache precision to save VRAM at the cost of some quality.
            f16 is the default; bf16 and f32 are full precision; q8_0 through
            iq4_nl are quantized.
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
          <span className={LABEL_CLASS}>Parallel Slots</span>
          <InfoHint>
            llama-server decode slots (--parallel) for concurrent requests.
            Leave blank for the server default. More slots share the context
            pool and use more VRAM; if they don't fit on GPU, fewer slots are
            launched.
          </InfoHint>
        </div>
        <input
          type="number"
          min={N_PARALLEL_MIN}
          max={N_PARALLEL_MAX}
          step={1}
          value={config.nParallel ?? ""}
          placeholder="auto"
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") {
              update({ nParallel: null });
              return;
            }
            const parsed = Number.parseInt(raw, 10);
            if (Number.isFinite(parsed)) {
              update({
                nParallel: Math.max(
                  N_PARALLEL_MIN,
                  Math.min(N_PARALLEL_MAX, parsed),
                ),
              });
            }
          }}
          aria-label="Parallel decode slots"
          className={NUMBER_INPUT_CLASS}
        />
      </div>

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
        isDiffusion={isDiffusion}
        gpuDevices={gpuDevices}
        gpuLayersInputRef={gpuLayersInputRef}
        moeLayersInputRef={moeLayersInputRef}
      />

      <ChatTemplateSetting config={config} onEditTemplate={onEditTemplate} />
    </>
  );
}

interface ModelConfigPageProps {
  target: ModelPickTarget;
  onBack?: () => void;
  onRun: (config: PerModelConfig, isDiffusion?: boolean) => void;
  loadedConfig?: PerModelConfig | null;
  loadedContextLength?: number | null;
  initialConfig?: PerModelConfig | null;
  isDiffusion?: boolean;
  variant?: "page" | "sidebar";
  /**
   * Page variant only: render the built-in "Run settings" title block. A host that
   * already shows the model name as its page heading turns this off.
   */
  showHeader?: boolean;
}

export function ModelConfigPage({
  target,
  onBack,
  onRun,
  loadedConfig = null,
  loadedContextLength = null,
  initialConfig = null,
  isDiffusion = false,
  variant = "page",
  showHeader = true,
}: ModelConfigPageProps) {
  const rememberId = useId();
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  const platformChatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const mlxKvQuantReason = useChatRuntimeStore((s) => s.mlxKvQuantReason);
  const chatTemplateOverrideReason = useChatRuntimeStore(
    (s) => s.chatTemplateOverrideReason,
  );
  const loadedChatTemplateOverride = useChatRuntimeStore(
    (s) => s.loadedChatTemplateOverride,
  );
  const mlxKvQuantNote = useChatRuntimeStore((s) => s.mlxKvQuantNote);
  const loadedMlxKvBitsRequested = useChatRuntimeStore(
    (s) => s.loadedMlxKvBitsRequested,
  );
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
  // What settings are stored under, which is not always what loads. Every read, write
  // and mirror uses it; the probes keep target.id, since they have to open the model.
  const configId = target.configId ?? target.id;
  const gpuDevices = useGpuDevices();
  const resolveInitial = () => {
    const resolved = resolveInitialConfig(configId, target.ggufVariant);
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
  const [config, setConfig] = useState<PerModelConfig>(() =>
    reconcileConfigGpuSelection(initial.config, isDiffusion, gpuDevices),
  );
  const [remember, setRemember] = useState(() => initial.remembered);
  const [savedRemember, setSavedRemember] = useState(() => initial.remembered);
  const [speculativeFallback] = useState(readPersistedSpeculativeType);
  const [templateOpen, setTemplateOpen] = useState(false);
  // Compare against what the backend was asked for, not what it applied: a
  // refusal applies nothing, and staging a new value must retire the verdict
  // rather than leave it under a request it never answered.
  const chatTemplateOutcome =
    isActiveModel &&
    (config.chatTemplateOverride ?? null) === (loadedChatTemplateOverride ?? null)
      ? chatTemplateOverrideReason
      : null;
  const mlxKvQuantOutcome =
    isActiveModel &&
    (config.mlxKvBits ?? null) === (loadedMlxKvBitsRequested ?? null)
      ? mlxKvQuantReason || mlxKvQuantNote || null
      : null;
  const servedByMlx = isServedByMlx(
    target.isGguf,
    platformDeviceType,
    platformChatOnlyReason,
  );
  // Read live, not snapshotted at mount: the sidebar copy of Run Settings stays
  // mounted while collapsed, so it has to follow a toggle made in the picker.
  const advancedPreference = useSyncExternalStore(
    subscribeAdvancedSettingsOpen,
    readAdvancedSettingsOpen,
    () => null,
  );
  // Until the switch is used anywhere, a model carrying non-default advanced
  // values opens the section on its own so those stay visible. Frozen at mount
  // so editing a field back to its default cannot close the section underfoot.
  const [autoOpenAdvanced] = useState(() => hasNonDefaultAdvanced(config));
  const showAdvanced = advancedPreference ?? autoOpenAdvanced;
  const toggleAdvanced = saveAdvancedSettingsOpen;
  const contextInputRef = useRef<NumericValueInputHandle>(null);
  const maxSeqLengthInputRef = useRef<NumericValueInputHandle>(null);
  const gpuLayersInputRef = useRef<NumericValueInputHandle>(null);
  const moeLayersInputRef = useRef<NumericValueInputHandle>(null);
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

  // Fetch GGUF header dims (context + layer/MoE counts) to size the GPU Memory
  // sliders; the context also fills in below when target.meta lacks it.
  const contextFetchKey = target.isGguf
    ? `${target.id}\n${target.ggufVariant ?? ""}\n${hfToken || ""}\n${nativePathToken ?? ""}`
    : null;
  const [fetchedStagedDims, setFetchedStagedDims] = useState<{
    key: string;
    contextLength: number | null;
    layerCount: number | null;
    moeLayerCount: number | null;
    isDiffusion?: boolean;
    diffusionUnknown?: boolean;
  } | null>(null);
  useEffect(() => {
    if (contextFetchKey == null) {
      return;
    }
    let cancelled = false;
    const settleWithoutMetadata = () => {
      if (!cancelled) {
        setFetchedStagedDims({
          key: contextFetchKey,
          contextLength: null,
          layerCount: null,
          moeLayerCount: null,
          isDiffusion: undefined,
        });
      }
    };
    void (async () => {
      const preparedToken = await prepareHfTokenForUse(hfToken || null);
      if (cancelled) {
        return;
      }
      if (!preparedToken.proceed) {
        settleWithoutMetadata();
        return;
      }
      const dims = await fetchGgufStagedMetadata({
        model_path: target.id,
        gguf_variant: target.ggufVariant ?? null,
        hf_token: preparedToken.token,
        nativePathToken,
      });
      if (!cancelled) {
        setFetchedStagedDims({ key: contextFetchKey, ...dims });
      }
    })().catch(() => {
      settleWithoutMetadata();
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
  const stagedMetadataPending =
    contextFetchKey != null &&
    stagedDims == null &&
    (config.gpuMemoryMode === "manual" || config.selectedGpuIds != null);
  // Tri-state on purpose: an inconclusive probe stays undefined so onRun hands
  // "unknown" to the selection. Collapsing it to false would tell a compare pane
  // this is an ordinary GGUF, letting it inherit another model's split (#7574).
  const classifiedIsDiffusion = resolveStagedDiffusionClassification(
    isDiffusion,
    stagedDims,
  );
  const resolvedIsDiffusion = classifiedIsDiffusion === true;
  const gpuIndexKind =
    pinnableGpuContext(gpuDevices, resolvedIsDiffusion).indexKind ?? null;
  useEffect(() => {
    setConfig((current) =>
      reconcileConfigGpuSelection(current, resolvedIsDiffusion, gpuDevices),
    );
  }, [gpuDevices, resolvedIsDiffusion]);

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
  const setContextLength = (v: number) => update({ customContextLength: v });
  const rawBaseline = loadedConfig ?? DEFAULT_PER_MODEL_CONFIG;
  const baseline = resolvedIsDiffusion
    ? withoutUnsupportedDiffusionSettings(rawBaseline, gpuIndexKind)
    : rawBaseline;
  const atBaseline = perModelConfigsEqual(config, baseline);
  // An explicit customContextLength equal to the native ceiling is still an
  // override (Reset stays enabled). "At default" means no override at all AND the
  // shown context matches native (or no native context length is exposed).
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
  // A non-GGUF active model seeds maxSeqLength from its loaded value. Once cleared
  // (Reset sets null), fall back to the app default, not the loaded runtime value,
  // else a remembered/active override can never be cleared.
  const maxSeqLengthValue =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength);
  const maxSeqLengthMax = Math.max(nativeMaxSeqLength, maxSeqLengthValue);
  // An auto-fit-below-native GGUF shows activeLoadedContext while
  // customContextLength stays null. If the user fixes GPU Layers (Manual) and
  // remembers, pin that shown context so a later fresh load keeps the fitted
  // placement instead of sending native/0 for fixed layers and recreating the OOM.
  const loadableConfig = resolvedIsDiffusion
    ? withoutUnsupportedDiffusionSettings(config, gpuIndexKind)
    : config;
  const pinFixedLayerContext =
    target.isGguf &&
    loadableConfig.gpuMemoryMode === "manual" &&
    loadableConfig.gpuLayers != null &&
    loadableConfig.gpuLayers >= 0 &&
    loadableConfig.customContextLength == null &&
    activeLoadedContext != null;
  // Persisted record: keep config as-is (non-GGUF keeps maxSeqLength null) so
  // isDefaultConfig recognises it and clears a remembered override instead of
  // pinning the app default.
  const runtimeConfig = target.isGguf
    ? pinFixedLayerContext
      ? { ...loadableConfig, customContextLength: activeLoadedContext }
      : loadableConfig
    : loadableConfig;
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
    // Same-click Load/Reload: a numeric draft the user just typed is flushed only
    // by that input's blur handler, which updates the parent config after this
    // click closure already captured the stale value. Commit every numeric input
    // imperatively so the staged load honors what the user just typed, not just
    // the Context field.
    const committedContext = target.isGguf
      ? contextInputRef.current?.commit()
      : undefined;
    const committedMaxSeqLength = target.isGguf
      ? undefined
      : maxSeqLengthInputRef.current?.commit();
    const committedGpuLayers = target.isGguf
      ? gpuLayersInputRef.current?.commit()
      : undefined;
    const committedMoeLayers = target.isGguf
      ? moeLayersInputRef.current?.commit()
      : undefined;

    const pendingPatch: Partial<PerModelConfig> = {};
    if (committedContext != null) {
      pendingPatch.customContextLength = committedContext;
    }
    if (committedMaxSeqLength != null) {
      pendingPatch.maxSeqLength = clampMaxSeqLength(
        committedMaxSeqLength,
        MAX_SEQ_LENGTH_MAX,
      );
    }
    if (committedGpuLayers != null) {
      pendingPatch.gpuLayers = committedGpuLayers;
    }
    if (committedMoeLayers != null) {
      pendingPatch.nCpuMoe = committedMoeLayers;
    }
    const hasPending =
      committedContext != null ||
      committedMaxSeqLength != null ||
      committedGpuLayers != null ||
      committedMoeLayers != null;

    const committedConfig = hasPending
      ? { ...config, ...pendingPatch }
      : config;
    const effectiveConfig = resolvedIsDiffusion
      ? withoutUnsupportedDiffusionSettings(committedConfig, gpuIndexKind)
      : committedConfig;
    // pinFixedLayerContext above was computed from the render-time config, before
    // the same-click GPU Layers draft was committed. Recompute it from
    // effectiveConfig so committing a positive fixed-layer value still pins the
    // fitted context; otherwise the saved config carries customContextLength: null
    // and a later fresh load sends the native context with fixed layers (the OOM
    // the pin exists to avoid).
    const effectivePinFixedLayerContext =
      target.isGguf &&
      effectiveConfig.gpuMemoryMode === "manual" &&
      effectiveConfig.gpuLayers != null &&
      effectiveConfig.gpuLayers >= 0 &&
      effectiveConfig.customContextLength == null &&
      activeLoadedContext != null;
    const effectiveRuntimeConfig = hasPending
      ? effectivePinFixedLayerContext
        ? { ...effectiveConfig, customContextLength: activeLoadedContext }
        : effectiveConfig
      : runtimeConfig;
    // Non-GGUF load substitutes the resolved max sequence length; recompute it
    // from the committed draft so a same-click Max Seq Length edit is not lost.
    const effectiveMaxSeqLengthValue =
      committedMaxSeqLength == null
        ? maxSeqLengthValue
        : (normalizeMaxSeqLength(effectiveConfig.maxSeqLength) ??
          clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength));
    // Recheck the committed draft so Save/Forget reloads when needed.
    const effectiveAtBaseline = perModelConfigsEqual(effectiveConfig, baseline);
    const effectivePersistenceOnly =
      isActiveModel && effectiveAtBaseline && rememberChanged;
    // Judge what storage keeps: savePerModelConfig normalizes first, so judging the raw
    // object reported saved while the write dropped it.
    const normalizedRuntimeConfig = normalizePerModelConfig(
      effectiveRuntimeConfig,
    );
    const defaultConfig = isDefaultConfig(normalizedRuntimeConfig);
    let saveFailed = false;
    const evicted: { modelId: string; ggufVariant: string | null }[] = [];
    if (remember) {
      saveFailed = !savePerModelConfig(
        configId,
        target.ggufVariant,
        normalizedRuntimeConfig,
        evicted,
      );
    } else {
      saveFailed = !deletePerModelConfig(configId, target.ggufVariant);
    }
    // Mirror to the server so an API load gets these settings, not app defaults. Best-effort:
    // the localStorage write above already governs this browser, and it is skipped when that
    // write failed, or the two would permanently disagree. Gated on auto-switch reach, not just
    // GGUF-ness, since the resolver skips Ollama, and a native-path lease is the same case: the
    // id is only the file's display name, which the resolver never keys.
    if (
      !saveFailed &&
      (target.apiLoadable ?? target.isGguf) &&
      !nativePathToken
    ) {
      syncModelOverride(
        configId,
        target.ggufVariant,
        remember ? normalizedRuntimeConfig : null,
      );
    }
    // Saving can push the local map over budget and drop other models, whose server
    // entries would keep applying with nothing able to forget them. Not a Forget: only
    // the mirrored fields go, and launch flags set through the API stay.
    for (const dropped of evicted) {
      syncModelOverride(dropped.modelId, dropped.ggufVariant, null, {
        keepLaunchFlags: true,
      });
    }
    if (effectivePersistenceOnly) {
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
    const effectiveLoadConfig = target.isGguf
      ? effectiveRuntimeConfig
      : { ...effectiveRuntimeConfig, maxSeqLength: effectiveMaxSeqLengthValue };
    onRun(effectiveLoadConfig, classifiedIsDiffusion);
  };

  return (
    <div className="flex flex-col">
      {variant === "page" && showHeader && (
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
            <div className="text-ui-10 font-semibold uppercase leading-none tracking-wider text-muted-foreground">
              Run settings
            </div>
            <div className="mt-1.5 truncate text-ui-14 font-semibold leading-tight text-nav-fg">
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
                  ref={contextInputRef}
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
              <p className="text-ui-11 leading-relaxed text-muted-foreground">
                Unsloth automatically fits the context to your device, using the
                full context when memory allows.
              </p>
              {isActiveModel &&
                loadedMaxContextLength != null &&
                contextValue > loadedMaxContextLength && (
                  <p className="text-ui-11 text-amber-500">
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
                isDiffusion={resolvedIsDiffusion}
                gpuDevices={gpuDevices}
                gpuLayersInputRef={gpuLayersInputRef}
                moeLayersInputRef={moeLayersInputRef}
              />
            )}

            <AdvancedSettingsToggle
              checked={showAdvanced}
              onCheckedChange={toggleAdvanced}
            />
          </>
        )}
        {!target.isGguf && (
          <>
            <MaxSeqLengthSetting
              value={maxSeqLengthValue}
              max={maxSeqLengthMax}
              inputMax={MAX_SEQ_LENGTH_MAX}
              inputRef={maxSeqLengthInputRef}
              onChange={(value) =>
                update({
                  maxSeqLength: clampMaxSeqLength(value, MAX_SEQ_LENGTH_MAX),
                })
              }
            />
            {showAdvanced && (
              <MlxAdvancedSettings
                config={config}
                update={update}
                outcome={mlxKvQuantOutcome}
                servedByMlx={servedByMlx}
                onEditTemplate={() => setTemplateOpen(true)}
                templateOutcome={chatTemplateOutcome}
              />
            )}
            <AdvancedSettingsToggle
              checked={showAdvanced}
              onCheckedChange={toggleAdvanced}
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
            className="cursor-pointer select-none truncate text-ui-13 text-nav-fg"
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
            disabled={
              stagedMetadataPending ||
              (isActiveModel && atBaseline && !rememberChanged)
            }
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
        readOnly={!target.isGguf && !servedByMlx}
        onSave={(override) => update({ chatTemplateOverride: override })}
      />
    </div>
  );
}
