// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InferenceEnginePicker } from "./inference-engines";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useLlamaCppBackend } from "@/hooks/use-llama-backend";
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
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  resolveStagedDiffusionClassification,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  distributeByWeight,
  rebalanceSplit,
} from "@/features/chat/stores/chat-runtime-store";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import {
  type VramBudgetSettings,
  dropVramBudgetRetry,
  flushVramBudgetSave,
  isVramBudgetLocked,
  loadVramBudgetSettings,
  setVramBudgetLocked,
  settleVramBudgetSave,
  stageVramBudgetSave,
  subscribeVramBudgetLock,
  subscribeVramBudgetSettings,
  updateVramBudgetSettings,
} from "@/features/settings/api/vram-budget";
import {
  type GpuIndexKind,
  type SystemGpuDevice,
  cachedPinnableGpuContext,
  pinnableGpuContext,
  reconcileGpuSelection,
  useGpuDevices,
  useInferenceGpuInfo,
} from "@/hooks/use-gpu-info";
import {
  DEFAULT_VRAM_FRACTION,
  resolveFreeGpuCapacityGb,
  resolveMemoryCapacityGb,
} from "@/hooks/gpu-vram";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import {
  type ReactNode,
  type Ref,
  type SetStateAction,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { useShallow } from "zustand/react/shallow";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  subscribeModelMemorySettings,
} from "@/features/settings/api/model-memory";
import {
  type LlamaFlagCatalog,
  loadLlamaFlagCatalog,
  loadManagedLlamaFlags,
  subscribeLlamaFlagCatalog,
} from "../api/llama-flags";
import { type MemoryEstimate } from "../api/memory-estimate";
import {
  resolveEstimateContext,
  resolveMlxEstimateContext,
  resolveMlxServedWindow,
  shouldRequestMemoryEstimate,
} from "../model-config/estimate-context";
import { resolveReclaimableMemoryCredit } from "../model-config/memory-fit";
import {
  resolveResidentEstimateRequest,
  selectResidentEstimateSettings,
} from "../model-config/resident-memory-request";
import { useMemoryEstimate } from "../hooks/use-memory-estimate";
import { useInt8PrefillAvailable } from "../hooks/use-int8-prefill-available";
import {
  fetchLoadModelOverride,
  fromApiOverride,
  modelOverrideKey,
  panelOverrideRow,
  syncModelOverride,
} from "../api/model-overrides";
import {
  diagnoseExtraArgs,
  extraArgsAreLoadable,
  formatExtraArgs,
  parseExtraArgs,
  sanitizeStoredExtraArgs,
} from "../model-config/llama-extra-args";
import {
  useDefaultChatTemplate,
  useModelMaxPositionEmbeddings,
} from "../hooks/use-model-defaults";
import { perModelConfigsEqual } from "../model-config/apply-per-model-config";
import {
  clearExtraArgsEditForDraft,
  clearModelConfigDraftEdited,
  isExtraArgsHydratedForDraft,
  isModelConfigDraftEdited,
  markModelConfigDraftEdited,
  markExtraArgsHydratedForDraft,
  modelConfigDraftKey,
  patchModelConfigDraft,
  primeModelConfigDraft,
  readExtraArgsEditForDraft,
  readModelConfigDraft,
  replaceModelConfigDraft,
  retainModelConfigDraft,
  setExtraArgsEditForDraft,
  setExtraArgsEditLoadableForDraft,
  setModelConfigDraftRemember,
  setModelConfigDraftSavedRemember,
  subscribeModelConfigDraft,
} from "../model-config/model-config-draft";
import { loadedConfigSignature } from "../model-config/config-signature";
import { ggufQuantLabel } from "../model-config/model-identity";
import {
  CACHE_RAM_LLAMA_DEFAULT,
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CONTEXT_LENGTH_MIN,
  CTX_CHECKPOINTS_LLAMA_DEFAULT,
  CTX_CHECKPOINTS_MAX,
  CTX_CHECKPOINTS_MIN,
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  DRAFT_N_MAX_SPEC_TYPES,
  kvCacheDtypeOptions,
  LOAD_MODES,
  LOAD_MODE_DEFAULT,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MAX_SEQ_LENGTH_STEP,
  MLX_KV_QUANTS,
  mlxKvQuantLabel,
  type MlxKvQuant,
  N_BATCH_LLAMA_DEFAULT,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  type PerModelConfig,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
  SPECULATIVE_TYPES,
  deletePerModelConfig,
  floorMaxSeqLength,
  isDefaultConfig,
  isReasoningBudgetMessageValid,
  contextPinPatch,
  isServedByMlx,
  savedContextPin,
  normalizeMaxSeqLength,
  normalizePerModelConfig,
  perModelConfigStorageChanged,
  readAdvancedSettingsOpen,
  resolveInitialConfig,
  saveAdvancedSettingsOpen,
  savePerModelConfig,
  subscribeAdvancedSettingsOpen,
  VRAM_BUDGET_PERCENT_STEP,
  vramFractionToPercent,
  vramPercentToFraction,
} from "../model-config/per-model-config";
import { isAudioRuntimeGguf } from "../../audio/audio-cpp-catalog";
import { isNpuModelId, NPU_DEFAULT_CONTEXT_LENGTH } from "../../npu";
import {
  type RunConfigImport,
  SharedRunConfigControls,
  SharedRunConfigReview,
  cancelRunConfigImportForEdit,
  isRunConfigEditorChange,
  isRunConfigVariantUnresolved,
} from "../sharing";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import { MemoryEstimateRow } from "./memory-estimate-row";
import type { ModelPickTarget } from "./model-selector/types";
import {
  NumericValueInput,
  type NumericValueInputHandle,
} from "./numeric-value-input";
import {
  ChevronLeftIcon,
} from "lucide-react";

const ROW_CLASS = "flex min-h-8 items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 truncate text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground";
const LABEL_CLASS_WRAP =
  "min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground";
const CONTROL_SURFACE =
  "rounded-full border-transparent bg-[var(--panel-input-surface)] hover:bg-[var(--panel-input-surface-hover)] dark:bg-[var(--panel-input-surface)] dark:hover:bg-[var(--panel-input-surface-hover)]";
// Fixed width so the box does not jump under the caret; narrow for a ~240px panel.
const INPUT_WIDTH_CLASS = "w-[calc(84px*var(--ui-space-scale,1))] shrink-0";
const SELECT_WIDTH_CLASS = "w-auto max-w-full shrink-0";
const SELECT_TRIGGER_CLASS = `panel-select-trigger grid h-8! min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 ${SELECT_WIDTH_CLASS} [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0`;
const NUMBER_INPUT_CLASS = `panel-field h-8 ${INPUT_WIDTH_CLASS}`;
const TEXT_INPUT_CLASS = `panel-field h-8 ${INPUT_WIDTH_CLASS} min-w-0`;
const FOOTER_BUTTON_CLASS =
  "h-9 w-auto px-4 rounded-full text-ui-13 font-medium tracking-nav";

// Mirrors the backend's Auto default once GPU-only placement is impossible.
const AUTO_OFFLOAD_CONTEXT_LENGTH = 8192;
const KV_CACHE_DTYPE_DEFAULT = "f16";
const SPECULATIVE_TYPE_LABELS: Record<
  (typeof SPECULATIVE_TYPES)[number],
  string
> = {
  auto: "Auto",
  mtp: "MTP",
  dspark: "DSpark",
  dflash: "DFlash",
  ngram: "Ngram",
  "mtp+ngram": "MTP+Ngram",
  off: "Off",
};

const LOAD_MODE_LABELS: Record<(typeof LOAD_MODES)[number], string> = {
  auto: "Auto",
  none: "None",
  mmap: "mmap",
  mlock: "mlock",
  "mmap+mlock": "mmap+mlock",
  dio: "DirectIO",
};

// Mirrors _LOAD_MODE_MLOCK_VALUES | _LOAD_MODE_RESERVING_VALUES in llama_server_args.py.
const RAM_RESERVING_LOAD_MODES = new Set(["none", "mlock", "mmap+mlock"]);

/** null when the pick reaches the command line untouched; Keep resident wins outright. */
function loadModeOverrideNotice(
  mode: string | null,
  settings: ModelMemorySettings | null,
): string | null {
  if (mode == null || settings == null) {
    return null;
  }
  if (settings.keepResident && !settings.noRamReserve) {
    return mode === "mmap+mlock"
      ? null
      : "This will be replaced by mmap+mlock: Keep model in GPU memory, in Settings, owns how the weights are held.";
  }
  if (settings.noRamReserve && RAM_RESERVING_LOAD_MODES.has(mode)) {
    return "This will be removed: Don't reserve system RAM, in Settings, chooses the loading policy. Supported Windows builds skip the mapping for a full GPU offload; other placements use the default mmap path.";
  }
  return null;
}

/** Hard floor 2, else served slots; a build without --kv-unified serves one slot. */
function effectiveBatchFloor(
  requestedSlots: number | null | undefined,
  limits:
    | { defaultParallelSlots?: number; parallelSlotsClamped?: boolean }
    | null
    | undefined,
): number {
  if (limits?.parallelSlotsClamped) {
    return 2;
  }
  return Math.max(2, requestedSlots ?? limits?.defaultParallelSlots ?? 2);
}

function hasNonDefaultAdvanced(config: PerModelConfig): boolean {
  return (
    config.kvCacheDtype != null ||
    (config.speculativeType ?? "auto") !== "auto" ||
    config.specDraftNMax != null ||
    config.specDraftCacheDtype != null ||
    config.nParallel != null ||
    config.reasoningBudget !== -1 ||
    config.reasoningBudgetMessage !== "" ||
    config.nBatch != null ||
    config.nUbatch != null ||
    config.loadMode != null ||
    config.ctxCheckpoints != null ||
    config.cacheRam != null ||
    config.tensorParallel ||
    config.disableVision ||
    config.chatTemplateOverride != null ||
    // Hidden flags change behaviour, so do not open collapsed claiming defaults.
    (config.llamaExtraArgs != null && config.llamaExtraArgs.length > 0) ||
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
    config.reasoningBudget === -1 &&
    config.reasoningBudgetMessage === "" &&
    !config.tensorParallel &&
    !config.disableVision &&
    config.nBatch == null &&
    config.nUbatch == null &&
    (config.llamaExtraArgs == null || config.llamaExtraArgs.length === 0) &&
    !hasUnsupportedGpuPick
  ) {
    return config;
  }
  return {
    ...config,
    gpuMemoryMode: "auto",
    gpuLayers: undefined,
    nCpuMoe: undefined,
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    tensorParallel: false,
    disableVision: false,
    nBatch: null,
    nUbatch: null,
    // The diffusion shim never passes llama-server flags but the load would record them.
    llamaExtraArgs: null,
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
            ? "Preview the model's chat template. This backend cannot take a custom one."
            : "Replace the model's chat template with custom Jinja. Applies on load."}
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
          className={`h-8 px-3.5 text-ui-13 ${CONTROL_SURFACE}`}
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
  isMlx,
  pinned,
  fittedToMemory,
  windowUnknown,
  hint,
}: {
  value: number;
  max: number;
  inputMax: number;
  onChange: (value: number) => void;
  inputRef?: Ref<NumericValueInputHandle>;
  isMlx?: boolean;
  pinned?: boolean;
  fittedToMemory?: boolean;
  windowUnknown?: boolean;
  hint?: string;
}) {
  const label = isMlx ? "Context Length" : "Max Seq Length";
  return (
    <div className="space-y-2">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>{label}</span>
          <InfoHint>
            {hint ?? (isMlx
              ? "Tokens of context the model is sized for." +
                (fittedToMemory
                  ? " Fitted to this machine's memory, which is less than the model's own " +
                    "window. Set a length to ask for a different one."
                  : "")
              : "Maximum context window in tokens. Applies on load.")}
          </InfoHint>
        </div>
        <NumericValueInput
          ref={inputRef}
          value={value}
          min={MAX_SEQ_LENGTH_MIN}
          max={inputMax}
          step={MAX_SEQ_LENGTH_STEP}
          onChange={onChange}
          displayValue={isMlx && windowUnknown ? "—" : undefined}
          derived={isMlx && !pinned}
          ariaLabel={label}
          className={NUMBER_INPUT_CLASS}
          fixedWidth={true}
          size={8}
        />
      </div>
      <Slider
        min={MAX_SEQ_LENGTH_MIN}
        max={max}
        step={MAX_SEQ_LENGTH_STEP}
        // Clamp into range, or the first nudge would step from the shown number onto the bound.
        value={[Math.min(Math.max(value, MAX_SEQ_LENGTH_MIN), max)]}
        onValueChange={([next]) => onChange(next)}
        className="panel-slider"
        aria-label={label}
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
  step = 1,
  disabled = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  displayValue?: string;
  info?: ReactNode;
  inputRef?: Ref<NumericValueInputHandle>;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-2">
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
          step={step}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          className={NUMBER_INPUT_CLASS}
          fixedWidth={true}
          size={8}
          disabled={disabled}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([next]) => onChange(next)}
        className="panel-slider"
        aria-label={label}
        disabled={disabled}
      />
    </div>
  );
}

// Server-wide, not per model. Whole percents so a dragged value round-trips exactly.
function VramBudgetRow() {
  // macOS sizes Metal from _APPLE_UNIFIED_MEMORY_FRACTION and ignores this budget.
  const isMac = usePlatformStore((s) => s.deviceType === "mac");
  const [settings, setSettings] = useState<VramBudgetSettings | null>(null);
  const [percent, setPercent] = useState<number | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const adviceId = useId();
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const wasModelLoading = useRef(false);
  // Locked while Run settles the budget, or an edit would be flushed with the load.
  const [locked, setLocked] = useState(isVramBudgetLocked);
  useEffect(() => subscribeVramBudgetLock(setLocked), []);

  // A queued edit outranks the publish, or a save landing mid-drag would move the slider.
  useEffect(() => {
    if (isMac) {
      return;
    }
    return subscribeVramBudgetSettings((next) => {
      if (saveTimer.current) {
        return;
      }
      setSettings(next);
      setPercent(vramFractionToPercent(next.fraction));
    });
  }, [isMac]);

  useEffect(() => {
    let cancelled = false;
    if (isMac) {
      return;
    }
    loadVramBudgetSettings().then((loaded) => {
      if (cancelled || !loaded) {
        return;
      }
      setSettings(loaded);
      setPercent(vramFractionToPercent(loaded.fraction));
    });
    return () => {
      cancelled = true;
    };
    // deviceType settles once /api/health answers, so re-run if the seed guess flips.
  }, [isMac]);

  // reloadRequired goes stale when a load finishes; refresh on the falling edge only.
  useEffect(() => {
    const finished = wasModelLoading.current && !modelLoading;
    wasModelLoading.current = modelLoading;
    if (isMac || !finished) {
      return;
    }
    let cancelled = false;
    // Forced: an earlier read describes the child being replaced.
    loadVramBudgetSettings({ force: true }).then((loaded) => {
      if (cancelled || !loaded) {
        return;
      }
      setSettings(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, [isMac, modelLoading]);

  // Flush, do not drop: the fraction is held nowhere else.
  useEffect(
    () => () => {
      if (saveTimer.current) {
        clearTimeout(saveTimer.current);
        saveTimer.current = null;
      }
      flushVramBudgetSave()?.catch((error: unknown) => {
        toast.error(
          error instanceof Error ? error.message : "Failed to save VRAM budget",
        );
      });
    },
    [],
  );

  if (isMac || !settings || percent === null) {
    return null;
  }

  const defaultPercent = vramFractionToPercent(settings.defaultFraction);
  // Stored beats UNSLOTH_VRAM_FRACTION; clearing (null) returns to inheriting it.
  const resetBudget = () => {
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    // Drop a queued drag, or its debounce would store back what this clears.
    stageVramBudgetSave(null);
    updateVramBudgetSettings(null)
      .then((next) => {
        setSettings(next);
        setPercent(vramFractionToPercent(next.fraction));
      })
      .catch((error: unknown) => {
        toast.error(
          error instanceof Error ? error.message : "Failed to reset VRAM budget",
        );
      });
  };
  const commit = (next: number) => {
    setPercent(next);
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
    }
    stageVramBudgetSave(vramPercentToFraction(next));
    saveTimer.current = setTimeout(() => {
      saveTimer.current = null;
      flushVramBudgetSave()
        ?.then(setSettings)
        .catch((error: unknown) => {
          // The client re-stages it, where the write generation can tell if it is still the newest.
          toast.error(
            error instanceof Error ? error.message : "Failed to save VRAM budget",
          );
        });
    }, 400);
  };

  return (
    <div className="space-y-1">
      <AdvancedGpuSlider
        label="VRAM Budget"
        value={percent}
        min={vramFractionToPercent(settings.minFraction)}
        max={vramFractionToPercent(settings.maxFraction)}
        step={VRAM_BUDGET_PERCENT_STEP}
        displayValue={`${percent}%`}
        onChange={commit}
        disabled={locked}
        info={
          <div className="flex flex-col gap-1.5">
            <div>
              Share of each GPU Unsloth will claim when it sizes the model and
              context. The rest is left for memory fragmentation, the per-device
              CUDA context on a multi-GPU split, and MoE routing.
            </div>
            <div>
              Applies to every model, not just this one, and takes effect on the
              next load. Default {defaultPercent}%. Even at 100% a load leaves a
              margin on each card, up to the 512 MiB llama.cpp keeps for its own
              fitter, and never more than the default would have reserved.
            </div>
            <div>
              Reset clears the stored value, so UNSLOTH_VRAM_FRACTION applies
              again if it is set.
            </div>
          </div>
        }
      />
      {percent !== defaultPercent && (
        <p id={adviceId} className="text-ui-11 text-amber-500">
          {percent > defaultPercent
            ? "Above the default fits more context but leaves less slack, so a load can run out of memory. llama.cpp treats that as a hard failure rather than falling back."
            : "Below the default is safer on a shared GPU, but a tight fit may push layers onto the CPU and generate slowly."}
        </p>
      )}
      {settings.isStored && (
        <button
          type="button"
          disabled={locked}
          onClick={resetBudget}
          className="text-ui-11 text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Reset to the server default
        </button>
      )}
      {settings.reloadRequired && (
        <p className="text-ui-11 text-muted-foreground">
          The loaded model was sized with a different budget. Reload it to apply
          this one.
        </p>
      )}
    </div>
  );
}

// --tensor-split is set per GPU row but not persisted per model.
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
  // At Auto, llama.cpp --fit owns the layout, so MoE-offload does not apply.
  const autoLayers = isManual && gpuLayers < 0;
  // llama.cpp counts the output layer as offloadable, hence +1.
  const gpuLayersMax = layerCount != null ? layerCount + 1 : 256;
  const nCpuMoe = config.nCpuMoe ?? 0;
  const moeLayersMax = moeLayerCount ?? 0;
  const showMoeSlider = isManual && !autoLayers && moeLayersMax > 0;
  const selectedGpuIds = config.selectedGpuIds ?? null;
  const gpuContext = pinnableGpuContext(gpuDevices, isDiffusion);
  const pinnableDevices = gpuContext.devices ?? [];
  const gpuIndexKind = gpuContext.indexKind ?? null;
  const singleGpuInUse = (selectedGpuIds ?? gpuContext.ids ?? []).length <= 1;
  const showGpuPicker = (gpuContext.ids?.length ?? 0) > 1;
  const isGpuChecked = (index: number) =>
    selectedGpuIds === null || selectedGpuIds.includes(index);
  // List order is the device order the backend pins, so a re-checked GPU goes to the end.
  const orderedGpuIds = selectedGpuIds ?? gpuContext.ids ?? [];
  // Mirrors the backend's --tensor-split gate.
  const splitTotal = Math.max(0, Math.min(gpuLayers, gpuLayersMax));
  const showSplit =
    !isDiffusion &&
    isManual &&
    !autoLayers &&
    splitTotal > 0 &&
    showGpuPicker &&
    orderedGpuIds.length > 1;
  const splitIsPercent = Boolean(config.tensorParallel);
  const splitScale = splitIsPercent ? 100 : splitTotal;
  const tensorSplit = config.tensorSplit ?? null;
  const splitIsCustom =
    tensorSplit != null && tensorSplit.length === orderedGpuIds.length;
  const splitShares = showSplit
    ? distributeByWeight(
        splitScale,
        splitIsCustom
          ? tensorSplit
          : orderedGpuIds.map(
              (id) =>
                pinnableDevices.find((d) => d.index === id)?.memoryTotalGb ?? 1,
            ),
      )
    : [];
  const setSplitShare = (id: number, value: number) => {
    const k = orderedGpuIds.indexOf(id);
    if (k < 0) return;
    update({ tensorSplit: rebalanceSplit(splitScale, splitShares, k, value) });
  };
  const commitGpuIds = (next: number[], nextSplit: number[] | null = null) => {
    if (next.length === 0) return;
    update({
      selectedGpuIds: next,
      selectedGpuIndexKind: gpuIndexKind,
      // Positional, so a different GPU set invalidates it.
      tensorSplit: nextSplit,
    });
  };
  const toggleGpu = (index: number) => {
    commitGpuIds(
      orderedGpuIds.includes(index)
        ? orderedGpuIds.filter((i) => i !== index)
        : [...orderedGpuIds, index],
    );
  };
  const orderedPinnableDevices = [
    ...orderedGpuIds
      .map((id) => pinnableDevices.find((d) => d.index === id))
      .filter((d): d is SystemGpuDevice => d !== undefined),
    ...pinnableDevices.filter((d) => !orderedGpuIds.includes(d.index)),
  ];
  const moveGpu = (index: number, delta: -1 | 1) => {
    const from = orderedGpuIds.indexOf(index);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= orderedGpuIds.length) return;
    const next = [...orderedGpuIds];
    [next[from], next[to]] = [next[to], next[from]];
    let nextSplit: number[] | null = null;
    if (splitIsCustom) {
      nextSplit = [...tensorSplit];
      [nextSplit[from], nextSplit[to]] = [nextSplit[to], nextSplit[from]];
    }
    commitGpuIds(next, nextSplit);
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
                yourself.
              </div>
            </div>
          </InfoHint>
        </div>
        <Select
          value={mode}
          onValueChange={(v) =>
            update(
              v === "manual"
                ? { gpuMemoryMode: "manual" }
                : {
                    gpuMemoryMode: "auto",
                    gpuLayers: undefined,
                    nCpuMoe: undefined,
                    selectedGpuIds: undefined,
                    selectedGpuIndexKind: undefined,
                    tensorSplit: null,
                  },
            )
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={SELECT_TRIGGER_CLASS}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value="auto">Default</SelectItem>
            <SelectItem value="manual">Manual</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {!isDiffusion && (!isManual || autoLayers) && gpuDevices.length > 0 && (
        <VramBudgetRow />
      )}
      {!isDiffusion && isManual && (
        <>
          <AdvancedGpuSlider
            label="GPU Layers"
            inputRef={gpuLayersInputRef}
            value={Math.max(GPU_LAYERS_AUTO, Math.min(gpuLayers, gpuLayersMax))}
            min={GPU_LAYERS_AUTO}
            max={gpuLayersMax}
            onChange={(v) =>
              update(v < 0 ? { gpuLayers: v, tensorSplit: null } : { gpuLayers: v })
            }
            displayValue={autoLayers ? "Auto" : undefined}
            info={
              <>
                Layers to keep on the GPU (--gpu-layers); the rest run on CPU.
                Auto sizes the split to fit VRAM.
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
                  (--n-cpu-moe) to save VRAM. 0 keeps every expert on the GPU.
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
              Unsloth picks GPUs automatically. Checking them here limits it to
              those.
              {!isDiffusion &&
                " Their order here is the order the model gets them."}{" "}
              Keep at least one selected.
              {showSplit &&
                (splitIsPercent
                  ? " The number beside each is its share of every layer, in percent (--tensor-split)."
                  : " The number beside each is how many of the GPU Layers it holds (--tensor-split).")}
            </InfoHint>
            {showSplit && splitIsCustom && (
              <button
                type="button"
                className="ml-auto shrink-0 rounded px-1 text-ui-12 text-muted-foreground hover:text-foreground"
                onClick={() => update({ tensorSplit: null })}
              >
                Reset split
              </button>
            )}
          </div>
          <div className="flex flex-col gap-2">
            {orderedPinnableDevices.map((d, position) => (
              <div
                key={d.index}
                className="flex items-center justify-between gap-3"
              >
                <span className="min-w-0 truncate text-ui-12 text-muted-foreground">
                  GPU {d.index}: {d.name}
                  {d.memoryTotalGb
                    ? ` · ${Math.round(d.memoryTotalGb)} GiB`
                    : ""}
                </span>
                {showSplit && isGpuChecked(d.index) && (
                  <div className="ml-auto flex shrink-0 items-center gap-1">
                    <NumericValueInput
                      value={splitShares[orderedGpuIds.indexOf(d.index)] ?? 0}
                      min={0}
                      max={splitScale}
                      step={1}
                      onChange={(v) => setSplitShare(d.index, v)}
                      derived={!splitIsCustom}
                      ariaLabel={
                        splitIsPercent
                          ? `Share of each layer on GPU ${d.index}, percent`
                          : `Layers on GPU ${d.index}`
                      }
                      className="panel-field h-7 w-[calc(52px*var(--ui-space-scale,1))] shrink-0"
                      fixedWidth={true}
                      size={4}
                    />
                    <span className="w-[3.25em] text-ui-12 text-muted-foreground">
                      {splitIsPercent ? "%" : "layers"}
                    </span>
                  </div>
                )}
                {/* Not for diffusion: that runner drives one device and matches_gpu_ids
                    reduces the request to its lowest id, so the arrows would move a row
                    without moving the model, under help text promising the opposite. */}
                {isGpuChecked(d.index) && !singleGpuInUse && !isDiffusion && (
                  <div className="flex shrink-0 items-center gap-0.5">
                    <button
                      type="button"
                      className="rounded px-1 text-ui-12 text-muted-foreground hover:text-foreground disabled:opacity-30"
                      aria-label={`Move GPU ${d.index} earlier`}
                      disabled={position === 0}
                      onClick={() => moveGpu(d.index, -1)}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="rounded px-1 text-ui-12 text-muted-foreground hover:text-foreground disabled:opacity-30"
                      aria-label={`Move GPU ${d.index} later`}
                      disabled={position >= orderedGpuIds.length - 1}
                      onClick={() => moveGpu(d.index, 1)}
                    >
                      ↓
                    </button>
                  </div>
                )}
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

const MLX_KV_QUANT_AUTO = "auto";

function AdvancedSettingsToggle({
  checked,
  onCheckedChange,
}: {
  checked: boolean;
  onCheckedChange: (next: boolean) => void;
}) {
  return (
    <div className={`${ROW_CLASS} border-t border-border pt-5`}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
          Advanced settings
        </span>
        <InfoHint>
          Extra options for how the model loads. Unsloth already picks the best
          settings for your device, so most setups don't need these.
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

const GGUF_PARALLEL_HINT =
  "Decode slots (--parallel) for concurrent requests. Leave blank for the server " +
  "default. More slots share the context pool and use more VRAM.";

const MLX_PARALLEL_HINT =
  "Chat replies this model decodes at once (--parallel). Leave blank for the " +
  "default. Replies sharing a decode finish sooner together, but each one holds " +
  "its own context in memory for as long as it runs, and nothing reduces the " +
  "number to fit — raise it only if the memory is there.";

function ParallelSlotsRow({
  config,
  update,
  hint,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  hint: string;
}) {
  return (
    <div className={ROW_CLASS}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={LABEL_CLASS}>Parallel Slots</span>
        <InfoHint>{hint}</InfoHint>
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
  );
}

function MlxAdvancedSettings({
  config,
  update,
  outcome,
  servedByMlx,
  int8PrefillAvailable,
  onInt8PrefillChange,
  onEditTemplate,
  templateOutcome,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  outcome: string | null;
  servedByMlx: boolean;
  int8PrefillAvailable: boolean;
  onInt8PrefillChange: (checked: boolean) => void;
  onEditTemplate: () => void;
  templateOutcome: string | null;
}) {
  return (
    <div className="flex flex-col gap-5">
      {servedByMlx && (
        <div className="space-y-1">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>KV Cache Dtype</span>
          <InfoHint>
            Lower KV cache precision to save memory, at some cost to quality.
            Auto keeps full precision; 8-bit is the safest step down.
            TurboQuant holds quality better at the low widths and adds 3.5-bit,
            which uses 3-bit keys beside 4-bit values; sliding-window and
            recurrent layers keep their native cache under it.
          </InfoHint>
        </div>
        <Select
          value={config.mlxKvQuant ?? MLX_KV_QUANT_AUTO}
          onValueChange={(v) =>
            update({ mlxKvQuant: v === MLX_KV_QUANT_AUTO ? null : (v as MlxKvQuant) })
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={SELECT_TRIGGER_CLASS}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value={MLX_KV_QUANT_AUTO}>Auto</SelectItem>
            {MLX_KV_QUANTS.map((quant) => (
              <SelectItem key={quant} value={quant}>
                {mlxKvQuantLabel(quant)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {outcome ? (
        <p className="text-ui-11 text-muted-foreground">{outcome}</p>
      ) : null}
        </div>
      )}
      {servedByMlx && (int8PrefillAvailable || config.mlxInt8Prefill) && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Int8 Prefill</span>
            <span className="shrink-0 rounded-md bg-[rgb(0_0_0_/_calc(0.04*var(--contrast-wash-gain,1)))] px-1.5 py-0.5 text-ui-10 font-medium uppercase tracking-wide text-muted-foreground dark:bg-muted">
              Exp
            </span>
            <InfoHint>
              Reads long prompts faster by doing part of the math at lower
              precision. Answers can change and may be less accurate.
            </InfoHint>
          </div>
          <Switch
            className="panel-switch shrink-0"
            checked={config.mlxInt8Prefill ?? false}
            onCheckedChange={onInt8PrefillChange}
          />
        </div>
      )}
      {servedByMlx && (
        <ParallelSlotsRow config={config} update={update} hint={MLX_PARALLEL_HINT} />
      )}
      <div className="space-y-1">
        <ChatTemplateSetting
          config={config}
          onEditTemplate={onEditTemplate}
          readOnly={!servedByMlx}
        />
        {templateOutcome ? (
          <p className="text-ui-11 text-muted-foreground">{templateOutcome}</p>
        ) : null}
      </div>
    </div>
  );
}

/** Its own component because it must keep following the Model Memory settings. */
function LoadModeRow({
  config,
  update,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
}) {
  const adviceId = useId();
  const [modelMemory, setModelMemory] = useState<ModelMemorySettings | null>(
    null,
  );
  useEffect(() => {
    let cancelled = false;
    loadModelMemorySettings()
      .then((loaded) => {
        if (!cancelled) {
          setModelMemory(loaded);
        }
      })
      .catch(() => {});
    const unsubscribe = subscribeModelMemorySettings(setModelMemory);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);
  const notice = loadModeOverrideNotice(config.loadMode ?? null, modelMemory);
  return (
    <div className="space-y-1">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>Mmap/Mlock</span>
          <InfoHint>
            How the weights are read off disk (--load-mode). Auto is the
            default and lets Unsloth pick. mmap maps the file, mlock keeps the
            model in RAM, DirectIO streams it, and None asks for no special
            mode.
            Model Memory, in Settings, overrides this when it is on.
          </InfoHint>
        </div>
        <Select
          value={config.loadMode ?? LOAD_MODE_DEFAULT}
          onValueChange={(v) =>
            update({ loadMode: v === LOAD_MODE_DEFAULT ? null : v })
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={SELECT_TRIGGER_CLASS}
            aria-describedby={notice ? adviceId : undefined}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            {LOAD_MODES.map((mode) => (
              <SelectItem key={mode} value={mode}>
                {LOAD_MODE_LABELS[mode]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {notice && (
        <p id={adviceId} className="text-ui-11 text-amber-500">
          {notice}
        </p>
      )}
    </div>
  );
}

function GgufAdvancedSettings({
  config,
  update,
  showDraftTokens,
  showSpecDraftCacheDtype,
  speculativeFallback,
  onEditTemplate,
  layerCount,
  moeLayerCount,
  isDiffusion,
  gpuDevices,
  gpuLayersInputRef,
  moeLayersInputRef,
  onExtraArgsLoadableChange,
  draftKey,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  showDraftTokens: boolean;
  showSpecDraftCacheDtype: boolean;
  speculativeFallback: string;
  onEditTemplate: () => void;
  layerCount: number | null;
  moeLayerCount: number | null;
  isDiffusion: boolean;
  gpuDevices: SystemGpuDevice[];
  gpuLayersInputRef?: Ref<NumericValueInputHandle>;
  moeLayersInputRef?: Ref<NumericValueInputHandle>;
  onExtraArgsLoadableChange: (loadable: boolean) => void;
  draftKey: string;
}) {
  const batchAdviceId = useId();
  const ubatchAdviceId = useId();
  const llamaBackend = useLlamaCppBackend();
  // llama-server asserts batch >= 2 and >= slot count, so the loader raises it to max(slots, 2).
  const batchFloor = Math.max(2, config.nParallel ?? 2);
  const batchBelowFloor = config.nBatch != null && config.nBatch < batchFloor;
  // llama.cpp runs at min(batch, ubatch) while /status echoes the request; judged on the emitted batch.
  const effectiveBatch =
    config.nBatch != null
      ? Math.max(config.nBatch, batchFloor)
      : N_BATCH_LLAMA_DEFAULT;
  const ubatchExceedsBatch =
    config.nUbatch != null && config.nUbatch > effectiveBatch;
  return (
    <>
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>KV Cache Dtype</span>
          <InfoHint>
            Lower KV cache precision to save VRAM, at some cost to quality. f16
            is the default; q8_0 through iq4_nl are quantized.
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
            className={SELECT_TRIGGER_CLASS}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value={KV_CACHE_DTYPE_DEFAULT}>
              {KV_CACHE_DTYPE_DEFAULT}
            </SelectItem>
            {kvCacheDtypeOptions(llamaBackend, config.kvCacheDtype).map((dtype) => (
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
            Faster generation. Auto picks the best strategy for the model and
            platform, or choose one to force it. DSpark and DFlash download a
            drafter sidecar (about 11 GB and 1.5 GB) and trade VRAM for speed;
            MTP and ngram do not change output.
          </InfoHint>
        </div>
        <Select
          value={config.speculativeType ?? speculativeFallback}
          onValueChange={(v) =>
            update({
              speculativeType: v,
              specDraftNMax: DRAFT_N_MAX_SPEC_TYPES.has(v)
                ? config.specDraftNMax
                : null,
              // The draft context exists only while a separate model is loaded.
              specDraftCacheDtype: SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(v)
                ? config.specDraftCacheDtype
                : null,
            })
          }
        >
          <SelectTrigger
            animateRadius={false}
            icon={ChevronDownStandardIcon}
            iconClassName="size-3.5"
            className={SELECT_TRIGGER_CLASS}
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

      {showDraftTokens && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Draft Tokens</span>
            <InfoHint>
              Max draft tokens per step. Leave blank for the default (2 or 3,
              depending on the strategy and device).
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

      {showSpecDraftCacheDtype && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS_WRAP}>Spec Decoding KV Cache Dtype</span>
            <InfoHint>
              KV cache precision for the draft model's own context, separate
              from the KV Cache Dtype above. f16 is the default; quantizing it
              saves VRAM on a drafter the target verifies anyway.
            </InfoHint>
          </div>
          <Select
            value={config.specDraftCacheDtype ?? KV_CACHE_DTYPE_DEFAULT}
            onValueChange={(v) =>
              update({
                specDraftCacheDtype: v === KV_CACHE_DTYPE_DEFAULT ? null : v,
              })
            }
          >
            <SelectTrigger
              animateRadius={false}
              icon={ChevronDownStandardIcon}
              iconClassName="size-3.5"
              className={SELECT_TRIGGER_CLASS}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
              <SelectItem value={KV_CACHE_DTYPE_DEFAULT}>
                {KV_CACHE_DTYPE_DEFAULT}
              </SelectItem>
              {kvCacheDtypeOptions(
                llamaBackend,
                config.specDraftCacheDtype,
              ).map((dtype) => (
                <SelectItem key={dtype} value={dtype}>
                  {dtype}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <ParallelSlotsRow config={config} update={update} hint={GGUF_PARALLEL_HINT} />

      {!isDiffusion && (
        <div className="space-y-1">
          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Batch Size</span>
              <InfoHint>
                Logical prompt batch size (--batch-size). Leave blank for the
                default (2048). The micro-batch below usually matters more.
              </InfoHint>
            </div>
            <input
              type="number"
              min={N_BATCH_MIN}
              max={N_BATCH_MAX}
              step={1}
              value={config.nBatch ?? ""}
              placeholder="auto"
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  update({ nBatch: null });
                  return;
                }
                const parsed = Number.parseInt(raw, 10);
                if (Number.isFinite(parsed)) {
                  update({
                    nBatch: Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, parsed)),
                  });
                }
              }}
              aria-label="Prompt batch size"
              aria-describedby={batchBelowFloor ? batchAdviceId : undefined}
              className={NUMBER_INPUT_CLASS}
            />
          </div>
          {batchBelowFloor && (
            <p id={batchAdviceId} className="text-ui-11 text-muted-foreground">
              Too small for llama-server, so the load will raise it to {batchFloor}.
              {config.nParallel != null && config.nParallel > 2
                ? " It needs one output slot per parallel slot."
                : " It cannot run a batch below 2."}
            </p>
          )}
        </div>
      )}

      {!isDiffusion && (
        <div className="space-y-1">
          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Micro-batch Size</span>
              <InfoHint>
                Physical prompt micro-batch size (--ubatch-size). Leave blank
                for the default (512, or 1120 on Gemma 4 vision models). Larger
                values speed up prompt processing but use more VRAM; capped at
                the batch size.
              </InfoHint>
            </div>
            <input
              type="number"
              min={N_BATCH_MIN}
              max={N_BATCH_MAX}
              step={1}
              value={config.nUbatch ?? ""}
              placeholder="auto"
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  update({ nUbatch: null });
                  return;
                }
                const parsed = Number.parseInt(raw, 10);
                if (Number.isFinite(parsed)) {
                  update({
                    nUbatch: Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, parsed)),
                  });
                }
              }}
              aria-label="Prompt micro-batch size"
              aria-describedby={ubatchExceedsBatch ? ubatchAdviceId : undefined}
              className={NUMBER_INPUT_CLASS}
            />
          </div>
          {ubatchExceedsBatch && (
            <p id={ubatchAdviceId} className="text-ui-11 text-muted-foreground">
              Micro-batch is larger than the batch size, so llama.cpp will run at{" "}
              {effectiveBatch}. Raise the batch size to use {config.nUbatch}.
            </p>
          )}
        </div>
      )}

      {!isDiffusion && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Tensor Parallelism</span>
            <InfoHint>
              Speeds up dense models across multiple GPUs. No effect on a single
              GPU, and MoE models don't benefit.
            </InfoHint>
          </div>
          <Switch
            className="panel-switch shrink-0"
            checked={config.tensorParallel}
            onCheckedChange={(checked) => update({ tensorParallel: checked })}
          />
        </div>
      )}

      {!isDiffusion && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Vision</span>
            <InfoHint>
              Loads the vision projector so the model can read images. Turning
              it off frees that VRAM for more layers on the GPU. Text generation
              is unaffected either way.
            </InfoHint>
          </div>
          <Switch
            className="panel-switch shrink-0"
            checked={!config.disableVision}
            onCheckedChange={(checked) => update({ disableVision: !checked })}
          />
        </div>
      )}

      {!isDiffusion && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Reasoning Budget</span>
            <InfoHint>
              Maximum thinking tokens. -1 is unlimited and 0 turns reasoning
              off.
            </InfoHint>
          </div>
          <input
            type="number"
            min={-1}
            max={2_147_483_647}
            step={1}
            value={config.reasoningBudget}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") {
                update({ reasoningBudget: -1 });
                return;
              }
              const parsed = Number.parseInt(raw, 10);
              if (Number.isFinite(parsed)) {
                update({
                  reasoningBudget: Math.max(
                    -1,
                    Math.min(2_147_483_647, parsed),
                  ),
                });
              }
            }}
            aria-label="Reasoning Budget"
            className={NUMBER_INPUT_CLASS}
          />
        </div>
      )}

      {!isDiffusion && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS_WRAP}>Reasoning Budget Message</span>
            <InfoHint>
              Optional text added before the end-of-thinking tag when the budget
              runs out.
            </InfoHint>
          </div>
          <input
            type="text"
            value={config.reasoningBudgetMessage}
            placeholder="None"
            onChange={(event) => {
              if (isReasoningBudgetMessageValid(event.target.value)) {
                update({ reasoningBudgetMessage: event.target.value });
              }
            }}
            aria-label="Reasoning Budget Message"
            className={TEXT_INPUT_CLASS}
          />
        </div>
      )}

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

      {!isDiffusion && (
        <>
          <LoadModeRow config={config} update={update} />

          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Checkpoints</span>
              <InfoHint>
                Checkpoints kept per slot (--ctx-checkpoints), which let a
                sliding-window model rewind instead of re-processing the prompt.
                Leave blank for the default ({CTX_CHECKPOINTS_LLAMA_DEFAULT}); 0
                disables them. Each one costs host memory.
              </InfoHint>
            </div>
            <input
              type="number"
              min={CTX_CHECKPOINTS_MIN}
              max={CTX_CHECKPOINTS_MAX}
              step={1}
              value={config.ctxCheckpoints ?? ""}
              placeholder="auto"
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  update({ ctxCheckpoints: null });
                  return;
                }
                const parsed = Number.parseInt(raw, 10);
                if (Number.isFinite(parsed)) {
                  update({
                    ctxCheckpoints: Math.max(
                      CTX_CHECKPOINTS_MIN,
                      Math.min(CTX_CHECKPOINTS_MAX, parsed),
                    ),
                  });
                }
              }}
              aria-label="Context checkpoints per slot"
              className={NUMBER_INPUT_CLASS}
            />
          </div>

          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Cache RAM</span>
              <InfoHint>
                Host memory in MiB for caching prompt state evicted from a slot
                (--cache-ram), so a returning conversation is not re-processed.
                Leave blank for the default ({CACHE_RAM_LLAMA_DEFAULT}); 0
                disables it and -1 lifts the limit.
              </InfoHint>
            </div>
            <input
              type="number"
              min={CACHE_RAM_MIN}
              max={CACHE_RAM_MAX}
              step={1}
              value={config.cacheRam ?? ""}
              placeholder="auto"
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  update({ cacheRam: null });
                  return;
                }
                const parsed = Number.parseInt(raw, 10);
                if (Number.isFinite(parsed)) {
                  update({
                    cacheRam: Math.max(
                      CACHE_RAM_MIN,
                      Math.min(CACHE_RAM_MAX, parsed),
                    ),
                  });
                }
              }}
              aria-label="Host prompt cache size in MiB"
              className={NUMBER_INPUT_CLASS}
            />
          </div>
        </>
      )}

      {!isDiffusion && (
        <ExtraArgsRow
          config={config}
          update={update}
          onLoadableChange={onExtraArgsLoadableChange}
          draftKey={draftKey}
        />
      )}
    </>
  );
}

/** The long tail of llama-server flags; validate_extra_args is the real boundary. */
function ExtraArgsRow({
  config,
  update,
  onLoadableChange,
  draftKey,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  onLoadableChange: (loadable: boolean) => void;
  draftKey: string;
}) {
  const [catalog, setCatalog] = useState<LlamaFlagCatalog | null>(null);
  const adviceId = useId();
  // The typed draft, not stored tokens, or the other editor re-quotes half-typed input.
  const edit = useSyncExternalStore(
    subscribeModelConfigDraft,
    () => readExtraArgsEditForDraft(draftKey),
  );
  const external = formatExtraArgs(config.llamaExtraArgs);
  // Comparing against the edit's own source stops a re-quote per keystroke.
  const text = edit && edit.source === external ? edit.text : external;

  // Re-read on invalidation: a llama.cpp update replaces the binary while the panel stays open.
  const [catalogEpoch, setCatalogEpoch] = useState(0);
  useEffect(
    () => subscribeLlamaFlagCatalog(() => setCatalogEpoch((epoch) => epoch + 1)),
    [],
  );
  useEffect(() => {
    let cancelled = false;
    loadLlamaFlagCatalog().then((loaded) => {
      if (!cancelled) {
        // Adopt null (cannot verify) too, or the row checks against the previous binary.
        setCatalog(loaded);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [catalogEpoch]);

  // apply_model_memory_policy strips load-mode flags, so a typed --mlock would never be passed.
  const [modelMemory, setModelMemory] = useState<ModelMemorySettings | null>(
    null,
  );
  useEffect(() => {
    let cancelled = false;
    loadModelMemorySettings()
      .then((loaded) => {
        if (!cancelled) {
          setModelMemory(loaded);
        }
      })
      .catch(() => {});
    const unsubscribe = subscribeModelMemorySettings(setModelMemory);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const diagnostics = diagnoseExtraArgs(text, catalog, {
    gpuSelectionActive: config.selectedGpuIds != null,
    manualGpuMemory: config.gpuMemoryMode === "manual",
    // Only manual mode with layers >= 0 rewrites --tensor-split; at Auto the launcher drops it.
    gpuLayers: config.gpuLayers,
    // A clamping build serves one slot, so an explicit Slots value must not raise the floor.
    batchFloor: effectiveBatchFloor(config.nParallel, catalog),
    keepResident: modelMemory?.keepResident ?? false,
    noRamReserve: modelMemory?.noRamReserve ?? false,
  });
  const tokenCount = parseExtraArgs(text).tokens.length;

  // Reported up so the load is not started just to fail.
  const loadable = extraArgsAreLoadable(diagnostics);
  useEffect(() => {
    onLoadableChange(loadable);
    // An unprobed row may only lower the verdict.
    if (catalog !== null || !loadable) {
      setExtraArgsEditLoadableForDraft(draftKey, loadable);
    }
    // No cleanup: the row unmounts on collapse but its tokens still load; cleared on model change.
  }, [loadable, onLoadableChange, draftKey, catalog]);

  const commit = (next: string) => {
    const { tokens } = parseExtraArgs(next);
    // Before the update, so the resulting config change matches this edit's source.
    setExtraArgsEditForDraft(draftKey, {
      text: next,
      source: formatExtraArgs(tokens.length > 0 ? tokens : null),
    });
    // null, not []; toApiOverride turns it into the explicit [] that clears the server copy.
    update({ llamaExtraArgs: tokens.length > 0 ? tokens : null });
  };

  return (
    <div className="space-y-2">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={LABEL_CLASS}>Extra Arguments</span>
        <InfoHint>
          <div className="flex flex-col gap-1.5">
            <div>
              Passed straight to llama-server after the settings above, so
              anything set in both is taken from here.
            </div>
            <div>
              Quote values with spaces or backslashes. Nothing runs a shell, so
              $HOME, ; and | are ordinary characters. Flags Unsloth owns, like
              the model and the port, are refused.
            </div>
          </div>
        </InfoHint>
      </div>
      <div className="space-y-1">
        <div className="panel-text-surface h-20 w-full overflow-hidden corner-squircle">
          <textarea
            value={text}
            onChange={(event) => commit(event.target.value)}
            spellCheck={false}
            placeholder="--rope-scaling yarn --yarn-orig-ctx 32768"
            aria-label="Extra llama-server arguments"
            aria-describedby={diagnostics.length > 0 ? adviceId : undefined}
            className="block size-full resize-none bg-transparent px-3.5 py-2.5 text-left font-mono text-ui-12 leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
          />
        </div>
        {(tokenCount > 0 || diagnostics.length > 0) && (
          <div id={adviceId} className="space-y-1">
            {tokenCount > 0 && (
              <p className="text-ui-11 text-muted-foreground">
                {tokenCount === 1 ? "1 argument" : `${tokenCount} arguments`}
              </p>
            )}
            {diagnostics.map((diagnostic) => (
              <p
                key={diagnostic.message}
                className={
                  diagnostic.level === "error"
                    ? "text-ui-11 text-red-500"
                    : diagnostic.level === "warning"
                      ? "text-ui-11 text-amber-500"
                      : "text-ui-11 text-muted-foreground"
                }
              >
                {diagnostic.message}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
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
  // Wording only ("Unified" vs "Shared"); not the capacity signal, see hasUnifiedMemory.
  const isAppleUnifiedMemory = usePlatformStore((s) => s.appleSilicon);
  const platformChatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const mlxKvQuantReason = useChatRuntimeStore((s) => s.mlxKvQuantReason);
  const chatTemplateOverrideReason = useChatRuntimeStore(
    (s) => s.chatTemplateOverrideReason,
  );
  const loadedChatTemplateOverride = useChatRuntimeStore(
    (s) => s.loadedChatTemplateOverride,
  );
  const loadedLlamaExtraArgs = useChatRuntimeStore((s) => s.loadedLlamaExtraArgs);
  const loadedGpuIds = useChatRuntimeStore((s) => s.loadedGpuIds);
  const loadedGpuIndexKind = useChatRuntimeStore((s) => s.loadedGpuIndexKind);
  const loadedCpuFallback = useChatRuntimeStore((s) => s.loadedCpuFallback);
  const residentModelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const residentEstimateSettings = useChatRuntimeStore(
    useShallow(selectResidentEstimateSettings),
  );
  const mlxKvQuantNote = useChatRuntimeStore((s) => s.mlxKvQuantNote);
  const loadedMlxKvQuantRequested = useChatRuntimeStore(
    (s) => s.loadedMlxKvQuantRequested,
  );
  const isActiveModel = loadedConfig != null;
  const sharedVariantUnresolved = isRunConfigVariantUnresolved(target);
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const loadedDefaultChatTemplate = useChatRuntimeStore(
    (s) => s.defaultChatTemplate,
  );
  const loadedMaxContextLength = useChatRuntimeStore(
    (s) => s.maxContextLength,
  );
  const configId = target.configId ?? target.id;
  const targetIsNpu = isNpuModelId(target.id);
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
  const draftKey = modelConfigDraftKey(configId, target.ggufVariant);
  const liveSignature = loadedConfigSignature(loadedConfig);
  // Layout effect: only layout cleanups run before a same-key remount re-primes.
  useLayoutEffect(() => retainModelConfigDraft(draftKey), [draftKey]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: liveSignature summarizes loadedConfig
  useLayoutEffect(() => {
    const resolved = resolveInitial();
    primeModelConfigDraft(
      draftKey,
      {
        config: reconcileConfigGpuSelection(
          resolved.config,
          isDiffusion,
          gpuDevices,
        ),
        remembered: resolved.remembered,
      },
      liveSignature,
    );
  }, [draftKey, liveSignature]);
  const draftSnapshot = useSyncExternalStore(
    subscribeModelConfigDraft,
    () => readModelConfigDraft(draftKey),
  );
  const initialFallback = useMemo(
    () => resolveInitial(),
    [configId, target.ggufVariant, loadedConfig, initialConfig],
  );
  const configState =
    draftSnapshot?.config ??
    reconcileConfigGpuSelection(initialFallback.config, isDiffusion, gpuDevices);
  const remember = draftSnapshot?.remember ?? initialFallback.remembered;
  const savedRemember =
    draftSnapshot?.savedRemember ?? initialFallback.remembered;
  // The peer's row may refuse text this editor has none of while Advanced is collapsed.
  const sharedExtraArgsEdit = useSyncExternalStore(
    subscribeModelConfigDraft,
    () => readExtraArgsEditForDraft(draftKey),
  );
  const sharedExtraArgsCurrent =
    sharedExtraArgsEdit != null &&
    sharedExtraArgsEdit.source === formatExtraArgs(configState.llamaExtraArgs);
  const sharedExtraArgsRefused =
    sharedExtraArgsCurrent && sharedExtraArgsEdit.loadable === false;
  const sharedExtraArgsCleared =
    sharedExtraArgsCurrent && sharedExtraArgsEdit.loadable === true;
  // Live config for async reads; a closure would hold the value from request start.
  const configRef = useRef(configState);
  configRef.current = configState;
  const rememberRef = useRef(remember);
  rememberRef.current = remember;
  const setConfig = useCallback(
    (action: SetStateAction<PerModelConfig>) => {
      // A plain value replaces, like useState; merging could not clear a GPU pick on Reset.
      patchModelConfigDraft(draftKey, (current) =>
        typeof action === "function" ? action(current) : action,
      );
    },
    [draftKey],
  );
  const setRemember = useCallback(
    (value: boolean) => {
      setModelConfigDraftRemember(draftKey, value);
    },
    [draftKey],
  );
  const setSavedRemember = useCallback(
    (value: boolean) => {
      setModelConfigDraftSavedRemember(draftKey, value);
    },
    [draftKey],
  );
  const [speculativeFallback] = useState(readPersistedSpeculativeType);
  // Only "manual" is persisted, so absent means the standing preference, not Auto.
  const [gpuMemoryModeFallback] = useState(readPersistedGpuMemoryMode);
  const [templateOpen, setTemplateOpen] = useState(false);
  // Held by the panel because the row unmounts when Advanced collapses.
  const [extraArgsLoadable, setExtraArgsLoadable] = useState(true);
  // A load before the server-row read lands would send none of its settings.
  const [extraArgsHydrating, setExtraArgsHydrating] = useState(
    () => !isDiffusion,
  );
  // Only a different model retires the row's objection, not its unmount.
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the model, not on the setter
  useEffect(() => {
    setExtraArgsLoadable(true);
    setExtraArgsHydrating(!isDiffusion);
  }, [configId, target.ggufVariant, target.isGguf, isDiffusion]);

  // Compare to what was requested, so a staged value retires a verdict for a different request.
  const chatTemplateOutcome =
    isActiveModel &&
    (configState.chatTemplateOverride ?? null) ===
      (loadedChatTemplateOverride ?? null)
      ? chatTemplateOverrideReason
      : null;
  const mlxKvQuantOutcome =
    isActiveModel &&
    (configState.mlxKvQuant ?? null) === (loadedMlxKvQuantRequested ?? null)
      ? // Both: dropping the note promises savings before quantization starts.
        [mlxKvQuantReason, mlxKvQuantNote].filter(Boolean).join(". ") || null
      : null;
  const servedByMlx = isServedByMlx(
    target.isGguf,
    platformDeviceType,
    platformChatOnlyReason,
  );
  const int8PrefillAvailable = useInt8PrefillAvailable(
    servedByMlx &&
      !(target.meta.nativePathToken ?? (isActiveModel ? activeNativePathToken : null))
      ? target.id
      : null,
    hfToken || null,
  );
  const [int8PrefillConfirmOpen, setInt8PrefillConfirmOpen] = useState(false);
  // Read live: the sidebar copy stays mounted while collapsed.
  const advancedPreference = useSyncExternalStore(
    subscribeAdvancedSettingsOpen,
    readAdvancedSettingsOpen,
    () => null,
  );
  // Frozen at mount so editing a field back to default cannot close the section.
  const [autoOpenAdvanced, setAutoOpenAdvanced] = useState(() =>
    hasNonDefaultAdvanced(configState),
  );
  const [importedConfig, setImportedConfig] = useState<RunConfigImport | null>(
    null,
  );
  const handleSharedConfigImport = useCallback((imported: RunConfigImport) => {
    setImportedConfig(imported);
    if (Object.keys(imported.changes).length > 0) {
      setAutoOpenAdvanced(true);
    }
  }, []);
  const [initialMlxKvQuant] = useState(() => configState.mlxKvQuant ?? null);
  const [initialMlxInt8Prefill] = useState(
    () => configState.mlxInt8Prefill ?? false,
  );
  // Applicability stays live: MLX can become available after mount.
  const autoOpenForMlxKvQuant = servedByMlx && initialMlxKvQuant != null;
  const autoOpenForMlxInt8Prefill = servedByMlx && initialMlxInt8Prefill;
  const showAdvanced =
    advancedPreference ??
    (autoOpenAdvanced || autoOpenForMlxKvQuant || autoOpenForMlxInt8Prefill);
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
    !target.isGguf && !targetIsNpu,
  );
  const hasLoadedDefaultTemplate =
    isActiveModel && loadedDefaultChatTemplate != null;
  const resolvedDefaultTemplate = hasLoadedDefaultTemplate
    ? loadedDefaultChatTemplate
    : templateDefaults.template;
  const resolvedDefaultLoading = hasLoadedDefaultTemplate
    ? false
    : templateDefaults.loading;

  const contextFetchKey = target.isGguf && !sharedVariantUnresolved
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
  // Tri-state on purpose: collapsing unknown to false lets a compare pane inherit another split.
  const classifiedIsDiffusion = resolveStagedDiffusionClassification(
    isDiffusion,
    stagedDims,
  );
  const resolvedIsDiffusion = classifiedIsDiffusion === true;
  // Audio-runtime GGUFs launch no llama-server, so none of its knobs apply.
  const audioRuntimeGguf =
    target.isGguf && isAudioRuntimeGguf(target.id, target.meta.audioType);

  // llama_extra_args can be set via the API with no UI, so fetch rather than assume none.
  const [hiddenCatalogEpoch, setHiddenCatalogEpoch] = useState(0);
  useEffect(
    () =>
      subscribeLlamaFlagCatalog(() =>
        setHiddenCatalogEpoch((epoch) => epoch + 1),
      ),
    [],
  );
  // biome-ignore lint/correctness/useExhaustiveDependencies: the arguments, the section and the binary are the inputs
  useEffect(() => {
    if (showAdvanced || !target.isGguf || resolvedIsDiffusion) {
      return;
    }
    const args = configState.llamaExtraArgs;
    if (args == null || args.length === 0) {
      // Reset with Advanced collapsed must re-enable Load.
      setExtraArgsLoadable(true);
      return;
    }
    let cancelled = false;
    loadLlamaFlagCatalog().then((catalog) => {
      if (cancelled || !catalog) {
        return;
      }
      const loadable = extraArgsAreLoadable(
        diagnoseExtraArgs(formatExtraArgs(args), catalog, {
          gpuSelectionActive: configState.selectedGpuIds != null,
          manualGpuMemory: configState.gpuMemoryMode === "manual",
          batchFloor: effectiveBatchFloor(configState.nParallel, catalog),
        }),
      );
      // Only ever tightens: formatExtraArgs rebalances quotes, so this cannot see unfinished input.
      if (!loadable) {
        setExtraArgsLoadable(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [
    showAdvanced,
    configState.llamaExtraArgs,
    configState.selectedGpuIds,
    configState.gpuMemoryMode,
    configState.nParallel,
    target.isGguf,
    resolvedIsDiffusion,
    hiddenCatalogEpoch,
  ]);

  // The server copy is shared across Desktop, LAN and tunnel origins; localStorage is only a seed.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the model is the identity
  useEffect(() => {
    if (resolvedIsDiffusion) {
      setExtraArgsHydrating(false);
      return;
    }
    // Same order as the auto-switch loader: load path first, then the derived configId alias.
    const loadId = target.id;
    const fileVariant =
      !target.ggufVariant && loadId.toLowerCase().endsWith(".gguf")
        ? ggufQuantLabel(loadId.replace(/\\/g, "/").split("/").pop() ?? loadId)
        : null;
    const keys = [
      modelOverrideKey(loadId, target.ggufVariant),
      modelOverrideKey(configId, target.ggufVariant),
      loadId,
      ...(fileVariant ? [`${loadId}:${fileVariant}`] : []),
      configId,
    ].filter((key, index, all) => all.indexOf(key) === index);
    // The draft alone: hosts derive different candidate key lists.
    if (isExtraArgsHydratedForDraft(draftKey)) {
      setExtraArgsHydrating(false);
      return;
    }
    let cancelled = false;
    // Anything changed after this was typed in flight; sanitizing it would rewrite live input.
    const configAtStart = configRef.current;
    const storedAtStart = resolveInitialConfig(configId, target.ggufVariant);
    const rememberAtStart = rememberRef.current;
    const localAtStart = configAtStart.llamaExtraArgs;
    // Last-resort release so a request that never settles cannot disable Load forever.
    const release = setTimeout(() => setExtraArgsHydrating(false), 15000);
    Promise.all([
      fetchLoadModelOverride(loadId, configId, target.ggufVariant, keys).then(
        (row) => panelOverrideRow(row, target.isGguf),
      ),
      target.isGguf ? loadManagedLlamaFlags() : null,
    ])
      .then(([resolvedOverride, managed]) => {
        // Marked here, not before the request: StrictMode replays the effect.
        if (cancelled) {
          return;
        }
        markExtraArgsHydratedForDraft(draftKey);
        const resolvedArgs = {
          tokens: resolvedOverride?.llama_extra_args ?? [],
          explicit: Array.isArray(resolvedOverride?.llama_extra_args),
        };
        // Sanitised because hydration makes the list explicit, which /load validates strictly.
        const stored = sanitizeStoredExtraArgs(
          resolvedArgs.tokens,
          managed?.managed ?? new Set<string>(),
          // The host's bounds: Windows takes 24 KiB plus a quoted-command budget.
          {
            maxBytes: managed?.maxBytes,
            windowsCommandBudget: managed?.windowsCommandBudget,
          },
        );
        // A list saved by an older, more permissive build can come from local storage too.
        const local = target.isGguf
          ? configRef.current.llamaExtraArgs
          : undefined;
        // Handing back a refused list would re-enable Load.
        let sanitizedLocal = localAtStart;
        if (local != null && local.length > 0 && local === localAtStart) {
          const cleaned = sanitizeStoredExtraArgs(
            local,
            managed?.managed ?? new Set<string>(),
            {
              maxBytes: managed?.maxBytes,
              windowsCommandBudget: managed?.windowsCommandBudget,
            },
          );
          if (cleaned.length !== local.length) {
            sanitizedLocal = cleaned.length > 0 ? cleaned : null;
            setConfig((current) =>
              current.llamaExtraArgs === local
                ? { ...current, llamaExtraArgs: sanitizedLocal }
                : current,
            );
          }
        }
        const resolvedRow = resolvedOverride
          ? {
              ...resolvedOverride,
              ...(resolvedArgs.explicit ? { llama_extra_args: stored } : {}),
            }
          : null;
        const serverConfig = resolvedRow
          ? fromApiOverride(resolvedRow, {
              ...configAtStart,
              llamaExtraArgs: sanitizedLocal,
            })
          : null;
        // Judged as a whole, or an empty server list would clear a locally refused flag.
        const hydratedArgs = serverConfig?.llamaExtraArgs ?? stored;
        const hydratedIsLoadable =
          !target.isGguf || hydratedArgs.length === 0
            ? true
            : extraArgsAreLoadable(
                diagnoseExtraArgs(
                  formatExtraArgs(hydratedArgs),
                  {
                    flags: {},
                    managed: managed?.managed ?? new Set<string>(),
                    switches: new Set<string>(),
                    maxBytes: managed?.maxBytes ?? 0,
                    windowsCommandBudget: managed?.windowsCommandBudget ?? 0,
                    defaultParallelSlots: managed?.defaultParallelSlots ?? 0,
                    parallelSlotsClamped:
                      managed?.parallelSlotsClamped ?? false,
                    probeOk: false,
                  },
                  {
                    batchFloor: effectiveBatchFloor(
                      serverConfig?.nParallel ?? configRef.current.nParallel,
                      managed,
                    ),
                  },
                ),
              );

        // The shared row outranks the local seed for fields it carries, never over an in-flight edit.
        if (
          resolvedRow &&
          serverConfig &&
          !isModelConfigDraftEdited(draftKey) &&
          configRef.current === configAtStart &&
          rememberRef.current === rememberAtStart
        ) {
          const storedConfig = resolveInitialConfig(
            configId,
            target.ggufVariant,
          );
          if (perModelConfigStorageChanged(storedAtStart, storedConfig)) {
            return;
          }
          setExtraArgsLoadable(hydratedIsLoadable);
          replaceModelConfigDraft(draftKey, serverConfig, {
            remember: true,
            savedRemember: true,
          });
          if (hasNonDefaultAdvanced(serverConfig)) {
            setAutoOpenAdvanced(true);
          }
          // Unconditionally: savePerModelConfig clears by deleting, so a default merge must still travel.
          const rememberedConfig = fromApiOverride(
            resolvedRow,
            storedConfig.config,
          );
          // Eviction is silent, so drop evicted models' mirrored server fields (not a Forget).
          const hydrationEvicted: {
            modelId: string;
            ggufVariant: string | null;
          }[] = [];
          // The write can fail via its return value; do not claim settings are remembered then.
          const hydrationSaved = savePerModelConfig(
            configId,
            target.ggufVariant,
            rememberedConfig,
            hydrationEvicted,
          );
          setSavedRemember(hydrationSaved);
          for (const dropped of hydrationEvicted) {
            syncModelOverride(dropped.modelId, dropped.ggufVariant, null, {
              keepLaunchFlags: true,
            });
          }
          return;
        }
        if (stored.length === 0) {
          // An explicit empty row is a decision; left undefined /load carries the resident args over.
          const local = configRef.current.llamaExtraArgs;
          if (resolvedArgs.explicit && local === undefined) {
            setExtraArgsLoadable(true);
            setConfig((current) =>
              current.llamaExtraArgs === undefined
                ? { ...current, llamaExtraArgs: [] }
                : current,
            );
          }
          return;
        }
        // Decided outside the updater, which must be side-effect free (StrictMode calls it twice).
        if (configRef.current.llamaExtraArgs !== undefined) {
          return;
        }
        setExtraArgsLoadable(hydratedIsLoadable);
        setConfig((current) =>
          // Guarded again, because the ref is only as fresh as the last render.
          current.llamaExtraArgs === undefined
            ? { ...current, llamaExtraArgs: stored }
            : current,
        );
        // Server-only arguments arrive after the auto-open snapshot, so open Advanced now.
        setAutoOpenAdvanced(true);
      })
      .catch(() => {
        // Nothing to say: the panel is still usable, and the load would report a real problem with the
        // overrides service far more clearly.
      })
      .finally(() => {
        // Including failure: a down overrides service must not leave Load disabled.
        if (!cancelled) {
          setExtraArgsHydrating(false);
        }
      });
    return () => {
      cancelled = true;
      clearTimeout(release);
    };
  }, [
    configId,
    target.id,
    target.ggufVariant,
    target.isGguf,
    resolvedIsDiffusion,
    draftKey,
  ]);
  const config = reconcileConfigGpuSelection(
    configState,
    resolvedIsDiffusion,
    gpuDevices,
  );
  const [budgetSettling, setBudgetSettling] = useState(false);
  // The budget is not a per-model field, so track its reload requirement separately.
  const [budgetReloadRequired, setBudgetReloadRequired] = useState(false);
  useEffect(
    () =>
      subscribeVramBudgetSettings((next) => {
        setBudgetReloadRequired(next.reloadRequired);
      }),
    [],
  );
  const stagedMetadataPending =
    contextFetchKey != null &&
    stagedDims == null &&
    (config.gpuMemoryMode === "manual" ||
      config.selectedGpuIds != null ||
      config.nBatch != null ||
      config.nUbatch != null);
  const gpuIndexKind =
    pinnableGpuContext(gpuDevices, resolvedIsDiffusion).indexKind ?? null;
  const handleSharedConfigEdit = () => {
    cancelRunConfigImportForEdit(draftKey);
    setImportedConfig(null);
  };
  const update = (patch: Partial<PerModelConfig>) => {
    handleSharedConfigEdit();
    // Hydration writes go through setConfig; marking those would make the read refuse its result.
    markModelConfigDraftEdited(draftKey);
    setConfig((current) => ({
      ...reconcileConfigGpuSelection(current, resolvedIsDiffusion, gpuDevices),
      ...patch,
    }));
  };

  const showDraftTokens =
    config.speculativeType != null &&
    DRAFT_N_MAX_SPEC_TYPES.has(config.speculativeType);
  // Only sidecar modes always load a second context; MTP may use heads in the target GGUF.
  const showSpecDraftCacheDtype =
    config.speculativeType != null &&
    SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(config.speculativeType);
  const nativeContextLength =
    target.meta.contextLength ?? stagedDims?.contextLength ?? null;
  const activeLoadedContext =
    isActiveModel && target.isGguf ? loadedContextLength : null;
  // resolveLoadMaxSeqLength returns 0 for a builtin-default GGUF load, so do not fall back to it.
  const activePresetSource = useChatRuntimeStore((s) => s.activePresetSource);
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
  const contextIsAuto = config.customContextLength == null;
  const contextInputValue = contextIsAuto
    ? Math.min(
        Math.max(
          activeLoadedContext ?? AUTO_OFFLOAD_CONTEXT_LENGTH,
          minContext,
        ),
        maxContext,
      )
    : contextValue;
  const contextSliderValue = contextIsAuto ? 0 : contextValue;
  const setContextLength = (v: number) => update({ customContextLength: v });
  const setContextSliderValue = (v: number) =>
    update({ customContextLength: v === 0 ? null : v });
  const rawBaseline = loadedConfig ?? DEFAULT_PER_MODEL_CONFIG;
  const baseline = resolvedIsDiffusion
    ? withoutUnsupportedDiffusionSettings(rawBaseline, gpuIndexKind)
    : rawBaseline;
  const platform = usePlatformStore();
  const targetIsMlx = isServedByMlx(
    target.isGguf,
    platform.deviceType,
    platform.chatOnlyReason,
  );
  const pinsContextLength = targetIsMlx || targetIsNpu;
  const atBaseline = perModelConfigsEqual(config, baseline, {
    followGlobal: true,
  });
  // The fitted value is an outcome, not an override.
  const contextAtDefault = !target.isGguf
    ? savedContextPin(config) == null
    : config.customContextLength == null;
  const atDefault =
    contextAtDefault &&
    perModelConfigsEqual(
      { ...config, customContextLength: null },
      DEFAULT_PER_MODEL_CONFIG,
    );
  const nativeMaxSeqLength =
    floorMaxSeqLength(
      targetIsNpu
        ? target.meta.contextLength
        : modelMaxPosition.maxPositionEmbeddings,
    ) ?? MAX_SEQ_LENGTH_MAX;
  // Reported as is, not snapped to the request step.
  const servedWindow = (value: unknown) =>
    typeof value === "number" && Number.isFinite(value) && value > 0
      ? Math.floor(value)
      : null;
  const mlxNativeWindow = targetIsMlx
    ? servedWindow(modelMaxPosition.maxPositionEmbeddings)
    : null;
  // The backend clamps auto-sized windows to the request ceiling.
  const mlxProspectiveWindow =
    mlxNativeWindow == null
      ? null
      : Math.min(mlxNativeWindow, MAX_SEQ_LENGTH_MAX);
  // Pin the shown fitted context when layers are fixed, or a fresh load recreates the OOM.
  const loadableConfig = resolvedIsDiffusion
    ? withoutUnsupportedDiffusionSettings(config, gpuIndexKind)
    : config;
  // Reclassification as diffusion strips the args, so retire the row's objection here.
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the classification, not on the setter
  useEffect(() => {
    if (resolvedIsDiffusion) {
      setExtraArgsLoadable(true);
    }
  }, [resolvedIsDiffusion]);
  const pinFixedLayerContext =
    target.isGguf &&
    loadableConfig.gpuMemoryMode === "manual" &&
    loadableConfig.gpuLayers != null &&
    loadableConfig.gpuLayers >= 0 &&
    loadableConfig.customContextLength == null &&
    activeLoadedContext != null;
  const runtimeConfig = target.isGguf
    ? pinFixedLayerContext
      ? { ...loadableConfig, customContextLength: activeLoadedContext }
      : loadableConfig
    : loadableConfig;
  const runtimeGpuMemoryMode =
    runtimeConfig.gpuMemoryMode ?? gpuMemoryModeFallback;
  // Read the classification as a tri-state: an unclassified GGUF may be DiffusionGemma.
  const memoryEstimateRequest =
    shouldRequestMemoryEstimate({
      isGguf: Boolean(target.isGguf),
      isAppleUnifiedMemory,
      classifiedIsDiffusion,
    })
      ? {
          modelPath: target.id,
          ggufVariant: target.ggufVariant ?? null,
          hfToken: hfToken || null,
          nativePathToken,
          // The context Load sends, not the displayed one (which shows 32,768 before the header).
          nCtx: resolveEstimateContext(
            runtimeConfig.customContextLength ?? null,
            activeLoadedContext,
            (target.isGguf === true &&
              runtimeGpuMemoryMode === "manual" &&
              (runtimeConfig.gpuLayers ?? GPU_LAYERS_AUTO) < 0) ||
              (target.isGguf === true && activePresetSource === "builtin-default"),
          ),
          cacheTypeKv: runtimeConfig.kvCacheDtype,
          maxSeqLength: target.isGguf
            ? null
            : resolveMlxEstimateContext(savedContextPin(config)),
          mlxKvQuant: runtimeConfig.mlxKvQuant ?? null,
          nParallel: runtimeConfig.nParallel,
          nBatch: runtimeConfig.nBatch,
          nUbatch: runtimeConfig.nUbatch,
          ctxCheckpoints: runtimeConfig.ctxCheckpoints ?? null,
          // Same substitution applyPerModelConfigToRuntime makes at load.
          speculativeType: runtimeConfig.speculativeType ?? speculativeFallback ?? null,
          specDraftNMax: runtimeConfig.specDraftNMax,
          specDraftCacheType: runtimeConfig.specDraftCacheDtype ?? null,
          tensorParallel: runtimeConfig.tensorParallel,
          disableVision: runtimeConfig.disableVision,
          gpuMemoryMode: runtimeGpuMemoryMode,
          gpuLayers:
            runtimeConfig.gpuLayers != null &&
            runtimeConfig.gpuLayers !== GPU_LAYERS_AUTO
              ? runtimeConfig.gpuLayers
              : null,
          nCpuMoe: runtimeConfig.nCpuMoe ?? null,
          selectedGpuIds: runtimeConfig.selectedGpuIds ?? null,
          llamaExtraArgs: runtimeConfig.llamaExtraArgs ?? null,
        }
      : null;
  const memoryEstimate = useMemoryEstimate(memoryEstimateRequest);
  const mlxFittedWindow = targetIsMlx
    ? servedWindow(memoryEstimate.estimate?.contextFitted)
    : null;
  const mlxServedWindow = resolveMlxServedWindow(
    targetIsMlx && isActiveModel ? servedWindow(loadedContextLength) : null,
    mlxFittedWindow,
    mlxProspectiveWindow,
  );
  const npuServedWindow = targetIsNpu
    ? ((isActiveModel ? servedWindow(loadedContextLength) : null) ??
      Math.min(NPU_DEFAULT_CONTEXT_LENGTH, nativeMaxSeqLength))
    : null;
  const maxSeqLengthValue =
    servedWindow(savedContextPin(config)) ??
    mlxServedWindow ??
    npuServedWindow ??
    clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength);
  const maxSeqLengthMax = Math.min(
    MAX_SEQ_LENGTH_MAX,
    Math.max(nativeMaxSeqLength, maxSeqLengthValue),
  );
  // activeLoadedContext is GGUF-only; an MLX resident reports its served window here.
  const residentContext = servedWindow(
    targetIsMlx && isActiveModel ? loadedContextLength : activeLoadedContext,
  );
  const residentEstimateRequest = resolveResidentEstimateRequest(
    isActiveModel ? memoryEstimateRequest : null,
    residentEstimateSettings,
    residentContext,
    targetIsMlx && !residentModelLoading
      ? { kvQuant: loadedMlxKvQuantRequested ?? null }
      : undefined,
  );
  const residentEstimate = useMemoryEstimate(residentEstimateRequest, {
    refreshMemory: true,
  });
  const reclaimableEstimate =
    residentEstimateRequest &&
    residentEstimate.estimate?.available &&
    !residentEstimate.loading &&
    !residentEstimate.stale
      ? residentEstimate.estimate
      : null;
  const reclaimableCredit = resolveReclaimableMemoryCredit(
    reclaimableEstimate,
    { ids: loadedGpuIds, indexKind: loadedGpuIndexKind },
    {
      ids: runtimeConfig.selectedGpuIds ?? null,
      indexKind: runtimeConfig.selectedGpuIndexKind ?? null,
    },
    {
      cpuFallback: loadedCpuFallback,
      devices: gpuDevices,
      gpuPlacementKnown:
        residentEstimateRequest?.gpuMemoryMode === "manual" &&
        residentEstimateRequest.gpuLayers != null &&
        residentEstimateRequest.gpuLayers >= 0 &&
        !loadedLlamaExtraArgs?.length,
      appleUnifiedMemory: isAppleUnifiedMemory,
    },
  );
  const [memoryBreakdownOpen, setMemoryBreakdownOpen] = useState(false);
  const inferenceGpu = useInferenceGpuInfo();
  // A pin can only use the cards it names, so judge the fit against those.
  const pinnedGpuIds = runtimeConfig.selectedGpuIds;
  // Per-device unified_memory, `.every()`: one discrete device means real VRAM beside RAM.
  // `[].every()` is true, so the empty set is excluded; use-gpu-info.ts uses `.some()` on purpose.
  const hasUnifiedMemory = useMemo(() => {
    if (isAppleUnifiedMemory) return true;
    const governing =
      pinnedGpuIds && pinnedGpuIds.length > 0
        ? gpuDevices.filter((device) => pinnedGpuIds.includes(device.index))
        : gpuDevices;
    if (governing.length === 0) return false;
    return governing.every((device) => device.unifiedMemory === true);
  }, [gpuDevices, pinnedGpuIds, isAppleUnifiedMemory]);
  // The budget slider caps per-GPU use; seeded with VRAM_FRACTION_DEFAULT, which loads apply
  // until the async read lands (or on an older backend).
  const [memoryVramBudgetFraction, setMemoryVramBudgetFraction] =
    useState(DEFAULT_VRAM_FRACTION);
  useEffect(() => {
    let cancelled = false;
    loadVramBudgetSettings().then((loaded) => {
      if (!cancelled && loaded) {
        setMemoryVramBudgetFraction(loaded.fraction);
      }
    });
    const unsubscribe = subscribeVramBudgetSettings((next) => {
      setMemoryVramBudgetFraction(next.fraction);
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);
  // Fixed Manual placement launches verbatim (--fit off), so the VRAM budget does not apply.
  const memoryBudgetGovernsLaunch =
    runtimeGpuMemoryMode !== "manual" ||
    (runtimeConfig.gpuLayers ?? GPU_LAYERS_AUTO) < 0;
  const memoryEffectiveBudgetFraction = memoryBudgetGovernsLaunch
    ? memoryVramBudgetFraction
    : 1;
  // _HOST_RAM_HEADROOM_MIB: the 2 GiB the loader keeps for the rest of the system.
  const memoryUsableSystemRamGb = Math.max(
    0,
    (inferenceGpu.systemRamAvailableHostGb || 0) - 2,
  );
  const memorySystemRamReserveDeficitGb = Math.max(
    0,
    2 - (inferenceGpu.systemRamAvailableHostGb || 0),
  );
  // The rule lives in gpu-vram.ts so tests exercise it directly.
  const {
    gb: memoryFreeGpuCapacityGb,
    known: memoryFreeGpuCapacityKnown,
    reserveDeficitGb: memoryFreeGpuReserveDeficitGb,
  } = useMemo(
    () =>
      resolveFreeGpuCapacityGb({
        devices: gpuDevices,
        pinnedGpuIds,
        budgetFraction: memoryEffectiveBudgetFraction,
        unifiedMemory: hasUnifiedMemory,
        unifiedPoolReportedAsGpuMemory: isAppleUnifiedMemory,
        usableSystemRamGb: memoryUsableSystemRamGb,
        systemRamReserveDeficitGb: memorySystemRamReserveDeficitGb,
        systemRamAvailableKnown: inferenceGpu.systemRamAvailableKnown,
        loadedGpuIds,
        loadedGpuIndexKind,
      }),
    [
      gpuDevices,
      pinnedGpuIds,
      memoryEffectiveBudgetFraction,
      hasUnifiedMemory,
      isAppleUnifiedMemory,
      memoryUsableSystemRamGb,
      memorySystemRamReserveDeficitGb,
      inferenceGpu.systemRamAvailableKnown,
      loadedGpuIds,
      loadedGpuIndexKind,
    ],
  );
  const {
    gpuCapacityGb: memoryGpuCapacityGb,
    totalCapacityGb: memoryTotalCapacityGb,
    singleMemoryPool,
  } = resolveMemoryCapacityGb({
      gpuBudgetFraction: memoryEffectiveBudgetFraction,
      pinnedDevices:
        pinnedGpuIds && pinnedGpuIds.length > 0
          ? gpuDevices.filter((device) => pinnedGpuIds.includes(device.index))
          : [],
      hostDevices: gpuDevices,
      hostGpuTotalGb: inferenceGpu.memoryTotalGb,
      hostDedicatedGpuTotalGb: inferenceGpu.dedicatedMemoryTotalGb,
      hostSharesSystemRam: inferenceGpu.sharedMemory,
      systemRamTotalGb: inferenceGpu.systemRamTotalGb,
      // General signal: ROCm APUs share one pool like Apple, so appleSilicon would double count.
      unifiedMemory: hasUnifiedMemory,
      // A ROCm APU's memory_total_gb is a BIOS window onto RAM, not the pool size.
      unifiedPoolReportedAsGpuMemory: isAppleUnifiedMemory,
    });

  const rememberChanged = remember !== savedRemember;
  const persistenceOnly = isActiveModel && atBaseline && rememberChanged;
  const primaryActionLabel = persistenceOnly
    ? remember
      ? "Save settings"
      : "Forget settings"
    : isActiveModel
      ? "Reload model"
      : "Load model";

  const [engineReady, setEngineReady] = useState(true);
  const commitDraft = () => {
    // Numeric drafts flush on blur after this closure captured them, so commit imperatively.
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
      Object.assign(pendingPatch, contextPinPatch(committedMaxSeqLength, pinsContextLength));
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

    // The peer's focused input commits into the shared draft during this click.
    const liveDraftConfig = readModelConfigDraft(draftKey)?.config;
    const baseConfig = liveDraftConfig
      ? reconcileConfigGpuSelection(liveDraftConfig, resolvedIsDiffusion, gpuDevices)
      : config;
    const peerChanged = !perModelConfigsEqual(baseConfig, config);
    const committedConfig =
      hasPending || peerChanged ? { ...baseConfig, ...pendingPatch } : baseConfig;
    const effectiveConfig = resolvedIsDiffusion
      ? withoutUnsupportedDiffusionSettings(committedConfig, gpuIndexKind)
      : target.isGguf
        ? committedConfig
        : {
            ...committedConfig,
            reasoningBudget: -1,
            reasoningBudgetMessage: "",
          };
    // Recompute from effectiveConfig: the same-click GPU Layers draft was not yet committed.
    const effectivePinFixedLayerContext =
      target.isGguf &&
      effectiveConfig.gpuMemoryMode === "manual" &&
      effectiveConfig.gpuLayers != null &&
      effectiveConfig.gpuLayers >= 0 &&
      effectiveConfig.customContextLength == null &&
      activeLoadedContext != null;
    const effectiveRuntimeConfig = (hasPending || peerChanged)
      ? effectivePinFixedLayerContext
        ? { ...effectiveConfig, customContextLength: activeLoadedContext }
        : effectiveConfig
      : runtimeConfig;
    const effectiveMaxSeqLengthValue =
      committedMaxSeqLength == null && !peerChanged
        ? maxSeqLengthValue
        : (normalizeMaxSeqLength(effectiveConfig.maxSeqLength) ??
          clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength));
    return { effectiveConfig, effectiveRuntimeConfig, effectiveMaxSeqLengthValue };
  };

  const persistConfig = (next: PerModelConfig) => {
    // savePerModelConfig normalizes first, so judge the normalized config.
    const normalized = normalizePerModelConfig(next);
    const evicted: { modelId: string; ggufVariant: string | null }[] = [];
    const saved = remember
      ? savePerModelConfig(configId, target.ggufVariant, normalized, evicted)
      : deletePerModelConfig(configId, target.ggufVariant);
    // Skipped when the localStorage write failed; gated on auto-switch reach, not GGUF-ness.
    if (saved && (target.apiLoadable ?? target.isGguf) && !nativePathToken) {
      syncModelOverride(
        configId,
        target.ggufVariant,
        remember ? normalized : null,
        remember
          ? {
              resetReasoningBudget:
                baseline.reasoningBudget !== -1 &&
                normalized.reasoningBudget === -1,
              resetReasoningBudgetMessage:
                baseline.reasoningBudgetMessage !== "" &&
                normalized.reasoningBudgetMessage === "",
            }
          : undefined,
      );
    }
    // Only once the write landed, or the next read replaces on-screen values with the old row.
    if (saved) {
      clearModelConfigDraftEdited(draftKey);
    }
    // Evicted models' server entries would keep applying; clear their mirrored fields.
    for (const dropped of evicted) {
      syncModelOverride(dropped.modelId, dropped.ggufVariant, null, {
        keepLaunchFlags: true,
      });
    }
    return { saved, defaultConfig: isDefaultConfig(normalized) };
  };

  const finishPersist = (defaultConfig: boolean) => {
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
  };

  const handleSave = () => {
    if (sharedVariantUnresolved) {
      return;
    }
    const { effectiveRuntimeConfig } = commitDraft();
    const { saved, defaultConfig } = persistConfig(effectiveRuntimeConfig);
    if (!saved) {
      toast.error("Couldn't save settings for this model.");
      return;
    }
    // setConfig, not update: this mirrors the save, so the draft must stay unedited.
    if (
      remember &&
      effectiveRuntimeConfig.customContextLength !== config.customContextLength
    ) {
      setConfig((current) => ({
        ...current,
        customContextLength: effectiveRuntimeConfig.customContextLength,
      }));
    }
    finishPersist(defaultConfig);
  };

  const handleRun = () => {
    if (sharedVariantUnresolved) {
      return;
    }
    if (budgetSettling) {
      return;
    }
    const { effectiveConfig, effectiveRuntimeConfig, effectiveMaxSeqLengthValue } =
      commitDraft();
    const effectivePersistenceOnly =
      isActiveModel &&
      perModelConfigsEqual(effectiveConfig, baseline, {
        followGlobal: true,
      }) &&
      rememberChanged;
    const { saved, defaultConfig } = persistConfig(effectiveRuntimeConfig);
    if (effectivePersistenceOnly) {
      if (!saved) {
        toast.error("Couldn't save settings for this model.");
        return;
      }
      finishPersist(defaultConfig);
      return;
    }
    if (!saved) {
      toast.error("Couldn't save these settings, loading with them anyway.");
    }
    const effectiveLoadConfig =
      target.isGguf || pinsContextLength
        ? effectiveRuntimeConfig
        : { ...effectiveRuntimeConfig, maxSeqLength: effectiveMaxSeqLengthValue };
    // The budget row flushes on unmount, after onRun staged the load, so save it first. Locked
    // before the settle so no drag lands in between.
    setVramBudgetLocked(true);
    const stagedBudget = settleVramBudgetSave();
    if (stagedBudget) {
      // Stays busy during the round trip, or a second click runs onRun twice.
      setBudgetSettling(true);
      // Caught, not voided: finally alone re-rejects as an unhandled rejection.
      void stagedBudget
        .catch((error: unknown) => {
            // Drop the retry, or the teardown flush would race this load's request.
          dropVramBudgetRetry();
          toast.error(
            error instanceof Error ? error.message : "Failed to save VRAM budget",
          );
        })
        .finally(() => {
          setVramBudgetLocked(false);
          setBudgetSettling(false);
          onRun(effectiveLoadConfig, classifiedIsDiffusion);
        });
      return;
    }
    setVramBudgetLocked(false);
    onRun(effectiveLoadConfig, classifiedIsDiffusion);
  };

  return (
    <div
      className="hint-on-hover flex flex-col"
      onChange={(event) => {
        if (isRunConfigEditorChange(event)) {
          handleSharedConfigEdit();
        }
      }}
    >
      {variant === "page" && showHeader && (
        // -ml-1.5 cancels the icon's inset so the chevron aligns with the rows below.
        <div className="flex items-center gap-2.5 pb-5">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              className="nav-icon-btn -ml-1.5 shrink-0 text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
              aria-label="Back to model list"
            >
              <ChevronLeftIcon
                className="size-4"
                strokeWidth={1.75}
              />
            </button>
          )}
          <div className="min-w-0 flex-1">
            <div className="text-ui-10 font-semibold uppercase leading-none tracking-wider text-muted-foreground">
              Run settings
            </div>
            <div className="mt-1.5 truncate text-ui-14 font-semibold leading-tight text-foreground">
              {target.displayName}
            </div>
          </div>
        </div>
      )}

      <SharedRunConfigReview
        target={target}
        imported={importedConfig}
        draftConfig={configState}
        currentConfig={config}
        remember={remember}
        hasSavedSettings={savedRemember}
      />
      <div className="space-y-5">
        {!target.isGguf && !targetIsMlx && !targetIsNpu && !classifiedIsDiffusion && !target.meta.isLora && !target.meta.audioType && (
          <InferenceEnginePicker parallelism={config.engineParallelism ?? "tensor"} onParallelismChange={engineParallelism => update({ engineParallelism })} precision={config.enginePrecision ?? "auto"} onPrecisionChange={enginePrecision => update({ enginePrecision })} value={config.engine ?? "auto"} onChange={engine => update({ engine })} onReadyChange={setEngineReady} onUse={handleRun} gpuIds={config.selectedGpuIds} onGpuChange={ids => update({ selectedGpuIds: ids, selectedGpuIndexKind: "physical" })} />
        )}
        {audioRuntimeGguf ? (
          <p className="text-ui-12 leading-snug text-muted-foreground">
            This model runs on the audio runtime, so llama.cpp settings do not
            apply. Its generation options are under Advanced on the Audio page
            once it is loaded.
          </p>
        ) : null}
        {memoryEstimateRequest != null && !audioRuntimeGguf && (
          <MemoryEstimateRow
            estimate={memoryEstimate.estimate}
            loading={memoryEstimate.loading}
            stale={memoryEstimate.stale}
            gpuCapacityGb={memoryGpuCapacityGb}
            totalCapacityGb={memoryTotalCapacityGb}
            systemRamCapacityGb={inferenceGpu.systemRamTotalGb}
            freeGpuCapacityGb={memoryFreeGpuCapacityGb}
            freeGpuCapacityKnown={memoryFreeGpuCapacityKnown}
            freeGpuReserveDeficitGb={memoryFreeGpuReserveDeficitGb}
            usableSystemRamGb={memoryUsableSystemRamGb}
            usableSystemRamKnown={inferenceGpu.systemRamAvailableKnown}
            systemRamReserveDeficitGb={memorySystemRamReserveDeficitGb}
            isUnifiedMemory={isAppleUnifiedMemory}
            singleMemoryPool={singleMemoryPool}
            reclaimableTotalBytes={reclaimableCredit.totalBytes}
            reclaimableGpuBytes={reclaimableCredit.gpuBytes}
            expanded={memoryBreakdownOpen}
            onExpandedChange={setMemoryBreakdownOpen}
          />
        )}
        {target.isGguf && !audioRuntimeGguf && (
          <>
            <div className="space-y-2">
              <div className={ROW_CLASS}>
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className={LABEL_CLASS}>Context Length</span>
                  <InfoHint>
                    Drag all the way left for Auto, which picks a context that
                    fits while keeping GPU speed. Custom values request an exact
                    context; higher ones use more memory.
                    {contextIsAuto && activeLoadedContext != null
                      ? ` Auto currently selected ${activeLoadedContext.toLocaleString()} tokens.`
                      : ""}
                    {nativeContextLength != null
                      ? ` This model's native context is ${nativeContextLength.toLocaleString()} tokens.`
                      : ""}
                  </InfoHint>
                </div>
                <NumericValueInput
                  ref={contextInputRef}
                  value={contextInputValue}
                  min={minContext}
                  max={maxContext}
                  step={1}
                  onChange={setContextLength}
                  displayValue={contextIsAuto ? "Auto" : undefined}
                  ariaLabel="Context Length"
                  className={NUMBER_INPUT_CLASS}
                  fixedWidth={true}
                  size={8}
                />
              </div>
              <div className="space-y-1">
                {nativeContextLength != null ? (
                  <div className="space-y-1">
                    <Slider
                      min={0}
                      max={maxContext}
                      step={128}
                      value={[contextSliderValue]}
                      onValueChange={([v]) => setContextSliderValue(v)}
                      className="panel-slider"
                      aria-label="Context Length"
                      // Position 0 is Auto, not a zero-token context.
                      thumbValueText={(v) =>
                        v !== 0
                          ? `${v.toLocaleString()} tokens`
                          : activeLoadedContext != null
                            ? `Auto, currently ${contextInputValue.toLocaleString()} tokens`
                            : "Auto"
                      }
                    />
                    <div className="flex justify-between text-ui-10 text-muted-foreground">
                      <span>Auto</span>
                      <span>{maxContext.toLocaleString()}</span>
                    </div>
                  </div>
                ) : null}
                {!contextIsAuto &&
                  isActiveModel &&
                  loadedMaxContextLength != null &&
                  contextValue > loadedMaxContextLength && (
                    <p className="text-ui-11 text-amber-500">
                      {isAppleUnifiedMemory ? (
                        <>
                          Above Studio&apos;s free-memory estimate (
                          {loadedMaxContextLength.toLocaleString()} tokens). It
                          may still load, but macOS may have to compress or swap
                          other apps and generation may slow down.
                        </>
                      ) : (
                        <>
                          Exceeds estimated VRAM capacity (
                          {loadedMaxContextLength.toLocaleString()} tokens). The
                          model may use system RAM.
                        </>
                      )}
                    </p>
                  )}
              </div>
            </div>

            <AdvancedSettingsToggle
              checked={showAdvanced}
              onCheckedChange={toggleAdvanced}
            />

            {showAdvanced && (
              <GgufAdvancedSettings
                config={config}
                update={update}
                showDraftTokens={showDraftTokens}
                showSpecDraftCacheDtype={showSpecDraftCacheDtype}
                speculativeFallback={speculativeFallback}
                onEditTemplate={() => setTemplateOpen(true)}
                layerCount={stagedDims?.layerCount ?? null}
                moeLayerCount={stagedDims?.moeLayerCount ?? null}
                isDiffusion={resolvedIsDiffusion}
                gpuDevices={gpuDevices}
                gpuLayersInputRef={gpuLayersInputRef}
                moeLayersInputRef={moeLayersInputRef}
                draftKey={draftKey}
                onExtraArgsLoadableChange={setExtraArgsLoadable}
              />
            )}
          </>
        )}
        {!target.isGguf && (
          <>
            <MaxSeqLengthSetting
              value={maxSeqLengthValue}
              max={maxSeqLengthMax}
              inputMax={targetIsNpu ? maxSeqLengthMax : MAX_SEQ_LENGTH_MAX}
              inputRef={maxSeqLengthInputRef}
              isMlx={pinsContextLength}
              pinned={savedContextPin(config) != null}
              fittedToMemory={
                savedContextPin(config) == null && mlxFittedWindow != null
              }
              windowUnknown={
                savedContextPin(config) == null &&
                mlxServedWindow == null &&
                npuServedWindow == null
              }
              hint={
                targetIsNpu
                  ? `Tokens of context FastFlowLM loads the model with. Unset, it loads ${NPU_DEFAULT_CONTEXT_LENGTH.toLocaleString()}, or the model's limit if that is lower.`
                  : undefined
              }
              onChange={(value) =>
                update(contextPinPatch(value, pinsContextLength))
              }
            />
            {!targetIsNpu && (
              <>
                <AdvancedSettingsToggle
                  checked={showAdvanced}
                  onCheckedChange={toggleAdvanced}
                />
                {showAdvanced && (
                  <MlxAdvancedSettings
                    config={config}
                    update={update}
                    outcome={mlxKvQuantOutcome}
                    servedByMlx={servedByMlx}
                    int8PrefillAvailable={int8PrefillAvailable}
                    onInt8PrefillChange={(checked) =>
                      checked
                        ? setInt8PrefillConfirmOpen(true)
                        : update({ mlxInt8Prefill: false })
                    }
                    onEditTemplate={() => setTemplateOpen(true)}
                    templateOutcome={chatTemplateOutcome}
                  />
                )}
              </>
            )}
          </>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-2 border-t border-border pt-5">
        <div className="flex min-w-0 items-center gap-2">
          <Checkbox
            id={rememberId}
            checked={remember}
            onCheckedChange={(checked) => {
              // Not in setRemember, which the save path calls again; an unticked box is a pending Forget.
              markModelConfigDraftEdited(draftKey);
              setRemember(checked === true);
            }}
          />
          <label
            htmlFor={rememberId}
            className="cursor-pointer select-none truncate text-ui-13 text-foreground"
          >
            Remember for this model
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            className={FOOTER_BUTTON_CLASS}
            disabled={
              sharedVariantUnresolved ||
              ((config.engine ?? "auto") !== "auto" && !engineReady) ||
              stagedMetadataPending ||
              budgetSettling ||
              (!extraArgsLoadable && !sharedExtraArgsCleared) ||
              sharedExtraArgsRefused ||
              extraArgsHydrating ||
              (isActiveModel &&
                atBaseline &&
                !rememberChanged &&
                !budgetReloadRequired)
            }
            onClick={handleRun}
          >
            {primaryActionLabel}
          </Button>
          {!persistenceOnly && (remember || savedRemember) && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className={FOOTER_BUTTON_CLASS}
              // Same gates as Load; Forget stores nothing, so broken saved args must not lock it.
              disabled={
                sharedVariantUnresolved ||
                stagedMetadataPending ||
                budgetSettling ||
                (remember &&
                  ((!extraArgsLoadable && !sharedExtraArgsCleared) ||
                    sharedExtraArgsRefused ||
                    extraArgsHydrating))
              }
              onClick={handleSave}
            >
              {remember ? "Save settings" : "Forget settings"}
            </Button>
          )}
          <Button
            type="button"
            variant="outline"
            size="sm"
            className={`${FOOTER_BUTTON_CLASS} text-muted-foreground`}
            disabled={atDefault}
            onClick={() => {
              handleSharedConfigEdit();
              // Reset writes through setConfig, not update, so it marks the draft itself.
              markModelConfigDraftEdited(draftKey);
              // Token equality alone would revive the discarded text later.
              clearExtraArgsEditForDraft(draftKey);
              setConfig({
                // null, not absent: absent makes the load inherit the running process's arguments.
                ...DEFAULT_PER_MODEL_CONFIG,
                llamaExtraArgs: null,
              });
            }}
          >
            Reset
          </Button>
          {target.isGguf && (
            <SharedRunConfigControls
              className={FOOTER_BUTTON_CLASS}
              target={target}
              config={config}
              ready={!extraArgsHydrating}
              canImport={variant !== "sidebar"}
              isDiffusion={resolvedIsDiffusion}
              disabled={
                sharedExtraArgsRefused ||
                (!extraArgsLoadable && !sharedExtraArgsCleared)
              }
              onImport={handleSharedConfigImport}
            />
          )}
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
      <AlertDialog
        open={int8PrefillConfirmOpen}
        onOpenChange={setInt8PrefillConfirmOpen}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Turn on Int8 Prefill?</AlertDialogTitle>
            <AlertDialogDescription>
              Long prompts are read faster, but the model&apos;s answers will
              change and may be less accurate. Some models are affected more
              than others. You can turn it off at any time.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
              onClick={() => {
                update({ mlxInt8Prefill: true });
                setInt8PrefillConfirmOpen(false);
              }}
            >
              Turn on
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
