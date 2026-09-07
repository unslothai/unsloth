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
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  resolveStagedDiffusionClassification,
  useChatRuntimeStore,
} from "@/features/chat";
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
  aggregateUsableFreeVramGb,
  resolveMemoryCapacityGb,
} from "@/hooks/gpu-vram";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  type Ref,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
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
import { resolveEstimateContext } from "../model-config/estimate-context";
import {
  type MemoryFitVerdict,
  formatMemoryGb,
  glueNoteItems,
  resolveDraftCacheNote,
  resolveKvNote,
  resolveMemoryFit,
} from "../model-config/memory-fit";
import { useMemoryEstimate } from "../hooks/use-memory-estimate";
import {
  fetchLoadModelOverride,
  fromApiOverride,
  modelOverrideKey,
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
  KV_CACHE_DTYPES,
  LOAD_MODES,
  LOAD_MODE_DEFAULT,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MAX_SEQ_LENGTH_STEP,
  MLX_KV_BITS,
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
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
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
  "min-w-0 truncate text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg";
const LABEL_CLASS_WRAP =
  "min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg";
const CONTROL_SURFACE =
  "rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1]";
const SELECT_TRIGGER_CLASS = `grid h-8 min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-ui-13! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0`;
const NUMBER_INPUT_CLASS = `h-8 w-[92px] ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-right text-ui-13 font-medium text-nav-fg outline-none focus-visible:ring-0`;

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

// Lower-case where the value is the flag's own spelling, so the pick reads the
// same as the launch command.
const LOAD_MODE_LABELS: Record<(typeof LOAD_MODES)[number], string> = {
  auto: "Auto",
  none: "None",
  mmap: "mmap",
  mlock: "mlock",
  "mmap+mlock": "mmap+mlock",
  dio: "DirectIO",
};

// What "Don't reserve system RAM" vetoes: mmap maps and dio streams, so neither
// holds a full host copy. Mirrors _LOAD_MODE_MLOCK_VALUES |
// _LOAD_MODE_RESERVING_VALUES in llama_server_args.py.
const RAM_RESERVING_LOAD_MODES = new Set(["none", "mlock", "mmap+mlock"]);

/**
 * What Model Memory does to this load mode, or null when the pick reaches the
 * command line untouched: a row showing the pick alone would name a flag the
 * child never sees. Keep resident is first because it wins outright.
 */
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
    return "This will be removed and the load runs the default mmap path: Don't reserve system RAM, in Settings, owns how the weights are held.";
  }
  return null;
}

/**
 * The batch size llama-server will not go below for this load.
 *
 * Two is its own hard floor (it aborts on a batch of 1 at any slot count), and above
 * that the floor follows the slots the launch SERVES. That is not always the number
 * chosen here: a build without --kv-unified serves one slot however many are asked
 * for, and load_model clamps to it, so sizing the floor from an explicit Slots value
 * refused "--batch-size 2" against a command the backend accepts. With the field
 * blank the server-wide default applies, which the catalogue already publishes
 * effective. A backend that publishes neither leaves the hard floor in charge.
 */
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
    config.nBatch != null ||
    config.nUbatch != null ||
    config.loadMode != null ||
    config.ctxCheckpoints != null ||
    config.cacheRam != null ||
    config.tensorParallel ||
    config.disableVision ||
    config.chatTemplateOverride != null ||
    // Hidden flags can change what the model does, so a panel that opens collapsed
    // over them says "defaults" about a load that is anything but.
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
    tensorParallel: false,
    disableVision: false,
    // the diffusion runner ignores the llama-server batch flags
    nBatch: null,
    nUbatch: null,
    // The diffusion shim never appends llama-server flags to its command, but the
    // load records them as though it had, so a box filled before classification
    // flipped would leave the model running without what it says.
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
  isMlx,
  pinned,
  windowUnknown,
}: {
  value: number;
  max: number;
  inputMax: number;
  onChange: (value: number) => void;
  inputRef?: Ref<NumericValueInputHandle>;
  isMlx?: boolean;
  pinned?: boolean;
  windowUnknown?: boolean;
}) {
  // MLX sizes itself when unpinned, so the control is the GGUF path's Context Length and
  // shows the length that will be served, not "Auto". A dash only while it is unknown.
  const label = isMlx ? "Context Length" : "Max Seq Length";
  return (
    <div className="space-y-3">
      <div className={ROW_CLASS}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={LABEL_CLASS}>{label}</span>
          <InfoHint>
            {isMlx
              ? "Tokens of context the model is sized for. Whether it also caps the " +
                "cache depends on the architecture."
              : "Maximum context window size in tokens. Applies when the model loads."}
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
          size={8}
        />
      </div>
      <Slider
        min={MAX_SEQ_LENGTH_MIN}
        max={max}
        step={MAX_SEQ_LENGTH_STEP}
        // Outside the control's range it sits at the nearer edge, or the first nudge
        // would step from the shown number onto the bound.
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
          step={step}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          className={NUMBER_INPUT_CLASS}
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

// How much of each card a load may claim. SERVER-WIDE, not a per-model override:
// it replaces two constants the fit reads, so there is nothing per-model to attach
// it to, and the label says so. Driven in whole percent so a dragged value
// round-trips exactly; the fraction is rebuilt at the API boundary.
function VramBudgetRow() {
  // The caller already gates on a discrete GPU and non-Manual mode. macOS is gated
  // here too: it reports no discrete GPUs, so the Metal path sizes itself from
  // _APPLE_UNIFIED_MEMORY_FRACTION and fits with budget_frac = 1.0. Showing the
  // slider would promise "applies on the next load" for a setting it ignores.
  const isMac = usePlatformStore((s) => s.deviceType === "mac");
  const [settings, setSettings] = useState<VramBudgetSettings | null>(null);
  const [percent, setPercent] = useState<number | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const adviceId = useId();
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const wasModelLoading = useRef(false);
  // Closed while a Run is settling this budget: an edit made in that window would
  // be flushed by the teardown alongside the load request.
  const [locked, setLocked] = useState(isVramBudgetLocked);
  useEffect(() => subscribeVramBudgetLock(setLocked), []);

  // Adopt what the last save or read published: the row's own GET races the PUT a
  // previous unmount fired, so reopening Advanced straight after a drag could show
  // a value the server no longer holds. A queued edit outranks the publish, or a
  // save landing mid-drag would pull the slider out from under the pointer.
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
      // Nothing to show, so do not even ask; the row is hidden below.
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

  // reloadRequired describes the running child, so it goes stale the moment a load
  // finishes. In the sidebar editor nothing remounts this row on a reload, so the
  // notice below would keep telling the user to reload a model already sized with
  // this budget. Only the falling edge: the read during a load answers about the
  // child being replaced.
  useEffect(() => {
    const finished = wasModelLoading.current && !modelLoading;
    wasModelLoading.current = modelLoading;
    if (isMac || !finished) {
      return;
    }
    let cancelled = false;
    // Forced: a read that began before the load finished describes the child being
    // replaced, and sharing it would republish the notice this refresh is clearing.
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

  // Flush, don't drop: the timer is cleared so nothing fires against a torn-down
  // view, but the fraction is still sent. It is held nowhere else, so a drag
  // followed within the debounce by Run, the Advanced toggle or closing the panel
  // would be lost. The view is gone, so the response only reaches subscribers.
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

  // A null response means an older backend; hide rather than render a dead control.
  if (isMac || !settings || percent === null) {
    return null;
  }

  const defaultPercent = vramFractionToPercent(settings.defaultFraction);
  // Stored beats UNSLOTH_VRAM_FRACTION, so once the slider has been touched there
  // is otherwise no way back to inheriting it: dragging to the same number stores
  // that number, and a later change to the variable stays masked. Clearing is what
  // the null the API already accepts is for.
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
    // Debounced: the slider fires per pointer move, so a drag would be dozens of
    // writes, each invalidating the read cache.
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
    }
    stageVramBudgetSave(vramPercentToFraction(next));
    saveTimer.current = setTimeout(() => {
      saveTimer.current = null;
      flushVramBudgetSave()
        ?.then(setSettings)
        .catch((error: unknown) => {
          // The client re-stages it, where the write generation can say whether
          // it is still the newest intent.
          toast.error(
            error instanceof Error ? error.message : "Failed to save VRAM budget",
          );
        });
    }, 400);
  };

  return (
    <div className="space-y-2">
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
          className="text-ui-11 text-muted-foreground underline underline-offset-2 hover:text-nav-fg"
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

// GPU Memory placement controls (mode / GPU Layers / MoE offload / GPU picker), GGUF only.
// Slider ceilings come from the GGUF header dims; --tensor-split is not persisted per model.
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
  // Ceiling = layer count + 1 (llama.cpp counts the output layer as offloadable), else a fallback.
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
      {/* A fixed manual layer count is the one placement the budget cannot move,
          and a CPU-only host has no GPU to act on, so "applies on the next load"
          would be false. Manual + Auto is included: it leaves the planner no device
          list, but --fit-target still carries the budget to llama.cpp's fitter. */}
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
                    ? ` · ${Math.round(d.memoryTotalGb)} GiB`
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

const MEMORY_VALUE_TONE: Record<MemoryFitVerdict, string> = {
  fits: "text-nav-fg",
  tight: "text-amber-500",
  exceeds: "text-red-500",
  unknown: "text-nav-fg",
};

/** One "GPU 29.41 GB" pill: dim caption, figure on the shared control surface. */
function MemoryFigure({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="flex shrink-0 items-center gap-1.5">
      <span className="text-ui-11 font-medium leading-none tracking-nav text-muted-foreground">
        {label}
      </span>
      <span
        className={`inline-flex h-6 items-center ${CONTROL_SURFACE} px-2 text-ui-12 font-medium tabular-nums ${tone ?? "text-nav-fg"}`}
      >
        {value}
      </span>
    </div>
  );
}

function MemoryBreakdownLine({
  label,
  value,
  note,
  muted,
}: {
  label: string;
  value: string;
  note?: string;
  muted?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="min-w-0 text-ui-11 leading-relaxed text-muted-foreground">
        {label}
        {note ? (
          <span className="ml-1 text-muted-foreground/70">{glueNoteItems(note)}</span>
        ) : null}
      </span>
      <span
        className={`shrink-0 text-ui-11 tabular-nums ${muted ? "text-muted-foreground" : "text-nav-fg"}`}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * "Estimated Memory Usage": what the settings above would cost, before they are run.
 *
 * Figures come from the loader's own sizing, not a weights-times-a-constant rule of
 * thumb, which stops working the moment the KV cache is no longer a rounding error.
 * So KV gets its own line, and when the header cannot size it the row quotes a floor
 * instead of a confident total.
 */
function MemoryEstimateRow({
  estimate,
  loading,
  stale,
  gpuCapacityGb,
  totalCapacityGb,
  systemRamCapacityGb,
  freeGpuCapacityGb,
  usableSystemRamGb,
  isUnifiedMemory,
  singleMemoryPool,
  expanded,
  onExpandedChange,
}: {
  estimate: MemoryEstimate | null;
  loading: boolean;
  stale: boolean;
  /** VRAM available, or the shared pool where there is only one. 0 when unknown. */
  gpuCapacityGb: number;
  /** GPU plus host RAM, the ceiling an offloaded load works against. 0 when unknown. */
  totalCapacityGb: number;
  /** Host RAM alone. The bytes a load pins OUTSIDE the GPU have to fit in this, and
   *  unused VRAM cannot help them, so it is a separate question from the total. */
  systemRamCapacityGb: number;
  /** VRAM free on the usable cards right now. Warns only: see the note at the call
   *  site for why this may not refuse a load. 0 when nothing was probed. */
  freeGpuCapacityGb: number;
  /** Host RAM the machine can hand out right now, less the reserve the loader keeps.
   *  Warns only, for the same reason the free-VRAM figure does. 0 when unknown. */
  usableSystemRamGb: number;
  isUnifiedMemory: boolean;
  /** GPU and host draw on the same memory, so an offloaded byte is not a freed one. */
  singleMemoryPool: boolean;
  expanded: boolean;
  onExpandedChange: (next: boolean) => void;
}) {
  const contentId = useId();
  if (!estimate?.available) {
    // Nothing honest to show. Silent while loading too, so the row does not flicker.
    return null;
  }
  // Every verdict and the one advisory paragraph, resolved in ../model-config/memory-fit.
  // Kept out of this file so the node test runner can reach it: the chain is long, the
  // cases it distinguishes are all hardware shapes nobody has on the desk, and while it
  // lived here an arm that could never be taken shipped unnoticed.
  const { gpuFit, totalFit, prefix, advisory } = resolveMemoryFit(estimate, {
    gpuCapacityGb,
    totalCapacityGb,
    systemRamCapacityGb,
    freeGpuCapacityGb,
    usableSystemRamGb,
    singleMemoryPool,
  });
  const kvNote = resolveKvNote(estimate);
  const draftCacheNote = resolveDraftCacheNote(
    estimate.drafterRuntimeGpuBytes,
    estimate.drafterRuntimeBytes,
  );
  return (
    <div className="space-y-2">
      {/* Wraps, unlike the other rows, because this one is the only header carrying a
          title AND two figures. The panel is w-[min(468px,...)], so under a ~460px
          window it shrinks with the viewport, and the figures do not shrink: the
          title absorbed the whole shortfall and truncated to "E..." at 320px. Letting
          the figures drop to their own line costs a line only where they would not
          have fitted anyway, and is identical above that width. */}
      <div className={`${ROW_CLASS} flex-wrap gap-y-1`}>
        <button
          type="button"
          onClick={() => onExpandedChange(!expanded)}
          aria-expanded={expanded}
          aria-controls={contentId}
          className="flex min-w-0 items-center gap-1.5 rounded-sm text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <span className={LABEL_CLASS}>Estimated Memory Usage</span>
          <span className="shrink-0 rounded-full bg-black/[0.06] px-1.5 py-px text-ui-10 font-medium uppercase leading-[1.4] tracking-wider text-muted-foreground dark:bg-white/[0.08]">
            Beta
          </span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={`size-3 shrink-0 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`}
            strokeWidth={1.75}
          />
        </button>
        {/* ml-auto so that on the wrapped line, where justify-between has nothing to
            push against, the figures still sit under the right edge they had. */}
        <div
          className={`ml-auto flex shrink-0 items-center gap-3 transition-opacity ${stale || loading ? "opacity-50" : ""}`}
        >
          {/* One pool: offloading a layer moves it within the same memory rather
              than out of it, so the honest single figure is the total. Reporting
              the GPU share here let a zero-layer load read as almost free. */}
          <MemoryFigure
            label={singleMemoryPool ? (isUnifiedMemory ? "Unified" : "Shared") : "GPU"}
            value={`${prefix}${formatMemoryGb(
              singleMemoryPool ? estimate.totalBytes : estimate.gpuBytes,
            )}`}
            tone={MEMORY_VALUE_TONE[singleMemoryPool ? totalFit : gpuFit]}
          />
          {singleMemoryPool ? null : (
            <MemoryFigure
              label="Total"
              value={`${prefix}${formatMemoryGb(estimate.totalBytes)}`}
              tone={MEMORY_VALUE_TONE[totalFit]}
            />
          )}
        </div>
      </div>
      {expanded && (
        <div id={contentId} className="space-y-1 pl-0.5">
          <MemoryBreakdownLine
            label="Weights"
            value={formatMemoryGb(estimate.weightsBytes)}
            note={
              estimate.gpuLayers != null && estimate.layerCount != null
                ? `${estimate.gpuLayers} of ${estimate.layerCount + 1} layers on GPU`
                : undefined
            }
          />
          <MemoryBreakdownLine
            label="KV cache"
            value={
              estimate.kvEstimable ? formatMemoryGb(estimate.kvBytes) : "unknown"
            }
            note={estimate.kvEstimable ? kvNote : undefined}
            muted={!estimate.kvEstimable}
          />
          <MemoryBreakdownLine
            label="Compute buffers"
            value={formatMemoryGb(estimate.computeBytes)}
          />
          {/* The encoder's buffers, which run about 1.3x the projector file. Only on a
              vision load, and named separately since the file is already in Weights. */}
          {estimate.projectorRuntimeBytes > 0 && (
            <MemoryBreakdownLine
              label="Vision encoder"
              value={formatMemoryGb(estimate.projectorRuntimeBytes)}
            />
          )}
          {/* Only when speculation loads a separate drafter, and it is the term most
              likely to surprise: its cache grows with context like the target's. */}
          {estimate.drafterRuntimeBytes > 0 && (
            <MemoryBreakdownLine
              label="Draft cache"
              value={formatMemoryGb(estimate.drafterRuntimeBytes)}
              note={draftCacheNote}
            />
          )}
        </div>
      )}
      {advisory && (
        <p
          className={`text-ui-11 leading-relaxed ${advisory.tone === "warn" ? "text-amber-500" : "text-muted-foreground"}`}
        >
          {advisory.text}
        </p>
      )}
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
        <p className="text-ui-11 leading-snug text-muted-foreground">
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
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {templateOutcome}
        </p>
      ) : null}
    </div>
  );
}

/**
 * Mmap/Mlock, with the note that says when Settings wins. Its own component
 * because it subscribes to the Model Memory settings and must keep following
 * them: the settings page is reachable without unmounting this panel.
 */
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
            default: Unsloth picks None when it can prove the model fits without
            paging, since a mapped read is slower, and otherwise leaves the
            choice to llama.cpp, which memory-maps unless a device cannot. mmap
            forces the mapping, mlock keeps the model in RAM rather than letting
            it swap or compress, mmap+mlock does both, DirectIO streams the file
            where the platform supports it, and None asks for no special mode.
            Model Memory, in Settings, owns this when either of its toggles is
            on.
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
            className={`w-[124px] shrink-0 ${SELECT_TRIGGER_CLASS}`}
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
  /** Which stored entries the extra-arguments row reads, most specific first. */
  onExtraArgsLoadableChange: (loadable: boolean) => void;
}) {
  const batchAdviceId = useId();
  const ubatchAdviceId = useId();
  // llama-server aborts below 2 (GGML_ASSERT(n_tokens_all <= cparams.n_batch)) and below
  // the slot count (GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)), so the loader
  // raises the emitted value to max(slots, 2). Surfaced so the number typed here is not
  // silently different from the one that runs. With Slots blank the count is the server
  // default this page cannot see, so only the hard floor of 2 is asserted.
  const batchFloor = Math.max(2, config.nParallel ?? 2);
  const batchBelowFloor = config.nBatch != null && config.nBatch < batchFloor;
  // llama.cpp runs at min(batch, ubatch) and the /status echo is the REQUESTED size, so
  // the control would otherwise keep showing a value the server never used (batch 8 /
  // ubatch 4096 measured 8.8x slower). Against the EMITTED batch, or the two advisories
  // contradict: batch 4 / slots 8 launches at 8, so saying it runs at 4 is wrong. A blank
  // batch emits no flag and runs llama.cpp's own 2048, which caps the micro-batch just the
  // same, so it is the default and not "unbounded". Extras or LLAMA_ARG_BATCH can move
  // that, but this page cannot see them, the same limit the slot floor above carries.
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
            Faster generation. Auto picks the best strategy for the model and
            platform: DSpark or DFlash when the model ships a drafter sidecar,
            otherwise MTP / ngram. Pick a strategy to force it, or Off to
            disable. DSpark downloads a sidecar of about 11 GB and DFlash one of
            about 1.5 GB, both trading VRAM for speed; on quantized targets
            their greedy output can differ from a non speculative run. MTP and
            ngram do not change output.
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
              // Dropped with the drafter, like the depth above: the draft
              // context exists only while a separate model is loaded.
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

      {showDraftTokens && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Draft Tokens</span>
            <InfoHint>
              Max draft tokens per step. Leave blank for the default (MTP and
              DFlash: 2 on GPU, 3 on CPU/Mac; DSpark: 3).
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
              KV cache precision for the draft model's own context
              (--spec-draft-type-k / --spec-draft-type-v). Separate from the KV
              Cache Dtype above, which is the target model's. f16 is the
              default; bf16 and f32 are full precision; q8_0 through iq4_nl are
              quantized, and a quantized draft cache saves VRAM on a drafter
              whose output is verified by the target anyway.
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
              className={`w-[92px] shrink-0 ${SELECT_TRIGGER_CLASS}`}
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

      {!isDiffusion && (
        <div className="space-y-1">
          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Batch Size</span>
              <InfoHint>
                Logical prompt batch size (--batch-size). Leave blank for the
                llama.cpp default (2048). Rarely needs changing; the micro-batch
                below is what usually matters.
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
            <p id={batchAdviceId} className="text-ui-12 text-muted-foreground">
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
                Physical prompt micro-batch size (--ubatch-size). Leave blank for
                the llama.cpp default (512). Larger values speed up prompt
                processing but use more VRAM for the compute buffer; capped at the
                batch size.
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
            <p id={ubatchAdviceId} className="text-ui-12 text-muted-foreground">
              Micro-batch is larger than the batch size, so llama.cpp will run at{" "}
              {effectiveBatch}. Raise the batch size to use {config.nUbatch}.
            </p>
          )}
        </div>
      )}

      {/* withoutUnsupportedDiffusionSettings forces tensorParallel back to false and
          the diffusion runner never reads it, so the switch flipped back under the
          pointer. Same gate as the Vision and batch rows around it. */}
      {!isDiffusion && (
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
      )}

      {/* withoutUnsupportedDiffusionSettings forces disableVision back to false on a
          diffusion model and the diffusion runner never reads it, so the switch would
          flip back under the pointer. Gated for the same reason the batch rows are. */}
      {!isDiffusion && (
        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Vision</span>
            <InfoHint>
              Loads the vision projector so the model can read images. Turning it
              off frees the VRAM the projector would use, which can leave room for
              more layers on the GPU. Text generation is unaffected either way.
              Models that ship no projector have nothing to load, so the setting
              does nothing for them.
            </InfoHint>
          </div>
          <Switch
            className="panel-switch shrink-0"
            checked={!config.disableVision}
            onCheckedChange={(checked) => update({ disableVision: !checked })}
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

      {/* Host-side knobs, after the ones that shape the model itself. All three
          are llama-server's own, and the diffusion runner launches no
          llama-server, so they are hidden for it like the batch sizes above. */}
      {!isDiffusion && (
        <>
          <LoadModeRow config={config} update={update} />

          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Checkpoints</span>
              <InfoHint>
                Context checkpoints kept per slot (--ctx-checkpoints), which let
                a sliding-window model rewind instead of re-processing the
                prompt. Leave blank for the llama.cpp default (
                {CTX_CHECKPOINTS_LLAMA_DEFAULT}); 0 disables them. Each one costs
                host memory, and models without a sliding window ignore the
                setting.
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
                Host memory in MiB llama-server may spend caching prompt state it
                has evicted from a slot (--cache-ram), so a returning
                conversation is not re-processed. Leave blank for the llama.cpp
                default ({CACHE_RAM_LLAMA_DEFAULT}); 0 disables the cache and -1
                lifts the limit.
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

      {/* Last, because it is the escape hatch for everything the rows above do not
          cover, and because llama.cpp's last-wins parsing means these really are
          appended after them. GGUF only: the flags are llama-server's. */}
      {!isDiffusion && (
        <ExtraArgsRow
          config={config}
          update={update}
          onLoadableChange={onExtraArgsLoadableChange}
        />
      )}
    </>
  );
}

/**
 * Pass-through llama-server arguments for this model.
 *
 * llama-server documents 283 flags and Unsloth already emits or manages about 115
 * of them, so the long tail is a text box rather than 168 more controls. The
 * boundary is `validate_extra_args` on the backend, which refuses the flags Unsloth
 * owns; this row is the same judgement shown early, plus a check against the flags
 * THIS build documents, which a list shipped with Unsloth could not do.
 */
function ExtraArgsRow({
  config,
  update,
  onLoadableChange,
}: {
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
  onLoadableChange: (loadable: boolean) => void;
}) {
  const [catalog, setCatalog] = useState<LlamaFlagCatalog | null>(null);
  // What is typed, which is not what is stored: the stored value is argv tokens, so
  // the text only exists here. Seeded from the config, then owned by the box, or
  // every re-render would re-quote what the user is halfway through typing.
  const [text, setText] = useState(() =>
    formatExtraArgs(config.llamaExtraArgs),
  );
  const adviceId = useId();
  // What the box last put INTO the config, so an external change can be told apart
  // from the echo of the user's own typing. Reset and the parent's hydration both
  // replace llamaExtraArgs while this row is mounted, and a box that kept its old
  // text would then disagree with what Load sends. Re-seeding on every config
  // change instead would re-quote a half-typed line on each keystroke.
  const selfWritten = useRef(formatExtraArgs(config.llamaExtraArgs));
  const external = formatExtraArgs(config.llamaExtraArgs);
  useEffect(() => {
    if (external === selfWritten.current) {
      return;
    }
    selfWritten.current = external;
    setText(external);
  }, [external]);

  // Re-read on invalidation, not only on mount: updating llama.cpp from the banner
  // replaces the binary while this panel stays open, and a row holding the old
  // build's catalogue would go on judging arity, and calling flags unknown, against
  // help text that no longer describes the server it is about to launch.
  const [catalogEpoch, setCatalogEpoch] = useState(0);
  useEffect(
    () => subscribeLlamaFlagCatalog(() => setCatalogEpoch((epoch) => epoch + 1)),
    [],
  );
  useEffect(() => {
    let cancelled = false;
    loadLlamaFlagCatalog().then((loaded) => {
      if (!cancelled) {
        // Null is "cannot verify": adopted as well, or the row would keep checking
        // against the previous binary after an update it cannot read.
        setCatalog(loaded);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [catalogEpoch]);

  // Model Memory removes some of these before the launch: apply_model_memory_policy
  // emits its own load mode and strips the rest, so an --mlock typed here would be
  // shown, saved, and never passed. Read once and then kept in sync, like the section
  // in Settings that owns it. A failed read leaves the row quiet rather than claiming
  // a removal it cannot confirm.
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
    // The same floor the batch control shows, for the same reason: with Slots blank
    // the count is the server default this page cannot see, so only the hard 2 holds.
    // A build that clamps serves one slot whatever is chosen here, so an explicit
    // Slots value must not raise the floor above a batch the backend would accept.
    batchFloor: effectiveBatchFloor(config.nParallel, catalog),
    keepResident: modelMemory?.keepResident ?? false,
    noRamReserve: modelMemory?.noRamReserve ?? false,
  });
  const tokenCount = parseExtraArgs(text).tokens.length;

  // The panel owns the Load button, and an error here is one validate_extra_args
  // would refuse. Reported up rather than only painted red, so the load is not
  // started just to come back as a failure.
  const loadable = extraArgsAreLoadable(diagnostics);
  useEffect(() => {
    onLoadableChange(loadable);
    // Deliberately no cleanup. Collapsing Advanced settings unmounts this row while
    // the tokens stay in the config and still go out with the load, so withdrawing
    // the objection there would re-enable a button for a request the backend
    // refuses. The panel clears it when the model changes instead.
  }, [loadable, onLoadableChange]);

  const commit = (next: string) => {
    setText(next);
    const { tokens } = parseExtraArgs(next);
    // Recorded before the update, so the config change this causes reads as the
    // box's own and does not bounce back through the effect above.
    selfWritten.current = formatExtraArgs(tokens.length > 0 ? tokens : null);
    // null, not [], so the panel's own "no flags" reads the same as the stored one;
    // toApiOverride turns it into the explicit [] that clears the server's copy.
    update({ llamaExtraArgs: tokens.length > 0 ? tokens : null });
  };

  return (
    <div className="space-y-2">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={LABEL_CLASS}>Extra Arguments</span>
        <InfoHint>
          <div className="flex flex-col gap-1.5">
            <div>
              Passed straight to llama-server for this model, after the settings
              above, so anything set in both is taken from here.
            </div>
            <div>
              Quote a value containing spaces or backslashes, including a
              Windows path. Nothing runs a shell, so $HOME, ; and | are ordinary
              characters. Flags Unsloth owns, like the model, the port and the
              API key, are refused.
            </div>
          </div>
        </InfoHint>
      </div>
      <div className="panel-text-surface h-20 w-full overflow-hidden corner-squircle">
        <textarea
          value={text}
          onChange={(event) => commit(event.target.value)}
          spellCheck={false}
          placeholder="--rope-scaling yarn --yarn-orig-ctx 32768"
          aria-label="Extra llama-server arguments"
          aria-describedby={diagnostics.length > 0 ? adviceId : undefined}
          className="block size-full resize-none bg-transparent px-3.5 py-2.5 text-left font-mono text-ui-12 leading-relaxed text-nav-fg outline-none placeholder:text-muted-foreground"
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
  * Page variant only: render the built-in "Run settings" title block. A host that already
  * shows the model name as its page heading turns this off. */
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
  // Apple Silicon specifically, and used ONLY for wording: "Unified" is what
  // Apple calls its pool, where an AMD APU's is a "Shared" one. It is NOT the
  // right signal for the capacity question below -- see hasUnifiedMemory.
  //
  // Unified memory, not just Darwin: an Intel Mac spills to system RAM like a PC.
  const isAppleUnifiedMemory = usePlatformStore((s) => s.appleSilicon);
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
    (s) => s.maxContextLength,
  );
  // What settings are stored under, which is not always what loads; the probes keep target.id.
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
  const [configState, setConfig] = useState<PerModelConfig>(() =>
    reconcileConfigGpuSelection(initial.config, isDiffusion, gpuDevices),
  );
  // The live config, for the async reads below: an effect that closed over it would
  // hold whatever it was when the request started.
  const configRef = useRef(configState);
  configRef.current = configState;
  const [remember, setRemember] = useState(() => initial.remembered);
  const [savedRemember, setSavedRemember] = useState(() => initial.remembered);
  const rememberRef = useRef(remember);
  rememberRef.current = remember;
  const [speculativeFallback] = useState(readPersistedSpeculativeType);
  // Same substitution as speculativeFallback, for the same reason: only "manual" is
  // persisted per model, so an absent mode means "follow the standing preference"
  // rather than Auto. applyPerModelConfigToRuntime resolves it that way at load, so
  // pricing the absence as Auto rated a Manual launch against the wrong plan.
  const [gpuMemoryModeFallback] = useState(readPersistedGpuMemoryMode);
  const [templateOpen, setTemplateOpen] = useState(false);
  // Raised by the extra-arguments row when what is typed is something
  // validate_extra_args would refuse, so the load is not started to fail. Held by
  // the panel rather than the row because the row unmounts whenever Advanced
  // settings are collapsed, while its tokens stay in the config.
  const [extraArgsLoadable, setExtraArgsLoadable] = useState(true);
  // True until the stored-arguments read below settles. A load started before it
  // lands sends no llama_extra_args, and /load cannot inherit them from a process
  // that is not running, so a fast click would launch a cold model without the
  // arguments that were about to appear on screen.
  const [extraArgsHydrating, setExtraArgsHydrating] = useState(
    () => target.isGguf && !isDiffusion,
  );
  // The row does not withdraw its own objection when it unmounts, or collapsing
  // Advanced settings would re-enable Load for arguments the backend refuses. A
  // different model is the one thing that really does retire it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the model, not on the setter
  useEffect(() => {
    setExtraArgsLoadable(true);
    setExtraArgsHydrating(target.isGguf && !isDiffusion);
  }, [configId, target.ggufVariant, target.isGguf, isDiffusion]);

  // Compare against what the backend was asked for, not what it applied: staging a
  // new value must retire a verdict that answered a different request.
  const chatTemplateOutcome =
    isActiveModel &&
    (configState.chatTemplateOverride ?? null) ===
      (loadedChatTemplateOverride ?? null)
      ? chatTemplateOverrideReason
      : null;
  const mlxKvQuantOutcome =
    isActiveModel &&
    (configState.mlxKvBits ?? null) === (loadedMlxKvBitsRequested ?? null)
      ? // Both, not either: dropping the note promises savings before the offset
        // where quantization actually starts.
        [mlxKvQuantReason, mlxKvQuantNote].filter(Boolean).join(". ") || null
      : null;
  const servedByMlx = isServedByMlx(
    target.isGguf,
    platformDeviceType,
    platformChatOnlyReason,
  );
  // Read live, not snapshotted at mount: the sidebar copy stays mounted while collapsed.
  const advancedPreference = useSyncExternalStore(
    subscribeAdvancedSettingsOpen,
    readAdvancedSettingsOpen,
    () => null,
  );
  // Until the switch is used anywhere, a model carrying non-default advanced values opens the
  // section itself. Frozen at mount so editing a field back to its default cannot close it.
  const [autoOpenAdvanced, setAutoOpenAdvanced] = useState(() =>
    hasNonDefaultAdvanced(configState),
  );
  // Frozen like the rest of the auto-open decision, so editing the width does not
  // reopen the section the user just closed.
  const [initialMlxKvBits] = useState(() => configState.mlxKvBits ?? null);
  // Applicability stays live, unlike the snapshot above: MLX can become available
  // after mount, and a width that starts applying then has to surface.
  const autoOpenForMlxKvBits = servedByMlx && initialMlxKvBits != null;
  const showAdvanced =
    advancedPreference ?? (autoOpenAdvanced || autoOpenForMlxKvBits);
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

  // Fetch GGUF header dims to size the GPU Memory sliders; the context also fills in below.
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
  // Tri-state on purpose: an inconclusive probe stays undefined so onRun hands "unknown" on.
  // Collapsing it to false would let a compare pane inherit another model's split (#7574).
  const classifiedIsDiffusion = resolveStagedDiffusionClassification(
    isDiffusion,
    stagedDims,
  );
  const resolvedIsDiffusion = classifiedIsDiffusion === true;

  // The one field on this page whose stored value the local config may never have
  // seen. Everything else here is written by this panel, but llama_extra_args can be
  // set through the overrides API with no UI involved, which is exactly why that
  // route preserves it when omitted. Showing an empty box for a model that has flags
  // would read as "none", and the first edit would then submit a list that silently
  // dropped them. Fetched, not guessed.
  //
  // The row judges what is typed while it is mounted, and deliberately leaves its
  // verdict standing when Advanced settings collapse, because the tokens still go
  // out with the load. But a verdict reached before the catalogue arrived was
  // reached without knowing which flags this build documents, and collapsing the
  // section freezes it: a bare --threads typed during a cold probe would keep Load
  // enabled and fail at llama-server startup. So while the row is unmounted, the
  // panel re-judges what the config holds once the catalogue lands.
  // The row holds its own subscription, but it is unmounted whenever this check is
  // the one running, so the panel needs its own: an in-app llama.cpp update with
  // Advanced collapsed would otherwise leave a verdict reached against the previous
  // binary standing, and Load live for a flag the new one has removed.
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
      // The row leaves its objection standing when it unmounts, because the tokens
      // it objected to still go out with the load. Once there are none, there is
      // nothing left to object to: Reset with Advanced collapsed used to leave Load
      // disabled over arguments the request no longer carries.
      setExtraArgsLoadable(true);
      return;
    }
    let cancelled = false;
    loadLlamaFlagCatalog().then((catalog) => {
      if (cancelled || !catalog) {
        // Null is "cannot verify": leaving the standing verdict alone is the same
        // benefit of the doubt the row gives an unprobed build.
        return;
      }
      const loadable = extraArgsAreLoadable(
        diagnoseExtraArgs(formatExtraArgs(args), catalog, {
          gpuSelectionActive: configState.selectedGpuIds != null,
          manualGpuMemory: configState.gpuMemoryMode === "manual",
          // The server-wide default when the Slots field is blank: that is the
          // count the launch will serve, and llama-server aborts on a batch below
          // it. Clamped builds serve one whatever is chosen.
          batchFloor: effectiveBatchFloor(configState.nParallel, catalog),
        }),
      );
      // Only ever tightens. This judges the TOKENS, and formatExtraArgs quotes them
      // back into a balanced string, so an unclosed quote the row objected to reads
      // as fine here; raising the verdict would re-enable Load over the unfinished
      // value the user is still typing. A list that is genuinely clean keeps
      // whatever verdict already stands, which is true unless the row set it.
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

  // The server copy is shared by Desktop, LAN, and tunnel origins. Hydrate the
  // whole remembered GGUF config here; localStorage is only the immediate seed.
  // This also has to run while Advanced is closed because extra arguments live in
  // that section but affect every load.
  const extraArgsHydrated = useRef<string | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the model is the identity
  useEffect(() => {
    if (!target.isGguf || resolvedIsDiffusion) {
      // Nothing else even has this field: the row is GGUF-only and so is the load
      // payload, so a Transformers or MLX model must not wait on either request.
      // A diffusion GGUF is GGUF-shaped but runs through the diffusion shim, which
      // appends no llama-server flags and whose config strips them, so it must not
      // wait either.
      setExtraArgsHydrating(false);
      return;
    }
    // The order the auto-switch loader tries, and for the same reason: the settings
    // UI keys a local row by the path it loads from, while configId is a derived
    // alias, so reading the alias first lets an older entry stay the one API loads
    // apply while the panel shows none of it. A loose .gguf was keyed by its
    // filename label in an early build, which is the fourth candidate here.
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
    // Joined because an array literal is a new value on every render.
    const identity = keys.join("\u0000");
    if (extraArgsHydrated.current === identity) {
      setExtraArgsHydrating(false);
      return;
    }
    let cancelled = false;
    // What the config held BEFORE the request went out. Anything else at the end of
    // it was typed by the user while it was in flight, and sanitizing that would
    // rewrite live input: typing --agent during the window cleared the box instead
    // of showing the error, and a long paste could be trimmed behind the cursor.
    const configAtStart = configRef.current;
    const storedAtStart = resolveInitialConfig(configId, target.ggufVariant);
    const rememberAtStart = rememberRef.current;
    const localAtStart = configAtStart.llamaExtraArgs;
    // The denylist, not the catalogue: sanitizing a stored list needs only the flags
    // Unsloth refuses, and that route answers without running `llama-server --help`.
    // Waiting on the probe instead would hold Load shut for as long as a cold --help
    // takes, and releasing on a deadline would leave a legacy flag in an explicit
    // request that /load then refuses.
    // A last-resort release, so a request that never settles cannot disable Load for
    // good: past this a load behaves as it did before the feature.
    const release = setTimeout(() => setExtraArgsHydrating(false), 15000);
    Promise.all([
      // Resolved by the backend, which owns the rules: the local resolver stays as
      // the fallback for a backend that predates the parameter.
      fetchLoadModelOverride(loadId, configId, target.ggufVariant, keys),
      loadManagedLlamaFlags(),
    ])
      .then(([resolvedOverride, managed]) => {
        // Marked here rather than before the request: StrictMode replays the effect
        // (setup, cleanup, setup), so a key marked up front would leave the first
        // fetch cancelled and the second setup returning early, and the box would
        // never fill in development.
        if (cancelled) {
          return;
        }
        extraArgsHydrated.current = identity;
        const resolvedArgs = {
          tokens: resolvedOverride?.llama_extra_args ?? [],
          explicit: Array.isArray(resolvedOverride?.llama_extra_args),
        };
        // Through the resolver, not a literal lookup: the backend folds identities
        // and reads whole entries in its own order before it reads a field.
        //
        // Sanitised before it becomes a request: hydrating turns a stored list into
        // an EXPLICIT one, which /load validates strictly instead of putting it
        // through the carry-over paths that drop a newly denied flag quietly.
        // Without this, an install upgraded across a denylist change stops loading a
        // model that worked the day before.
        const stored = sanitizeStoredExtraArgs(
          resolvedArgs.tokens,
          managed?.managed ?? new Set<string>(),
          // The host's bounds, not the constants: a Windows server takes 24 KiB and
          // holds a quoted-command budget as well, so trimming to 32 KiB here sends a
          // list /load answers 400 on.
          {
            maxBytes: managed?.maxBytes,
            windowsCommandBudget: managed?.windowsCommandBudget,
          },
        );
        // A list this build refuses can equally have come from local storage, saved
        // by a build that still allowed it. The server copy is sanitised above; this
        // is the same treatment for the one already in hand, which nothing else
        // would catch while Advanced stays collapsed.
        const local = configRef.current.llamaExtraArgs;
        // Kept for the merge below as well as for the box: a row that carries no
        // arguments leaves the local list standing, and handing back the list this
        // build refuses would re-enable Load for a request /load answers 400 on.
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
        // The row as it should be read: its own fields, with the arguments it carries
        // sanitized against what this build refuses.
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
        // What hydration would leave in the config: the row's list when it carries
        // one, this browser's when it does not, since a row that says nothing about
        // arguments cannot overrule the list already here. Judged as a whole, or a
        // local flag the expanded row has ALREADY refused is declared loadable by a
        // verdict read off the empty server list, and the row republishes its own
        // only when its verdict changes, so the objection never comes back.
        const hydratedArgs = serverConfig?.llamaExtraArgs ?? stored;
        const hydratedIsLoadable =
          hydratedArgs.length === 0
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

        // The shared row outranks the local seed for every field it carries, so LAN
        // and Desktop cannot show different remembered values; fromApiOverride keeps
        // the rest of this browser's config rather than resetting it, since an
        // absent field is as much a gap in the mirror as a chosen default. Never
        // over an edit made while the request was in flight.
        if (
          resolvedRow &&
          serverConfig &&
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
          setConfig(serverConfig);
          setRemember(true);
          setSavedRemember(true);
          if (hasNonDefaultAdvanced(serverConfig)) {
            setAutoOpenAdvanced(true);
          }
          // Unconditionally, because savePerModelConfig says "no settings" by
          // DELETING the entry: a merge that comes out default is a clear that has
          // to travel, not a write to skip. Clearing a model's only remembered
          // flag leaves the row as an explicit empty list, which is exactly that
          // case, and skipping it stranded the old flag here for model-selector's
          // quick select to reload without ever opening this panel. Writing when
          // no record exists is already a no-op.
          const rememberedConfig = fromApiOverride(
            resolvedRow,
            storedConfig.config,
          );
          // Same budget as any other write, so the same clean-up: eviction is silent
          // and still reports success, and a dropped model would keep applying its
          // server row to API loads while the picker showed defaults, with nothing
          // able to forget it. Not a Forget, only the mirrored fields go.
          const hydrationEvicted: {
            modelId: string;
            ggufVariant: string | null;
          }[] = [];
          // Checked for the same reason the Save path checks it: the write can fail
          // outright (storage full or unavailable, or a record from a newer build that
          // must not be replaced) and say so in the return rather than throwing. Marked
          // saved regardless, the panel would claim the server settings are remembered
          // here while quick select and background loads, which read resolveInitialConfig
          // and never open this panel, still saw the stale record or none. Leaving it
          // unsaved makes it a pending change instead, so Save is reachable and reports
          // the failure the way every other write does.
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
          // An EXPLICIT empty row is a decision, not an absence: the settings page
          // writes one when the box is cleared for a quant whose bare-repository
          // row still carries arguments, and it is what stops the server's lookup
          // there. Left undefined the panel omits the field on Load, and /load
          // carries the resident model's arguments over: the very ones just
          // cleared. Only when nothing was typed here in the meantime.
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
        // Read from a ref rather than inside the updater below: an updater must stay
        // free of side effects (StrictMode calls it twice), and this decides one.
        if (configRef.current.llamaExtraArgs !== undefined) {
          // Typed into while this was in flight. The row is judging that text and its
          // verdict is the live one, so this answer about the stored list must not
          // overwrite either of them: doing so re-enabled Load for invalid input.
          return;
        }
        setExtraArgsLoadable(hydratedIsLoadable);
        setConfig((current) =>
          // Guarded again, because the ref is only as fresh as the last render.
          current.llamaExtraArgs === undefined
            ? { ...current, llamaExtraArgs: stored }
            : current,
        );
        // The frozen snapshot above was taken before this answer arrived, so without
        // this a model whose arguments live only on the server opens looking like
        // every default. An explicit preference still wins: this only feeds the
        // fallback.
        setAutoOpenAdvanced(true);
      })
      .catch(() => {
        // Nothing to say: the panel is still usable, and the load would report a
        // real problem with the overrides service far more clearly than this could.
      })
      .finally(() => {
        // Including the failure: an overrides service that is down must not leave
        // Load disabled for good.
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
  ]);
  const config = reconcileConfigGpuSelection(
    configState,
    resolvedIsDiffusion,
    gpuDevices,
  );
  // Held for the window where Run is waiting on a budget PUT rather than on the
  // load it looks like it started.
  const [budgetSettling, setBudgetSettling] = useState(false);
  // The budget is on no per-model field, so changing it leaves this page at its
  // baseline and the Reload button disabled, with the row asking for a reload the
  // user could not start. Read off the same publish the row uses, so nothing is
  // threaded through GpuMemorySettings and a row that never mounts leaves it false.
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
  const update = (patch: Partial<PerModelConfig>) =>
    setConfig((current) => ({
      ...reconcileConfigGpuSelection(current, resolvedIsDiffusion, gpuDevices),
      ...patch,
    }));

  // True for every mode that takes a draft depth, DSpark included, so this
  // gates the Draft Tokens row rather than naming a drafter.
  const showDraftTokens =
    config.speculativeType != null &&
    DRAFT_N_MAX_SPEC_TYPES.has(config.speculativeType);
  // Narrower than the row above: the dtype needs a second context, and only the
  // sidecar modes always load one. MTP may read baked-in heads out of the target
  // GGUF, which the loader knows and this panel does not.
  const showSpecDraftCacheDtype =
    config.speculativeType != null &&
    SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(config.speculativeType);
  const nativeContextLength =
    target.meta.contextLength ?? stagedDims?.contextLength ?? null;
  const activeLoadedContext =
    isActiveModel && target.isGguf ? loadedContextLength : null;
  // resolveLoadMaxSeqLength returns 0 for a builtin-default GGUF load before it looks
  // at the resident context, so the estimate must not fall back to it either.
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
  const atBaseline = perModelConfigsEqual(config, baseline);
  // The fitted value is an outcome, not an override. Auto stays at the default even
  // when a loaded model reports less than its native context. A non-GGUF pin is an
  // override too, read from whichever field it was saved in.
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
    floorMaxSeqLength(modelMaxPosition.maxPositionEmbeddings) ??
    MAX_SEQ_LENGTH_MAX;
  // The pin, else the length a self-sizing backend would serve, so the control states a
  // context rather than declining to. Only an unread window falls back to the app default,
  // and Reset clears the pin to null so no fallback may rebuild one from a runtime value.
  // Reported as it stands, not snapped to the request step: 2,056 would read 2,048.
  const servedWindow = (value: unknown) =>
    typeof value === "number" && Number.isFinite(value) && value > 0
      ? Math.floor(value)
      : null;
  const mlxNativeWindow = targetIsMlx
    ? servedWindow(modelMaxPosition.maxPositionEmbeddings)
    : null;
  // What a load would serve. The backend clamps an auto-sized window to the request
  // ceiling, so a wider native one (Scout declares 10,485,760) would name a length no
  // load can serve. Native above stays raw: metadata, not a promise.
  const mlxProspectiveWindow =
    mlxNativeWindow == null
      ? null
      : Math.min(mlxNativeWindow, MAX_SEQ_LENGTH_MAX);
  const mlxServedWindow =
    (targetIsMlx && isActiveModel ? servedWindow(loadedContextLength) : null) ??
    mlxProspectiveWindow;
  const maxSeqLengthValue =
    servedWindow(savedContextPin(config)) ??
    mlxServedWindow ??
    clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength);
  // The slider picks a request, so it stops at the widest a load may make.
  const maxSeqLengthMax = Math.min(
    MAX_SEQ_LENGTH_MAX,
    Math.max(nativeMaxSeqLength, maxSeqLengthValue),
  );
  // An auto-fit-below-native GGUF shows activeLoadedContext while
  // customContextLength stays null. If the user fixes GPU Layers (Manual) and
  // remembers, pin that shown context so a later fresh load keeps the fitted
  // placement instead of sending native/0 for fixed layers and recreating the OOM.
  const loadableConfig = resolvedIsDiffusion
    ? withoutUnsupportedDiffusionSettings(config, gpuIndexKind)
    : config;
  // A model classified as diffusion after the box was typed into keeps the row's
  // objection while the arguments themselves are stripped from what loads, and the
  // row does not withdraw it when it unmounts. Retired here, or Load stays disabled
  // over arguments the request no longer carries.
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
  // Kept as-is so isDefaultConfig clears a remembered override rather than pinning the
  // app default; the write rule already settled which field holds the pin.
  const runtimeConfig = target.isGguf
    ? pinFixedLayerContext
      ? { ...loadableConfig, customContextLength: activeLoadedContext }
      : loadableConfig
    : loadableConfig;
  const runtimeGpuMemoryMode =
    runtimeConfig.gpuMemoryMode ?? gpuMemoryModeFallback;
  // Priced against the config that would actually load, at the context on screen, so
  // the figures answer for what the Load button will do. Diffusion stands down: its
  // runner allocates on a different plan than llama-server's.
  //
  // The tri-state is read as a tri-state here, not through resolvedIsDiffusion, which
  // folds "not yet classified" in with "not diffusion". A GGUF whose classification is
  // still in flight may be DiffusionGemma, and starting the llama-server estimate on
  // that guess paints a footprint from the wrong allocation plan -- one that stays on
  // screen indefinitely if the classifying probe never answers, since the tri-state
  // then never leaves undefined. Waiting costs a moment of no row; guessing costs a
  // confident wrong number, which is the trade this whole panel is written around.
  const memoryEstimateRequest =
    target.isGguf && classifiedIsDiffusion === false
      ? {
          modelPath: target.id,
          ggufVariant: target.ggufVariant ?? null,
          hfToken: hfToken || null,
          nativePathToken,
          // The context the Load button sends, not the one the control displays: an
          // unset length with no header yet shows 32,768 and loads at native.
          nCtx: resolveEstimateContext(
            runtimeConfig.customContextLength ?? null,
            activeLoadedContext,
            // The two shapes where resolveLoadMaxSeqLength answers 0 before it
            // reaches the resident context: --fit owns the sizing, or a
            // builtin-default preset on a GGUF load.
            (target.isGguf === true &&
              runtimeGpuMemoryMode === "manual" &&
              (runtimeConfig.gpuLayers ?? GPU_LAYERS_AUTO) < 0) ||
              (target.isGguf === true && activePresetSource === "builtin-default"),
          ),
          cacheTypeKv: runtimeConfig.kvCacheDtype,
          nParallel: runtimeConfig.nParallel,
          nBatch: runtimeConfig.nBatch,
          nUbatch: runtimeConfig.nUbatch,
          ctxCheckpoints: runtimeConfig.ctxCheckpoints ?? null,
          // The same substitution applyPerModelConfigToRuntime makes at load: a model
          // with no per-model override sends null, which the backend reads as Auto,
          // while the selector has been showing the global fallback all along. With
          // the global set to Off and an 11 GB DSpark sidecar in the repo, the row
          // charged the sidecar for a load that disables it.
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
  const [memoryBreakdownOpen, setMemoryBreakdownOpen] = useState(false);
  const inferenceGpu = useInferenceGpuInfo();
  // A pin can only draw on the cards it names, so the verdict is measured against
  // those. Judging a one-card pin against a two-card total called an 8 GB load a fit
  // on 16 GB of VRAM it could not reach.
  const pinnedGpuIds = runtimeConfig.selectedGpuIds;
  // Whether the devices THIS LOAD will use draw on one pool with the rest of the
  // system, from the backend's per-device unified_memory flag rather than from
  // the platform. Apple Silicon is the obvious case and a ROCm APU is the one
  // that was being missed: both share the pool, so adding VRAM to system RAM to
  // reach a machine-wide ceiling counts the same bytes twice.
  //
  // Scoped to the devices this load will actually use, and true only when EVERY
  // one of them is unified.
  //
  // Two separate mistakes were made here, so both are written down. Reading the
  // host-wide flag was the first: an APU beside a discrete card makes it true,
  // and a pin on the discrete card then collapsed totalCapacityGb from 143.5 GiB
  // to 15.5 GiB. Narrowing to the pin but keeping `.some()` was the second: an
  // unpinned load, or a pin naming BOTH, still marked the whole set unified and
  // reported 62.1 GiB instead of 143.5 GiB. Both produce false "more than this
  // machine holds" warnings for loads that fit comfortably.
  //
  // `.every()` is the right question for CAPACITY: one independent-memory device
  // in the set means there is real VRAM beside system RAM, so the two are not
  // one pool. Note use-gpu-info.ts deliberately uses `.some()` for its own flag,
  // and that is not an inconsistency -- it answers a different question, whether
  // the aggregate is still a VRAM ceiling a verdict can be measured against, and
  // one unified part is enough to spoil that.
  //
  // The empty set is excluded explicitly, because `[].every()` is true and a
  // host with no devices at all is not a unified-memory machine.
  //
  // The Apple fallback stays host-wide and unconditional: there every device is
  // the one pool whatever is pinned, and it also covers the window before the
  // per-device probe lands, so this is never a weaker answer than the platform
  // check it replaced.
  const hasUnifiedMemory = useMemo(() => {
    if (isAppleUnifiedMemory) return true;
    const governing =
      pinnedGpuIds && pinnedGpuIds.length > 0
        ? gpuDevices.filter((device) => pinnedGpuIds.includes(device.index))
        : gpuDevices;
    if (governing.length === 0) return false;
    return governing.every((device) => device.unifiedMemory === true);
  }, [gpuDevices, pinnedGpuIds, isAppleUnifiedMemory]);
  // The VRAM Budget slider sits in this same panel and caps what the next load may
  // claim per GPU, so the verdict has to be measured against the capped figure or the
  // row contradicts the control directly above it. Subscribed as well as read once:
  // dragging that slider must re-classify without a remount.
  // Seeded with the loader's own default rather than a full card. The read is async
  // and returns null on failure -- an older backend has no such route -- and in both
  // windows the launch still applies VRAM_FRACTION_DEFAULT. Starting at 1 claimed a
  // reserve no setting of that slider ever gives back, so the row read a hair
  // optimistic until the answer arrived and stayed there if it never did.
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
  // What is free RIGHT NOW on the cards this load may use: _select_gpus admits on
  // free-minus-reserve, not on totals, so another process holding VRAM changes what
  // happens to this load.
  //
  // WARNS, never refuses. The bytes a pending load reclaims are mostly the resident
  // model's own, which Studio unloads first, and cannot be attributed per device from
  // here, so raw free memory would call reloading the loaded model impossible. Capping
  // the verdict at "tight" is honest either way.
  //
  // A fixed Manual placement is launched verbatim -- load_model empties its probed GPU
  // set and emits --gpu-layers N --fit off -- so the server-wide VRAM Budget never
  // reaches the planner. Discounting capacity by it called a 16 GiB fixed placement on
  // a 24 GiB card over budget at 50%. Auto is still budgeted: Auto is the mode that
  // hands the decision to the planner.
  const memoryBudgetGovernsLaunch =
    runtimeGpuMemoryMode !== "manual" ||
    (runtimeConfig.gpuLayers ?? GPU_LAYERS_AUTO) < 0;
  const memoryEffectiveBudgetFraction = memoryBudgetGovernsLaunch
    ? memoryVramBudgetFraction
    : 1;
  // _HOST_RAM_HEADROOM_MIB: the 2 GiB the loader keeps for the rest of the system
  // before admitting an offloaded load.
  //
  // The HOST reading, not the sum-shaped one beside it, which is zeroed whenever
  // ANY device shares system RAM -- every dGPU + iGPU box -- so that was
  // permanently 0 there and the host-pressure advisory could not fire even under
  // a pin on the discrete card, the one case where host RAM really is a separate
  // pool.
  //
  // Hoisted because the free-capacity memo below needs the same figure: a ROCm
  // APU's free pool IS this, and two copies of the subtraction is how they would
  // drift apart.
  const memoryUsableSystemRamGb = Math.max(
    0,
    (inferenceGpu.systemRamAvailableHostGb || 0) - 2,
  );
  const memoryFreeGpuCapacityGb = useMemo(() => {
    const pinned =
      pinnedGpuIds && pinnedGpuIds.length > 0
        ? gpuDevices.filter((device) => pinnedGpuIds.includes(device.index))
        : gpuDevices;
    // Per device, and by the loader's absolute-reserve rule rather than a
    // multiplication: the budget is subtracted from the card, not applied to what
    // happens to be free, and the two agree only on an idle one.
    //
    // Aggregated with the same count-the-shared-pool-once rule the TOTALS use. A
    // plain sum over a mixed inventory counted the iGPU's share of host RAM a second
    // time, and this figure is what freeGpuFit -- and through it the fit verdict --
    // is measured against.
    // The reserve keeps its budget-derived floor even for a fixed Manual placement:
    // what is free right now still constrains the launch, whatever set the number.
    const freeVram = aggregateUsableFreeVramGb(
      pinned,
      memoryEffectiveBudgetFraction,
    );
    // On a ROCm APU this figure is the free space inside a BIOS-carved window,
    // and resolveMemoryFit asks it the WHOLE-LOAD question as soon as the pool is
    // single. Those two together warned that a 60 GiB load does not fit a 96 GiB
    // machine with 60+ GiB free, purely because it exceeds a 48 GiB window.
    //
    // The pool's real free memory is the host's, exactly as its real CAPACITY is
    // the host's RAM rather than the window. Same discriminator as the capacity
    // side, so the two cannot answer differently: Apple's GPU figure already IS
    // the pool and is left alone.
    if (hasUnifiedMemory && !isAppleUnifiedMemory) {
      return Math.max(freeVram, memoryUsableSystemRamGb);
    }
    return freeVram;
  }, [
    gpuDevices,
    pinnedGpuIds,
    memoryEffectiveBudgetFraction,
    hasUnifiedMemory,
    isAppleUnifiedMemory,
    memoryUsableSystemRamGb,
  ]);
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
      // The GENERAL signal, not the Apple-only one. A ROCm APU reports
      // unified_memory per device and shares one pool exactly as Apple does, but
      // reading appleSilicon here charged it as discrete VRAM PLUS host RAM --
      // the same bytes counted twice, so the panel could report a fit that
      // cannot happen. The Hub bar gets this right and abstains on this very
      // flag; this is the panel catching up.
      unifiedMemory: hasUnifiedMemory,
      // ...but "one pool" and "how big is the pool" are different questions, and
      // the two platforms answer the second differently. Apple's memory_total_gb
      // IS the machine's unified memory; a ROCm APU's is a BIOS-carved window
      // onto system RAM, so taking it as the ceiling reported 46.56 GiB on a
      // 96 GiB machine and warned that a 60 GiB load exceeds the host. Only the
      // Apple half may be read as the whole pool.
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

  const handleRun = () => {
    if (budgetSettling) {
      return;
    }
    // Same-click Load/Reload: a numeric draft the user just typed is flushed only by that input's
    // blur handler, which runs after this click closure captured the stale value. Commit every
    // numeric input imperatively so the staged load honors what was typed, not just Context.
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
      Object.assign(pendingPatch, contextPinPatch(committedMaxSeqLength, targetIsMlx));
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
    // pinFixedLayerContext above was computed from the render-time config, before the same-click
    // GPU Layers draft was committed. Recompute from effectiveConfig so a positive fixed-layer
    // value still pins the fitted context, else a later fresh load recreates the OOM.
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
    // Non-GGUF load substitutes the resolved max sequence length; recompute from the committed draft.
    const effectiveMaxSeqLengthValue =
      committedMaxSeqLength == null
        ? maxSeqLengthValue
        : (normalizeMaxSeqLength(effectiveConfig.maxSeqLength) ??
          clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength));
    // Recheck the committed draft so Save/Forget reloads when needed.
    const effectiveAtBaseline = perModelConfigsEqual(effectiveConfig, baseline);
    const effectivePersistenceOnly =
      isActiveModel && effectiveAtBaseline && rememberChanged;
    // Judge what storage keeps: savePerModelConfig normalizes first, so the raw object over-reports.
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
    // Mirror to the server so an API load gets these settings, not app defaults. Best-effort, and
    // skipped when the localStorage write failed or the two would permanently disagree. Gated on
    // auto-switch reach, not GGUF-ness: the resolver skips Ollama, and a native-path lease is the same.
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
    // Saving can push the local map over budget and drop other models, whose server entries would
    // keep applying with nothing able to forget them. Not a Forget: only the mirrored fields go.
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
    // MLX pins in customContextLength as GGUF does, so unpinned sends nothing.
    const effectiveLoadConfig = target.isGguf
      ? effectiveRuntimeConfig
      : targetIsMlx
        ? effectiveRuntimeConfig
        : { ...effectiveRuntimeConfig, maxSeqLength: effectiveMaxSeqLengthValue };
    // Same reason as the numeric commits above: the budget row flushes on unmount,
    // which for this click lands after onRun has staged the load, so that load (the
    // very one the control promises) would use the old fraction. A failed save must
    // not swallow the load, hence finally; nothing staged stays synchronous.
    // Closed before the settle, not after: the control has to stop taking edits at
    // the moment this click owns the fraction, or a drag landing in between is the
    // very race the lock exists to remove.
    setVramBudgetLocked(true);
    const stagedBudget = settleVramBudgetSave();
    if (stagedBudget) {
      // The click is answered by a network round trip now, so the button stays live
      // for as long as that takes. A second click in that window would settle the
      // same chain again and run onRun twice, i.e. two loads from one intent.
      setBudgetSettling(true);
      // Caught, not voided: finally alone re-rejects into an unhandled rejection,
      // and the load would proceed on the old fraction with nothing said.
      void stagedBudget
        .catch((error: unknown) => {
            // Dropped rather than left for the next flush: the picker teardown that
            // onRun triggers flushes it, and that PUT would then race the load
            // request this click is about to send. Either fraction could win, which
            // is the one thing this control promises cannot happen. Only the retry
            // goes, never a newer edit staged over it. The toast says the budget
            // did not change, and the slider is still there to retry.
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
    // Nothing to wait for, so the control never actually closes for the user.
    setVramBudgetLocked(false);
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
            <div className="mt-1.5 truncate text-ui-14 font-semibold leading-tight text-nav-fg">
              {target.displayName}
            </div>
          </div>
        </div>
      )}

      <div className="space-y-3.5">
        {target.isGguf && (
          <>
            {/* Above Context Length on purpose: that is the control moving this
                number most, and a readout below it is one you go looking for. */}
            <MemoryEstimateRow
              estimate={memoryEstimate.estimate}
              loading={memoryEstimate.loading}
              stale={memoryEstimate.stale}
              gpuCapacityGb={memoryGpuCapacityGb}
              totalCapacityGb={memoryTotalCapacityGb}
              systemRamCapacityGb={inferenceGpu.systemRamTotalGb}
              freeGpuCapacityGb={memoryFreeGpuCapacityGb}
              usableSystemRamGb={memoryUsableSystemRamGb}
              isUnifiedMemory={isAppleUnifiedMemory}
              singleMemoryPool={singleMemoryPool}
              expanded={memoryBreakdownOpen}
              onExpandedChange={setMemoryBreakdownOpen}
            />
            <div className="space-y-3">
              <div className={ROW_CLASS}>
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className={LABEL_CLASS}>Context Length</span>
                  <InfoHint>
                    Drag all the way left for Auto, which chooses a context that
                    fits while prioritizing GPU speed. Custom values request an
                    exact context; higher values use more memory and may move
                    model layers to system RAM.
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
                  size={8}
                />
              </div>
              {nativeContextLength != null ? (
                <div className="space-y-1.5">
                  <Slider
                    min={0}
                    max={maxContext}
                    step={128}
                    value={[contextSliderValue]}
                    onValueChange={([v]) => setContextSliderValue(v)}
                    className="panel-slider"
                    aria-label="Context Length"
                    // Position 0 is Auto, not a zero-token context, so
                    // aria-valuenow alone reads as a length no model has. The
                    // number is only spoken once one exists: before a load
                    // contextInputValue is the offload fallback used to seed the
                    // input, not a selection, and Auto may still fit native.
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
                        Exceeds what fits in unified memory (
                        {loadedMaxContextLength.toLocaleString()} tokens). The
                        GPU and the rest of the system share one pool here, so
                        there is nothing to offload to.
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
                onExtraArgsLoadableChange={setExtraArgsLoadable}
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
              isMlx={targetIsMlx}
              pinned={savedContextPin(config) != null}
              windowUnknown={
                savedContextPin(config) == null && mlxServedWindow == null
              }
              onChange={(value) => update(contextPinPatch(value, targetIsMlx))}
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
            onClick={() =>
              setConfig({
                // null, not the default's absent: absent omits the field, and the
                // load then INHERITS the running process's arguments, so a reload
                // after Reset kept the very flags the empty box says are gone.
                ...DEFAULT_PER_MODEL_CONFIG,
                llamaExtraArgs: null,
              })
            }
          >
            Reset
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-8"
            disabled={
              stagedMetadataPending ||
              budgetSettling ||
              !extraArgsLoadable ||
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
