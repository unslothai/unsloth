// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type DragEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  FolderAddIcon,
  Settings02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";

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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { TrainingSeriesPoint } from "@/features/training";
// eslint-disable-next-line no-restricted-imports -- matches images-page.tsx's token access
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { cn } from "@/lib/utils";
import {
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
} from "@/features/settings/api/upload-limit";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";

import {
  type DiffusionDatasetExample,
  type DiffusionTrainableFamily,
  type DiffusionTrainingInfo,
  type DiffusionTrainingRunDetail,
  type DiffusionTrainingRunSummary,
  type DiffusionTrainingStatus,
  datasetItemCount,
  getDiffusionTrainingInfo,
  getDiffusionTrainingRun,
  getDiffusionTrainingStatus,
  listDiffusionDatasetExamples,
  listDiffusionDatasetImages,
  listDiffusionTrainingRuns,
  startDiffusionTraining,
  stopDiffusionTraining,
  uploadDiffusionDataset,
} from "../api";
import { DatasetLabelingGrid, LabelingGridToggle } from "./dataset-labeling-grid";
import {
  DATASET_CLIP_EXTS,
  DATASET_FILE_ACCEPT,
  DATASET_IMAGE_EXTS,
  chunkDatasetUpload,
  existingStemClash,
  filesFromDataTransfer,
  metadataKeyedOnSubfolders,
  oversizedChunk,
  selectDatasetFiles,
} from "./dataset-files";
import { DatasetShowcase } from "./dataset-showcase";
import { DiffusionCharts } from "./diffusion-charts";
import {
  ExampleDatasetCards,
  runExampleImport,
  shortExampleLabel,
} from "./example-dataset-cards";
import {
  buildDiffusionResumePayload,
  resumeActionLabel,
} from "./resume-diffusion-run";
import {
  resolveDiffusionDeployBase,
  resolveDiffusionTrainingBase,
} from "./diffusion-train-deploy";
import { resolveDiffusionTrainingFacts } from "./diffusion-train-family-facts";
import { type LrScheduler, lrSchedulePreset } from "./diffusion-train-lr-schedule";

// Families the Train tab can train, in popularity order; a fallback for an older backend whose /info reports none.
type FamilyPreset = {
  name: string;
  label: string;
  base_repos: string[];
  defaults: {
    rank: number;
    lr: number;
    resolution: number;
    // The family's LR ramp: both or neither, since a warmup count is inert under "constant". Only
    // the backend reports them, so the static presets leave the pair unset.
    lrScheduler?: LrScheduler;
    lrWarmupSteps?: number;
  };
  vram_note: string;
  gated?: boolean;
  // The note's facts, one per chip. Absent on an older backend, which falls back to vram_note prose.
  params?: string;
  qlora_vram_gb?: number | null;
  note?: string;
  base_specs?: DiffusionTrainableFamily["base_specs"];
};

const FAMILY_PRESETS: FamilyPreset[] = [
  {
    name: "flux.1",
    label: "FLUX.1-dev (12B)",
    base_repos: ["black-forest-labs/FLUX.1-dev"],
    defaults: { rank: 16, lr: 0.0001, resolution: 512 },
    vram_note: "Gated: needs its license and your HF token.",
    gated: true,
    params: "12B",
    qlora_vram_gb: 16,
  },
  {
    name: "qwen-image",
    label: "Qwen-Image (20B)",
    base_repos: ["unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "Qwen/Qwen-Image"],
    defaults: { rank: 16, lr: 0.00005, resolution: 512 },
    vram_note: "The biggest: needs a large GPU. Start at 512px.",
    params: "20B",
    qlora_vram_gb: 24,
    note: "The heaviest option. Start at 512px.",
  },
  {
    name: "z-image",
    label: "Z-Image (6B)",
    base_repos: [
      "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
      "Tongyi-MAI/Z-Image-Turbo",
      "Tongyi-MAI/Z-Image",
    ],
    defaults: { rank: 16, lr: 0.0001, resolution: 768 },
    vram_note: "The smallest and fastest. A good first pick.",
    params: "6B",
    qlora_vram_gb: 12,
    note: "The smallest and fastest. A good first pick.",
  },
  {
    name: "sdxl",
    label: "SDXL (U-Net)",
    base_repos: ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo"],
    defaults: { rank: 16, lr: 0.0001, resolution: 1024 },
    vram_note: "The classic. Fine at 1024px.",
    qlora_vram_gb: 12,
    note: "The classic. Fine at 1024px.",
  },
];

const CUSTOM_BASE = "__custom__";
const UPLOAD_DATASET = "__upload__";
// Dense DiT base precisions: they load a dense (bf16) base and quantise it, so the backend
// rejects them for a bnb-4bit repo.
const DENSE_PRECISIONS = new Set(["bf16", "int8", "fp8", "mxfp8"]);
// Mirror the backend repo_is_prequantized heuristic: a bitsandbytes 4-bit repo cannot serve the dense base precisions.
function repoIsPrequantized(baseModel: string): boolean {
  const name = baseModel.toLowerCase();
  return (
    name.includes("bnb-4bit") ||
    name.includes("-4bit") ||
    name.includes("int4") ||
    name.includes("nf4")
  );
}
// Dataset-select option value prefix for a not-yet-imported example; picking it imports.
const EXAMPLE_PREFIX = "example:";

/** "12 images", "12 clips", or "12 items" for a mixed folder: a clip folder reading "0 images"
 *  is the bug this labels away. */
function datasetItemLabel(d: { image_count: number; clip_count?: number }): string {
  const clips = d.clip_count ?? 0;
  const total = d.image_count + clips;
  const noun = clips === 0 ? "image" : d.image_count === 0 ? "clip" : "item";
  return `${total} ${noun}${total === 1 ? "" : "s"}`;
}
// min-w-0 with a truncating value: a long option would otherwise set the grid column min width
// and push into its neighbour.
const selectClass =
  "h-8 w-full min-w-0 text-xs *:data-[slot=select-value]:min-w-0 *:data-[slot=select-value]:truncate";
// Every settings cell is a grid item, so it needs min-w-0 to shrink; grid-cols-1 carries that
// shrink to the contents, since a bare `grid` froze the implicit track at its widest child's
// min-content (150px) and painted over the next column.
const fieldClass = "grid grid-cols-1 min-w-0 gap-2";

/** opens the folder picker; icon-only because the 416px rail already holds the select and the file-pick button. */
function FolderPickButton({
  disabled,
  onPick,
}: {
  disabled: boolean;
  onPick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <Button
          type="button"
          size="icon"
          variant="outline"
          aria-label="Add a folder of images"
          className="size-8 shrink-0"
          onClick={onPick}
          disabled={disabled}
        >
          <HugeiconsIcon icon={FolderAddIcon} className="size-3.5" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Add a whole folder of images</TooltipContent>
    </Tooltip>
  );
}

/** A field's label with its guidance behind an "i" tooltip; only facts a user must act on stay on the page as text. */
function FieldLabel({
  hint,
  children,
}: {
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-center gap-1">
      {/* block, not Label's default flex: text-overflow does nothing on a flex container, so truncate
          cut the text mid-glyph. */}
      <Label className="block min-w-0 truncate text-xs">{children}</Label>
      {hint ? <InfoHint>{hint}</InfoHint> : null}
    </div>
  );
}

/** The family's training facts as chips: size, QLoRA VRAM floor, access. What a chip cannot carry
 *  stays as a line below. */
function FamilyFacts({ family, baseModel }: { family?: FamilyPreset; baseModel?: string }) {
  if (!family) return null;
  const facts = resolveDiffusionTrainingFacts(family, baseModel);
  const hasChips = Boolean(facts.params || facts.qlora_vram_gb || facts.gated);
  if (!hasChips) {
    return family.vram_note ? (
      <p className="text-ui-11 leading-snug text-muted-foreground">
        {family.vram_note}
      </p>
    ) : null;
  }
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {facts.params ? (
          <Badge variant="secondary" className="font-normal">
            {facts.params}
          </Badge>
        ) : null}
        {facts.qlora_vram_gb != null ? (
          <Badge variant="secondary" className="font-normal">
            QLoRA {facts.qlora_vram_gb}GB+ VRAM
          </Badge>
        ) : null}
        {/* Access, not a spec: a neutral fill sets it apart from the capability chips. */}
        {facts.gated ? (
          <Badge
            variant="secondary"
            className="bg-muted font-normal text-muted-foreground"
          >
            Gated
          </Badge>
        ) : null}
      </div>
      {(facts.gated || facts.note) && (
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {facts.gated ? "Needs its license and your HF token." : null}
          {facts.gated && facts.note ? " " : null}
          {facts.note}
        </p>
      )}
    </div>
  );
}

// Merge backend-reported families over the presets, keeping preset ordering and filling anything the backend omits.
function mergeFamilies(reported?: DiffusionTrainableFamily[]): FamilyPreset[] {
  if (!reported || reported.length === 0) return FAMILY_PRESETS;
  const byName = new Map(reported.map((f) => [f.name, f]));
  const merged: FamilyPreset[] = FAMILY_PRESETS.map((p) => {
    const r = byName.get(p.name);
    if (!r) return p;
    byName.delete(p.name);
    return {
      name: p.name,
      label: r.label || p.label,
      base_repos: r.base_repos?.length ? r.base_repos : p.base_repos,
      defaults: {
        rank: r.defaults?.lora_rank ?? p.defaults.rank,
        lr: r.defaults?.learning_rate ?? p.defaults.lr,
        resolution: r.defaults?.resolution ?? p.defaults.resolution,
        // No preset fallback for the ramp: a reported family owns it outright, so a backend that drops
        // its warmup preset drops the ramp too.
        ...lrSchedulePreset(r.defaults),
      },
      vram_note: r.vram_note || p.vram_note,
      gated: r.gated ?? p.gated,
      base_specs: r.base_specs,
      // The chips travel together: a backend reporting any of them owns the whole set, so a preset
      // value cannot sit beside a live one.
      ...(r.params != null || r.qlora_vram_gb != null || r.note != null
        ? {
            params: r.params ?? "",
            qlora_vram_gb: r.qlora_vram_gb ?? null,
            note: r.note ?? "",
          }
        : { params: p.params, qlora_vram_gb: p.qlora_vram_gb, note: p.note }),
    };
  });
  // Any backend family not in the presets goes last, so a newly added trainer still shows.
  for (const r of byName.values()) {
    merged.push({
      name: r.name,
      label: r.label || r.name,
      base_repos: r.base_repos ?? [],
      defaults: {
        rank: r.defaults?.lora_rank ?? 16,
        lr: r.defaults?.learning_rate ?? 0.0001,
        resolution: r.defaults?.resolution ?? 768,
        ...lrSchedulePreset(r.defaults),
      },
      vram_note: r.vram_note ?? "",
      gated: r.gated ?? false,
      params: r.params ?? "",
      qlora_vram_gb: r.qlora_vram_gb ?? null,
      note: r.note ?? "",
      base_specs: r.base_specs,
    });
  }
  return merged;
}

// Full-page training workspace: left = configure, right = live run. Kept mounted with the page
// so a long run survives tab switches; polling is gated on `active`.
export function DiffusionTrainPanel({
  active,
  loadedFamily,
  loadedBaseRepo,
  onTrainingComplete,
  onDeploy,
  familyName,
  onFamilyNameChange,
  baseChoice,
  onBaseChoiceChange,
  onFamiliesChange,
}: {
  active: boolean;
  // Loaded generation model family / base repo, to preselect a matching training base when it is one we can train.
  loadedFamily?: string | null;
  loadedBaseRepo?: string | null;
  // Family + base are controlled by the page: the top bar picks the training base while Train is showing.
  familyName: string;
  onFamilyNameChange: (name: string) => void;
  baseChoice: string;
  onBaseChoiceChange: (repo: string) => void;
  // /info owns the family list, so publish it for the top bar's picker.
  onFamiliesChange?: (families: FamilyPreset[]) => void;
  // Bump the page's LoRA discovery so a freshly trained adapter appears in the picker.
  onTrainingComplete?: () => void;
  // Deploy a finished adapter into Create mode: load the base then preselect the adapter.
  onDeploy?: (args: {
    baseRepo: string;
    family: string;
    catalogPath: string;
    trigger: string;
  }) => void;
}) {
  const [info, setInfo] = useState<DiffusionTrainingInfo | null>(null);
  const families = useMemo(() => mergeFamilies(info?.families), [info?.families]);

  const setFamilyName = onFamilyNameChange;
  useEffect(() => {
    onFamiliesChange?.(families);
  }, [families, onFamiliesChange]);
  const family = useMemo(
    () => families.find((f) => f.name === familyName) ?? families[0],
    [families, familyName],
  );
  // Raw backend family record (precision_modes / recommended_precision / supports_compile live
  // only here). Absent on an older backend.
  const reportedFamily = useMemo(
    () => info?.families?.find((f) => f.name === familyName),
    [info?.families, familyName],
  );
  // sdxl trains the U-Net in mixed precision, so it uses mixed_precision instead of base_precision.
  // Everything else is a DiT family.
  const isDiT = familyName !== "sdxl";
  // An EMPTY precision_modes list on a DiT family means this host cannot train it at all (the
  // reason rides in vram_note); only an ABSENT field means an older backend. SDXL reports []
  // too but is not precision-gated, hence the isDiT scope.
  const familyUntrainable =
    isDiT &&
    reportedFamily?.precision_modes != null &&
    reportedFamily.precision_modes.length === 0;
  // Quantised base precisions this family can train in, with a stable fallback when the backend does not report them.
  const precisionModes = useMemo<
    Array<"nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto">
  >(() => {
    if (familyUntrainable) return [];
    const reported = reportedFamily?.precision_modes?.filter(
      (m): m is "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" =>
        m === "nf4" || m === "bf16" || m === "int8" || m === "fp8" || m === "mxfp8",
    );
    if (reported && reported.length > 0) return ["auto", ...reported];
    // Fallback without a backend report: the GPU-independent modes only (mxfp8 needs a Blackwell probe).
    return ["auto", "nf4", "bf16", "int8", "fp8"];
  }, [reportedFamily?.precision_modes, familyUntrainable]);
  // Whether to show the torch.compile control; default on for DiT families on an older backend.
  const supportsCompile = reportedFamily?.supports_compile ?? isDiT;
  // Same for checkpoints: MiniMax-H3's loop writes no resume bundle and its validation REFUSES a
  // nonzero save_steps, so offering the field meant a rejected Start with nothing saying why.
  const supportsCheckpoints = reportedFamily?.supports_checkpoints ?? true;
  // Same for the batch axis: MiniMax-H3's forward covers ONE packed sequence and its validation
  // REFUSES a batch above 1 rather than clamping, so a value carried over from another family
  // rejected Start with nothing saying why. Hidden when the family caps it at 1.
  const maxBatchSize = reportedFamily?.max_train_batch_size ?? null;
  const batchIsFixed = maxBatchSize != null && maxBatchSize <= 1;

  const setBaseChoice = onBaseChoiceChange;
  const [customBase, setCustomBase] = useState("");

  const [dataset, setDataset] = useState<string>(UPLOAD_DATASET);
  const [uploadName, setUploadName] = useState("my-images");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Adds to the selected set; the other input creates a new one.
  const addInputRef = useRef<HTMLInputElement | null>(null);
  // one folder picker serves both targets, so the destination is captured when it opens.
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const folderTarget = useRef("");
  const [dropActive, setDropActive] = useState(false);
  // the authoritative in-flight guard: `uploading` is render state and reads stale in a closure.
  const uploadInFlight = useRef(false);
  const [gridOpen, setGridOpen] = useState(false);
  const [gridRefresh, setGridRefresh] = useState(0);
  const [examples, setExamples] = useState<DiffusionDatasetExample[]>([]);
  const [importingId, setImportingId] = useState<string | null>(null);

  const [outputDir, setOutputDir] = useState("");
  const [instancePrompt, setInstancePrompt] = useState("");

  const [steps, setSteps] = useState(500);
  // Run length is set in either steps or epochs; the trainer resolves epochs -> steps once the dataset size is known.
  const [durationUnit, setDurationUnit] = useState<"steps" | "epochs">("steps");
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(family?.defaults.lr ?? 0.0001);
  const [rank, setRank] = useState(family?.defaults.rank ?? 16);
  const [resolution, setResolution] = useState(family?.defaults.resolution ?? 768);
  const [batchSize, setBatchSize] = useState(1);
  const effectiveBatchSize = maxBatchSize == null ? batchSize : Math.min(batchSize, maxBatchSize);
  const [gradAccum, setGradAccum] = useState(1);
  const [seed, setSeed] = useState(42);
  // Periodic resume points. 0 (off) keeps the default: only a stop-and-save writes one, so nothing
  // is spent on disk unless asked.
  const [saveSteps, setSaveSteps] = useState(0);
  // LR schedule. Warmup applies only to non-constant schedules. Seeded from the family, which is
  // where the flow-matching DiTs' short ramp comes from.
  const [lrScheduler, setLrScheduler] = useState<LrScheduler>("constant");
  const [lrWarmupSteps, setLrWarmupSteps] = useState(0);
  // Gradient checkpointing trades ~20-30% step time for a large activation-VRAM saving.
  const [gradCheckpoint, setGradCheckpoint] = useState(true);
  // sdxl (U-Net) trains in a mixed-precision autocast; the DiT families quantise the frozen base
  // weights and ignore this.
  const [precision, setPrecision] = useState<"bf16" | "fp16" | "no">("bf16");
  // Quantised base precision for DiT families (nf4 QLoRA default). "auto" lets the backend pick the
  // family's recommendation.
  const [basePrecision, setBasePrecision] = useState<
    "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto"
  >("auto");
  // Whether to torch.compile the DiT transformer. "auto" defers to the backend.
  const [compileTransformer, setCompileTransformer] = useState<"off" | "on" | "auto">(
    "auto",
  );
  // Track whether the user hand-edited the numeric settings; if not, a family change re-seeds them
  // from that family's defaults.
  const settingsDirty = useRef(false);
  // The LR schedule pair tracks its own edits rather than riding on settingsDirty: it is the one
  // setting whose control DISAPPEARS ("Warmup steps" is hidden under plain "constant"), so a
  // value carried past a family change is invisible rather than merely stale.
  const lrScheduleDirty = useRef(false);
  // Track whether the user hand-picked a base precision; if not, a family change re-seeds it from
  // recommended_precision.
  const precisionDirty = useRef(false);
  // Same for the base repo. `family` is a fresh object after every refreshInfo(), so the seeding
  // effect re-runs on an unrelated refresh and would replace the pick; track it by name.
  const baseDirty = useRef(false);
  const seededBaseFamily = useRef<string | null>(null);

  const [starting, setStarting] = useState(false);
  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);
  // Persisted previous runs (terminal), listed on the idle view; selecting one re-plots its logs read-only.
  const [prevRuns, setPrevRuns] = useState<DiffusionTrainingRunSummary[]>([]);
  const [viewRun, setViewRun] = useState<DiffusionTrainingRunDetail | null>(null);
  // The confirm-stop dialog (mirrors the LLM Train tab): Continue / Stop / Stop and save.
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  // Set when the user confirms a stop. Clamped to the running state at read time so a fresh run never inherits it.
  const [stopRequestedLocal, setStopRequestedLocal] = useState(false);
  // The run whose Resume request is in flight, so its button alone shows the pending label.
  const [resumingJobId, setResumingJobId] = useState<string | null>(null);

  const refreshInfo = useCallback(async (): Promise<DiffusionTrainingInfo | null> => {
    try {
      const i = await getDiffusionTrainingInfo();
      setInfo(i);
      return i;
    } catch {
      return null;
    }
  }, []);

  // On first activation, load the dataset list and preselect a base matching the loaded generation
  // model when it is trainable.
  useEffect(() => {
    if (!active) return;
    void refreshInfo().then((i) => {
      setDataset((cur) => {
        if (cur !== UPLOAD_DATASET && i?.datasets.some((d) => d.name === cur)) return cur;
        return i && i.datasets.length > 0 ? i.datasets[0].name : UPLOAD_DATASET;
      });
    });
  }, [active, refreshInfo]);

  // Load the curated example list once. Best-effort: an older backend without the endpoint just yields no examples.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    listDiffusionDatasetExamples()
      .then((list) => {
        if (!cancelled) setExamples(list);
      })
      .catch(() => {
        if (!cancelled) setExamples([]);
      });
    return () => {
      cancelled = true;
    };
  }, [active]);

  // Examples whose folder is not on disk yet. An example imports into a folder named after its id,
  // so a matching dataset name means it is already imported.
  const importedNames = useMemo(
    () => new Set((info?.datasets ?? []).map((d) => d.name)),
    [info?.datasets],
  );
  const pendingExamples = useMemo(
    () => examples.filter((ex) => !importedNames.has(ex.id)),
    [examples, importedNames],
  );

  // Import a curated example, then select the resulting folder. Seeds the trigger prompt only when
  // the field is meaningful.
  const importExample = useCallback(
    async (ex: DiffusionDatasetExample) => {
      setImportingId(ex.id);
      try {
        const res = await runExampleImport(ex);
        await refreshInfo();
        setDataset(res.name);
        setGridOpen(false);
        setGridRefresh((k) => k + 1);
        if (ex.suggested_trigger && res.caption_count === 0 && !instancePrompt.trim()) {
          setInstancePrompt(ex.suggested_trigger);
        }
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Import failed");
      } finally {
        setImportingId(null);
      }
    },
    [refreshInfo, instancePrompt],
  );

  // If the loaded generation model is a trainable family, jump the family selector to it once.
  const seededFromLoaded = useRef(false);
  useEffect(() => {
    if (seededFromLoaded.current) return;
    if (!loadedFamily) return;
    if (families.some((f) => f.name === loadedFamily)) {
      setFamilyName(loadedFamily);
      seededFromLoaded.current = true;
    }
  }, [loadedFamily, families]);

  // Re-seed base and numeric settings from family defaults on family change (unless the user edited
  // them). Prefer the loaded base repo when it belongs to this family.
  useEffect(() => {
    if (!family) return;
    // A NEW family invalidates any earlier base pick; a mere info refresh does not, so compare by
    // name rather than object identity.
    if (seededBaseFamily.current !== family.name) {
      seededBaseFamily.current = family.name;
      baseDirty.current = false;
    }
    // An already-valid base wins: the top bar sets family and base together. A loaded checkpoint may
    // be the DISTILLED half of a pair, which is never in base_repos, so fall back to the paired
    // training base before base_repos[0], or opening Train with the 9B model loaded seeds the 4B
    // base. reportedFamily, since deploy_bases is backend-only.
    const pairedTrainingBase = loadedBaseRepo
      ? resolveDiffusionTrainingBase(reportedFamily, loadedBaseRepo)
      : null;
    const preferLoaded = family.base_repos.includes(baseChoice)
      ? baseChoice
      : loadedBaseRepo && family.base_repos.includes(loadedBaseRepo)
        ? loadedBaseRepo
        : (pairedTrainingBase ?? family.base_repos[0] ?? CUSTOM_BASE);
    if (!baseDirty.current) setBaseChoice(preferLoaded);
    if (!settingsDirty.current) {
      setLearningRate(family.defaults.lr);
      setRank(family.defaults.rank);
      setResolution(family.defaults.resolution);
    }
    // The ramp is seeded from the family or reset with it, or the 20 steps recommended for a
    // flow-matching DiT ride along into SDXL, which never asked for one.
    if (!lrScheduleDirty.current) {
      setLrScheduler(family.defaults.lrScheduler ?? "constant");
      setLrWarmupSteps(family.defaults.lrWarmupSteps ?? 0);
    }
    // Re-seed the DiT base precision from the family recommendation (unless the user picked one);
    // "auto" is always safe.
    if (!precisionDirty.current) {
      const rec = reportedFamily?.recommended_precision;
      setBasePrecision(
        rec === "nf4" || rec === "bf16" || rec === "int8" || rec === "fp8"
          ? rec
          : "auto",
      );
    }
  }, [family, loadedBaseRepo, reportedFamily]);

  // mixed_precision is an SDXL-only lever and every DiT family trains in bf16, so reset to bf16 on
  // a change to a DiT family, or an fp16 value left from SDXL rides along and is rejected.
  useEffect(() => {
    if (isDiT) setPrecision("bf16");
  }, [isDiT]);

  // The base actually used everywhere. baseChoice can briefly hold another family's repo, where a
  // raw <select value> would DISPLAY the first option while the request carried the stale repo,
  // so clamp to the current family.
  const effectiveBase =
    baseChoice === CUSTOM_BASE || (family?.base_repos ?? []).includes(baseChoice)
      ? baseChoice
      : family?.base_repos[0] ?? CUSTOM_BASE;

  // The resolved base repo/path the request will carry, and whether it looks prequantized. The
  // dense precisions are invalid for such a repo.
  const resolvedBase = (effectiveBase === CUSTOM_BASE ? customBase : effectiveBase).trim();
  const basePrequantized = isDiT && repoIsPrequantized(resolvedBase);

  // A prequantized base cannot serve the dense precisions; auto-flip a dense selection back to
  // "auto" (nf4) so the run does not fail at the backend validator. Reuses precisionDirty so a
  // later family change still re-seeds.
  useEffect(() => {
    if (basePrequantized && DENSE_PRECISIONS.has(basePrecision)) {
      precisionDirty.current = false;
      setBasePrecision("auto");
    }
  }, [basePrequantized, basePrecision]);

  const poll = useCallback(async () => {
    try {
      setStatus(await getDiffusionTrainingStatus());
    } catch {
      /* best-effort; a failed poll should not surface an error while the tab is open */
    }
  }, []);

  // Poll status while the panel is active.
  useEffect(() => {
    if (!active) return;
    void poll();
    const id = window.setInterval(() => void poll(), 1500);
    return () => window.clearInterval(id);
  }, [active, poll]);

  // "Train another" dismisses the completed run card locally: the backend keeps the terminal status
  // until the next start.
  const [dismissedJobId, setDismissedJobId] = useState<string | null>(null);
  const running = Boolean(status?.active) || status?.status === "running";
  const completed =
    status?.status === "completed" && status.job_id !== dismissedJobId;
  // "Stop and save" ends the run WITH a saved partial adapter, so it gets the same ready-to-deploy card as a full run.
  const stoppedWithAdapter =
    status?.status === "stopped" &&
    Boolean(status?.lora_path) &&
    status.job_id !== dismissedJobId;
  const pct =
    status && status.total_steps > 0
      ? Math.min(100, Math.round((status.step / status.total_steps) * 100))
      : 0;

  // The pending-stop flag only matters while a run is active; clamping at read time means a fresh
  // run never inherits a stale "Stopping...".
  const stopRequested = running && stopRequestedLocal;

  // The just-finished run's persisted record, where can_resume lives. It appears a beat after the
  // terminal status via the delayed refetch below, so Resume enables itself once it lands.
  const liveRunSummary = useMemo(
    () => prevRuns.find((r) => r.job_id === status?.job_id) ?? null,
    [prevRuns, status?.job_id],
  );

  // Whether there is a run to show live: running, or ANY terminal run the user has not dismissed.
  // Dismissing must cover every terminal status, or "Train another" traps the run view.
  const terminalStatuses = ["completed", "stopped", "error"];
  const hasRun = Boolean(
    status &&
      status.status !== "idle" &&
      !(terminalStatuses.includes(status.status) && status.job_id === dismissedJobId),
  );

  // Notify the parent once per run that produced an adapter so it rescans the LoRA picker. Re-armed
  // here and in onStart, so a second run still notifies even if "running" is never observed.
  const notifiedComplete = useRef(false);
  useEffect(() => {
    const producedAdapter =
      status?.status === "completed" ||
      (status?.status === "stopped" && Boolean(status?.lora_path));
    if (producedAdapter && !notifiedComplete.current) {
      notifiedComplete.current = true;
      onTrainingComplete?.();
    } else if (status?.status === "running" && notifiedComplete.current) {
      notifiedComplete.current = false;
    }
  }, [status?.status, status?.lora_path, onTrainingComplete]);

  const selectedDataset =
    dataset !== UPLOAD_DATASET ? info?.datasets.find((d) => d.name === dataset) : undefined;
  // A deleted dataset leaves a name that no longer resolves; fall back to the upload form.
  const uploadMode = dataset === UPLOAD_DATASET || (info !== null && !selectedDataset);
  // Trainable items in the picked dataset, images and clips alike. caption_count is the folder
  // total over both kinds, so every ratio must be against this and not image_count.
  const selectedItemCount = selectedDataset ? datasetItemCount(selectedDataset) : 0;
  // A dataset where every item already ships a caption needs no trigger prompt; hide the field and explain why.
  const fullyCaptioned = Boolean(
    selectedDataset &&
      selectedItemCount > 0 &&
      selectedDataset.caption_count >= selectedItemCount,
  );

  // Map the backend's paired history arrays into the chart component's {step,value} series.
  const lossHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = status?.metric_history;
    if (!h) return [];
    return h.steps.map((step, i) => ({ step, value: h.loss[i] })).filter((p) => p.value != null);
  }, [status?.metric_history]);
  const gradNormHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = status?.metric_history;
    if (!h?.grad_norm) return [];
    return h.steps
      .map((step, i) => ({ step, value: h.grad_norm?.[i] ?? null }))
      .filter((p): p is TrainingSeriesPoint => p.value != null);
  }, [status?.metric_history]);

  // Refresh the previous-runs list whenever the service is not mid-run (on mount and right after a run terminates).
  useEffect(() => {
    if (!active) return;
    if (status?.status === "running") return;
    let cancelled = false;
    const refetch = () => {
      listDiffusionTrainingRuns()
        .then((r) => {
          if (!cancelled) setPrevRuns(r.runs);
        })
        .catch(() => {});
    };
    refetch();
    // The service exposes a terminal status before the pump has necessarily written the run JSON, so
    // a short delayed second refetch lets the record land.
    let delayed: ReturnType<typeof setTimeout> | undefined;
    if (status?.status === "completed" || status?.status === "stopped" || status?.status === "error") {
      delayed = setTimeout(refetch, 1500);
    }
    return () => {
      cancelled = true;
      if (delayed !== undefined) clearTimeout(delayed);
    };
  }, [active, status?.status]);

  const openPrevRun = useCallback(async (jobId: string) => {
    try {
      setViewRun(await getDiffusionTrainingRun(jobId));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not load that run");
    }
  }, []);

  // Chart series for a selected previous run (from its persisted metric logs).
  const viewLossHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = viewRun?.metric_history;
    if (!h) return [];
    return h.steps.map((step, i) => ({ step, value: h.loss[i] })).filter((p) => p.value != null);
  }, [viewRun?.metric_history]);
  const viewGradNormHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = viewRun?.metric_history;
    if (!h?.grad_norm) return [];
    return h.steps
      .map((step, i) => ({ step, value: h.grad_norm?.[i] ?? null }))
      .filter((p): p is TrainingSeriesPoint => p.value != null);
  }, [viewRun?.metric_history]);

  // Uploads accumulate, so the same call both creates a folder and adds to an existing one.
  // `picked` can come from a file pick, a folder pick or a drop, so it is filtered here.
  const uploadTo = useCallback(
    async (name: string, picked: File[]) => {
      if (picked.length === 0) return;  // the picker was cancelled
      if (!name) {
        toast.error("Give the dataset a folder name, e.g. my-style-photos.");
        return;
      }
      const { files, imageCount, clipCount, skipped, collisions } = selectDatasetFiles(picked);
      if (collisions.length > 0) {
        const { kind, first, second } = collisions[0];
        const more =
          collisions.length > 1
            ? ` ${collisions.length - 1} other pair${collisions.length === 2 ? "" : "s"} clash too.`
            : "";
        toast.error(
          (kind === "name"
            ? `"${first}" and "${second}" have the same file name, and a dataset folder is ` +
              "flat, so one would overwrite the other."
            : `"${first}" and "${second}" differ only by extension, so they would share one ` +
              "caption file.") + `${more} Rename them and upload again.`,
        );
        return;
      }
      if (files.length === 0) {
        toast.error(
          `Nothing to upload. Pick images (${DATASET_IMAGE_EXTS.join(", ")}) or ` +
            `clips (${DATASET_CLIP_EXTS.join(", ")}) and, ` +
            "optionally, a caption file beside each one.",
        );
        return;
      }
      // /diffusion/info only lists folders holding a trainable item, so say why the set will not appear
      // yet rather than refusing a captions-first upload.
      const newCaptionsOnly =
        imageCount === 0 && clipCount === 0 && !(info?.datasets ?? []).some((d) => d.name === name);
      if (uploadInFlight.current) {
        toast.error("An upload is already running. Wait for it to finish, then try again.");
        return;
      }
      uploadInFlight.current = true;
      setUploading(true);
      try {
        const misKeyed = await metadataKeyedOnSubfolders(files);
        // the endpoint accumulates into the same folder, so a tree past the multipart part or byte cap
        // goes up in slices. The cap decides where the slices fall, so it is forced past the cache,
        // and a cap that cannot be read stops the upload.
        let maxBytes: number;
        try {
          maxBytes = (await loadUploadLimitSettings({ force: true })).maxUploadSizeBytes;
        } catch (e) {
          toast.error(
            "Could not read the upload size limit, so nothing was uploaded: " +
              `${e instanceof Error ? e.message : "the limit could not be read"}. Try again.`,
          );
          return;
        }
        const chunks = chunkDatasetUpload(files, maxBytes);
        // a slice no split can fit under the cap 413s, so refuse before the first request: otherwise the
        // slices ahead of it are already committed.
        const over = oversizedChunk(chunks, maxBytes);
        if (over) {
          toast.error(
            `"${over}" is over the ${getCachedUploadLimitLabel()} upload limit, so nothing was ` +
              "uploaded. Raise the limit in Settings, or leave that file out.",
          );
          return;
        }
        // one request is all-or-nothing on the backend, so only a split top-up needs the folder checked
        // as well: there a stem it already holds would 400 a later slice, mid-commit.
        if (chunks.length > 1) {
          // the list held here is the one from mount, so it cannot say the folder is new.
          const known = await refreshInfo();
          if (!known) {
            toast.error(
              "Could not read the dataset list, so nothing was uploaded. Try again.",
            );
            return;
          }
          // a case-insensitive dataset root resolves "photos" onto an existing "Photos", so an exact match
          // would call that folder new and skip the check. Fall back to a folded match.
          const folder =
            known.datasets.find((d) => d.name === name) ??
            known.datasets.find((d) => d.name.toLowerCase() === name.toLowerCase());
          if (folder) {
            // no listing, no guarantee: skipping the check would upload the slices it exists to hold back, so
            // a failure here stops the upload.
            let held: Awaited<ReturnType<typeof listDiffusionDatasetImages>>;
            try {
              held = await listDiffusionDatasetImages(folder.name);
            } catch (e) {
              toast.error(
                `Could not read what "${folder.name}" already holds, so nothing was uploaded: ` +
                  `${e instanceof Error ? e.message : "the dataset could not be listed"}. Try again.`,
              );
              return;
            }
            const clash = existingStemClash(files, held.images.map((i) => i.filename));
            if (clash) {
              toast.error(
                `"${clash.second}" and "${clash.first}", already in "${folder.name}", differ only by ` +
                  "extension, so they would share one caption file. Nothing was uploaded; " +
                  "rename it and try again.",
              );
              return;
            }
          }
        }
        let res = await uploadDiffusionDataset(name, chunks[0]);
        let sent = res.uploaded;
        let stopped: string | null = null;
        for (const chunk of chunks.slice(1)) {
          try {
            res = await uploadDiffusionDataset(name, chunk);
            sent += res.uploaded;
          } catch (e) {
            stopped = e instanceof Error ? e.message : "upload failed";
            break;
          }
        }
        // caption_count is the folder total the backend resolves over sidecars and metadata alike.
        if (stopped) {
          toast.error(
            `Uploaded ${sent} of ${files.length} files into "${res.name}", then stopped: ` +
              `${stopped}`,
          );
        } else {
          toast.success(
            `Uploaded ${sent} file${sent === 1 ? "" : "s"} - ` +
              `"${res.name}" now has ${datasetItemLabel(res)}, ` +
              `${res.caption_count} captioned`,
          );
        }
        if (skipped > 0) {
          toast.info(
            `Skipped ${skipped} file${skipped === 1 ? "" : "s"} that ` +
              `${skipped === 1 ? "was" : "were"} neither an image, a clip, nor a caption.`,
          );
        }
        if (newCaptionsOnly) {
          toast.info(
            `"${res.name}" holds captions but no images or clips yet, so it stays out of the ` +
              "dataset picker until you add some.",
          );
        }
        if (misKeyed) {
          toast.info(
            `${misKeyed} keys its captions on subfolder paths, and a dataset folder is flat, ` +
              "so those rows will not match. A .txt beside each file always will.",
          );
        }
        await refreshInfo();
        setDataset(res.name);
        setGridRefresh((k) => k + 1);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Upload failed");
      } finally {
        uploadInFlight.current = false;
        setUploading(false);
      }
    },
    [info, refreshInfo],
  );

  const pickFolder = useCallback((name: string) => {
    folderTarget.current = name;
    folderInputRef.current?.click();
  }, []);

  // a drop lands on whichever set the field is showing, filling a new folder or topping up the selected one.
  const dropTarget = uploadMode ? uploadName.trim() : dataset;
  const onDrop = useCallback(
    async (event: DragEvent) => {
      if (isTauri) return;
      // only claim file drops: preventDefault on a text drag kills editing in the caption boxes.
      if (!event.dataTransfer.types.includes("Files")) return;
      event.preventDefault();
      setDropActive(false);
      if (uploadInFlight.current) {
        toast.error("An upload is already running. Wait for it to finish, then drop again.");
        return;
      }
      let dropped: File[];
      // held across the walk too: a big tree takes seconds, and every control is live until then.
      uploadInFlight.current = true;
      setUploading(true);
      try {
        dropped = await filesFromDataTransfer(event.dataTransfer);
      } catch {
        toast.error(
          "Could not read every file in that folder, so nothing was uploaded. Try the folder " +
            "button instead.",
        );
        return;
      } finally {
        // uploadTo re-takes it synchronously below, so no await sits in the gap.
        uploadInFlight.current = false;
        setUploading(false);
      }
      if (dropped.length === 0) {
        toast.error("That drop had no files in it.");
        return;
      }
      await uploadTo(dropTarget, dropped);
    },
    [dropTarget, uploadTo],
  );

  const onStart = useCallback(async () => {
    const baseModel = (effectiveBase === CUSTOM_BASE ? customBase : effectiveBase).trim();
    if (!baseModel) {
      toast.error("Pick a base model (or fill in the custom repo/path).");
      return;
    }
    if (dataset === UPLOAD_DATASET) {
      toast.error("Upload your training images first (or pick an existing dataset).");
      return;
    }
    if (!outputDir.trim()) {
      toast.error("Name the adapter (this becomes its folder under Unsloth outputs).");
      return;
    }
    // Require a trigger prompt whenever ANY image lacks a caption: without an instance_prompt the
    // backend silently skips every uncaptioned image.
    if (
      selectedDataset &&
      selectedDataset.caption_count < selectedItemCount &&
      !instancePrompt.trim()
    ) {
      toast.error(
        selectedDataset.caption_count === 0
          ? "This dataset has no captions - add a trigger prompt so the trainer knows " +
              "what to learn (it becomes the caption for every item)."
          : `Only ${selectedDataset.caption_count} of ${selectedItemCount} items ` +
              "have captions - the rest would be silently skipped. Add a trigger prompt " +
              "(it becomes their caption) or caption every one.",
      );
      return;
    }
    if (durationUnit === "epochs") {
      if (epochs < 1) return toast.error("Epochs must be at least 1.");
    } else if (steps < 1) {
      return toast.error("Steps must be at least 1.");
    }
    if (rank < 1) return toast.error("LoRA rank must be at least 1.");
    if (resolution < 64 || resolution % 8 !== 0) {
      return toast.error("Resolution must be a multiple of 8 and at least 64.");
    }
    if (batchSize < 1) return toast.error("Batch size must be at least 1.");
    if (gradAccum < 1) return toast.error("Gradient accumulation must be at least 1.");
    if (learningRate <= 0) return toast.error("Learning rate must be greater than 0.");
    if (lrWarmupSteps < 0) return toast.error("Warmup steps cannot be negative.");
    setStarting(true);
    // A previous run's confirmed stop must not leak into this run: the read-time clamp would re-arm a
    // permanently disabled "Stopping..." button.
    setStopRequestedLocal(false);
    // Re-arm the completion notification, so a second run still notifies even if its "running" phase is never observed.
    notifiedComplete.current = false;
    // A history view must not shadow the new live run.
    setViewRun(null);
    try {
      await startDiffusionTraining({
        base_model: baseModel,
        model_family: family?.name,
        data_dir: dataset,
        output_dir: outputDir.trim(),
        instance_prompt: instancePrompt.trim() || undefined,
        resolution,
        // Epochs mode overrides train_steps on the backend, so send num_epochs and omit train_steps.
        train_steps: durationUnit === "epochs" ? undefined : steps,
        num_epochs: durationUnit === "epochs" ? epochs : undefined,
        learning_rate: learningRate,
        // The family cap, not the field: it is only hidden, not reset, so a value typed for another
        // family would otherwise still be sent and refused.
        train_batch_size: effectiveBatchSize,
        gradient_accumulation_steps: gradAccum,
        seed,
        gradient_checkpointing: gradCheckpoint,
        lr_scheduler: lrScheduler,
        lr_warmup_steps: lrScheduler === "constant" ? 0 : lrWarmupSteps,
        lora_rank: rank,
        // Zero rather than the field's value when the family has none: the field is only hidden, not
        // reset, so a value typed for one family would otherwise still be sent and refused.
        save_steps: supportsCheckpoints ? Math.max(0, Math.floor(saveSteps)) : 0,
        mixed_precision: precision,
        // DiT families quantise the base weights; sdxl uses mixed_precision above and ignores this. Only
        // send compile where supported.
        base_precision: isDiT ? basePrecision : undefined,
        compile_transformer: supportsCompile ? compileTransformer : undefined,
        hf_token: hfApiToken(getHfToken()) || undefined,
      });
      toast.success("Training started");
      void poll();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      setStarting(false);
    }
  }, [
    effectiveBase,
    customBase,
    family,
    dataset,
    selectedDataset,
    selectedItemCount,
    outputDir,
    instancePrompt,
    resolution,
    steps,
    durationUnit,
    epochs,
    learningRate,
    batchSize,
    gradAccum,
    seed,
    saveSteps,
    gradCheckpoint,
    lrScheduler,
    lrWarmupSteps,
    rank,
    precision,
    isDiT,
    basePrecision,
    supportsCompile,
    supportsCheckpoints,
    effectiveBatchSize,
    compileTransformer,
    poll,
  ]);

  // Continue a stopped run from its newest checkpoint: the stored config is replayed verbatim with
  // the run's output directory as resume_from_checkpoint. The detail is re-fetched on click so
  // `can_resume` reflects the checkpoints on disk right now.
  const onResume = useCallback(
    async (jobId: string) => {
      setResumingJobId(jobId);
      try {
        const detail = await getDiffusionTrainingRun(jobId);
        const payload = buildDiffusionResumePayload(detail, {
          hfToken: hfApiToken(getHfToken()) || undefined,
        });
        await startDiffusionTraining(payload);
        // Only after the start is ACCEPTED: a refusal must leave the history detail the user was reading
        // exactly as it was.
        setStopRequestedLocal(false);
        notifiedComplete.current = false;
        setViewRun(null);
        setDismissedJobId(null);
        toast.success(
          detail.checkpoint_step != null
            ? `Resuming from step ${detail.checkpoint_step}`
            : "Resuming training",
        );
        void poll();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed to resume training");
      } finally {
        setResumingJobId(null);
      }
    },
    [poll],
  );

  // Confirm-then-stop. `save` writes the current adapter before halting; false discards it.
  const onStop = useCallback(
    async (save: boolean) => {
      setStopDialogOpen(false);
      setStopRequestedLocal(true);
      try {
        await stopDiffusionTraining(save);
        toast.success(
          save
            ? "Stop requested; saving the adapter after the current step."
            : "Stop requested; discarding this run after the current step.",
        );
        void poll();
      } catch (e) {
        setStopRequestedLocal(false);
        toast.error(e instanceof Error ? e.message : "Failed to stop training");
      }
    },
    [poll],
  );

  // Resolve the repo an adapter should be previewed on. Variant-specific pairs cover FLUX.2 Klein's
  // 4B and 9B bases; the scalar fallback keeps older backends and Krea 2 working.
  const deployBaseFor = useCallback(
    (trainedBase: string, famName: string): string => {
      const rec = info?.families?.find((f) => f.name === famName);
      return resolveDiffusionDeployBase(rec, trainedBase);
    },
    [info?.families],
  );

  const onDeployClick = useCallback(() => {
    if (!status?.catalog_path) {
      toast.error("The trained adapter is not available yet.");
      return;
    }
    const trainedBase = status.base_model || (effectiveBase === CUSTOM_BASE ? customBase : effectiveBase);
    if (!trainedBase) {
      toast.error("Could not determine the base model to load for this adapter.");
      return;
    }
    const famName = status.family || family?.name || "";
    onDeploy?.({
      baseRepo: deployBaseFor(trainedBase, famName),
      family: famName,
      catalogPath: status.catalog_path,
      trigger: instancePrompt.trim(),
    });
  }, [status, effectiveBase, customBase, family, instancePrompt, onDeploy, deployBaseFor]);

  const numberField = (
    label: string,
    value: number,
    set: (n: number) => void,
    fallback: number,
    // markDirty overrides which dirty flag an edit claims. Only "Warmup steps" passes one: it is
    // seeded from the family like rank/LR/resolution but tracked by lrScheduleDirty, so charging
    // it to the shared flag would freeze the other three at the previous family's values.
    extra?: { min?: number; step?: number; hint?: ReactNode; markDirty?: () => void },
  ) => (
    <div className={fieldClass}>
      <FieldLabel hint={extra?.hint}>{label}</FieldLabel>
      <Input
        type="number"
        min={extra?.min ?? 1}
        step={extra?.step}
        value={value}
        onChange={(e) => {
          if (extra?.markDirty) extra.markDirty();
          else settingsDirty.current = true;
          // Only fall back when the input parses to NaN; a real 0 is legal for zero-legal fields (Seed, LR
          // warmup steps).
          const parsed = Number(e.target.value);
          set(Number.isNaN(parsed) ? fallback : parsed);
        }}
        className="h-8 text-xs"
      />
    </div>
  );

  // Run length: a number paired with a compact unit select. Epochs mode trains that many full
  // passes; the backend resolves it to steps.
  const durationField = (
    <div className={fieldClass}>
      <FieldLabel
        hint={
          <>
            How long the run trains. Steps count optimizer updates; epochs count full
            passes over your images. 500&ndash;1500 steps suits most small sets.
          </>
        }
      >
        {durationUnit === "epochs" ? "Epochs" : "Steps"}
      </FieldLabel>
      <div className="flex gap-1.5">
        <Input
          type="number"
          min={1}
          value={durationUnit === "epochs" ? epochs : steps}
          onChange={(e) => {
            settingsDirty.current = true;
            const n = Number(e.target.value) || 1;
            if (durationUnit === "epochs") setEpochs(n);
            else setSteps(n);
          }}
          className="h-8 min-w-0 flex-1 text-xs"
        />
        <Select
          value={durationUnit}
          onValueChange={(v) => {
            settingsDirty.current = true;
            setDurationUnit(v as "steps" | "epochs");
          }}
        >
          {/* Tighter than the default trigger: it holds one short word, not a model id. */}
          <SelectTrigger
            className="h-8 w-24 pr-2.5 text-xs [&_svg]:size-3.5"
            aria-label="Run length unit"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="steps">Steps</SelectItem>
            <SelectItem value="epochs">Epochs</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );

  const precisionLabel = (
    m: "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto",
  ): string => {
    if (m === "auto") return "Auto";
    if (m === "nf4") return "nf4 (lowest VRAM)";
    if (m === "bf16") return "bf16 (fastest)";
    if (m === "int8") return "int8";
    if (m === "mxfp8") return "mxfp8 (Blackwell)";
    return "fp8 (experimental)";
  };

  // The training settings, the run area's MAIN content before a run starts. Columns key off this
  // pane's OWN width, not the window's, or a viewport breakpoint put three columns in a ~280px
  // pane. A cell needs 150px (66px number field + 6px gap + 78px unit select), hence 324px for
  // two columns and 498px for three.
  const trainingSettings = (
    <div className="@container flex flex-col gap-6">
      <div className="grid grid-cols-1 gap-x-6 gap-y-5 @min-[324px]:grid-cols-2 @min-[498px]:grid-cols-3">
        {durationField}
        {numberField("LoRA rank", rank, setRank, 1, {
          hint: "How much the adapter can learn. Higher captures more detail and makes a bigger file; 16 suits most styles, 32+ for complex subjects.",
        })}
        {numberField("Resolution", resolution, setResolution, 512, {
          min: 64,
          step: 64,
          hint: "The pixel size images train at, in multiples of 64. Higher is sharper and costs noticeably more VRAM.",
        })}
        {!batchIsFixed &&
          numberField("Batch", batchSize, setBatchSize, 1, {
            hint: "Images trained on per step. Higher is faster per image and needs more VRAM.",
          })}
        {numberField("Grad accumulation", gradAccum, setGradAccum, 1, {
          hint: batchIsFixed
            ? "Collects this many clips before each update. This model trains one clip at a time, so this is the only way to raise the effective batch."
            : "Collects this many batches before each update, for the effect of a larger batch without the VRAM. Effective batch = Batch x Grad accumulation.",
        })}
        {numberField("Seed", seed, setSeed, 42, {
          min: 0,
          hint: "Fixes the run's randomness, so the same settings and images reproduce the same LoRA.",
        })}
        {supportsCheckpoints &&
          numberField("Checkpoint every", saveSteps, setSaveSteps, 0, {
            min: 0,
            hint: "Saves a resume point every this many steps, so a crash or a shutdown can be picked up where it left off. 0 turns it off; stopping and saving always leaves one either way.",
          })}
      </div>

      <div className="grid grid-cols-1 items-start gap-x-6 gap-y-5 @min-[324px]:grid-cols-2 @min-[498px]:grid-cols-3">
        {numberField("Learning rate", learningRate, setLearningRate, 0.0001, {
          min: 0,
          step: 0.00001,
          hint: "How big each update is. Too high burns the style in and adds artifacts; too low barely learns. 0.0001 is a safe start.",
        })}
        <div className={fieldClass}>
          <FieldLabel hint="How the learning rate moves over the run. Constant is fine for most runs; a decay can help a long one settle.">
            LR schedule
          </FieldLabel>
          <Select
            value={lrScheduler}
            onValueChange={(v) => {
              // The family re-seed now writes this field, so without a dirty mark a hand-picked schedule is
              // replaced on the next family switch.
              lrScheduleDirty.current = true;
              setLrScheduler(v as LrScheduler);
            }}
          >
            <SelectTrigger className={selectClass} aria-label="LR schedule">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="constant">Constant</SelectItem>
              <SelectItem value="constant_with_warmup">Constant + warmup</SelectItem>
              <SelectItem value="cosine">Cosine decay</SelectItem>
              <SelectItem value="linear">Linear decay</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {lrScheduler !== "constant" &&
          numberField("Warmup steps", lrWarmupSteps, setLrWarmupSteps, 0, {
            min: 0,
            hint: "Ramps the learning rate up over the first steps instead of starting at full size.",
            // The other half of the pair, so an edit claims the pair's flag and only that, without freezing
            // rank/LR/resolution on their way to the new family.
            markDirty: () => {
              lrScheduleDirty.current = true;
            },
          })}
      </div>

      <div className="grid grid-cols-1 items-start gap-x-6 gap-y-5 @min-[324px]:grid-cols-2 @min-[498px]:grid-cols-3">
        <div className={fieldClass}>
          <FieldLabel hint="Recomputes activations instead of holding them in memory: less VRAM, slightly slower steps.">
            Gradient checkpointing
          </FieldLabel>
          <Select
            value={gradCheckpoint ? "on" : "off"}
            onValueChange={(v) => setGradCheckpoint(v === "on")}
          >
            <SelectTrigger className={selectClass} aria-label="Gradient checkpointing">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="on">On (less VRAM)</SelectItem>
              <SelectItem value="off">Off (faster steps)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {isDiT ? (
          <div className={fieldClass}>
            <FieldLabel hint="How the frozen base is quantised while the LoRA trains. Auto picks the best fit for your GPU; nf4 uses the least VRAM.">
              Base precision
            </FieldLabel>
            <Select
              value={basePrecision}
              onValueChange={(v) => {
                precisionDirty.current = true;
                setBasePrecision(v as typeof basePrecision);
              }}
              disabled={familyUntrainable}
            >
              <SelectTrigger className={selectClass} aria-label="Base precision">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {precisionModes.map((m) => (
                  <SelectItem
                    key={m}
                    value={m}
                    disabled={basePrequantized && DENSE_PRECISIONS.has(m)}
                  >
                    {precisionLabel(m)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {/* Only state stays on the page: both lines say why the control is limited right now. The general
                guidance is in the label's tooltip. */}
            {(familyUntrainable || basePrequantized) && (
              <p className="text-ui-11 leading-snug text-muted-foreground">
                {familyUntrainable
                  ?  // The reason itself already shows in the family picker note above.
                    "This GPU cannot train this model family."
                  : "This base is already 4-bit, so only nf4/auto apply."}
              </p>
            )}
          </div>
        ) : (
          <div className={fieldClass}>
            <FieldLabel hint="The mixed-precision mode for training math. bf16 is right for modern GPUs; fp16 is for older ones that lack it.">
              Precision
            </FieldLabel>
            <Select
              value={precision}
              onValueChange={(v) => setPrecision(v as "bf16" | "fp16" | "no")}
            >
              <SelectTrigger className={selectClass} aria-label="Precision">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bf16">bf16 (default)</SelectItem>
                <SelectItem value="fp16">fp16 (older GPUs)</SelectItem>
                <SelectItem value="no">fp32 (no mixed)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        {supportsCompile && (
          <div className={fieldClass}>
            <FieldLabel hint="torch.compile the transformer. The first step is slower while it compiles, every step after is faster.">
              Compile transformer
            </FieldLabel>
            <Select
              value={compileTransformer}
              onValueChange={(v) =>
                setCompileTransformer(v as typeof compileTransformer)
              }
            >
              <SelectTrigger className={selectClass} aria-label="Compile transformer">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="on">On (faster after warmup)</SelectItem>
                <SelectItem value="off">Off</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
    </div>
  );

  // overflow-x-hidden: an unset overflow-x computes to auto beside overflow-y-auto, letting a wide
  // row pan the page sideways on a phone.
  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden pr-5 sm:pr-8 @[50rem]:flex-row @[50rem]:overflow-hidden">
      {/* Left: configure. The 408px rail and container breakpoint match Create and the shared header. */}
      <div className="flex w-full min-w-0 shrink-0 flex-col border-b border-border/60 pl-10 @[50rem]:w-[408px] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0">
        {/* Keep the former row-level top inset inside the pane so the divider reaches the header. */}
        <div
          ref={attachSettingsScroll}
          onScroll={onSettingsScroll}
          className={cn(
            "hover-scrollbar panel-scroll-fade-action flex min-h-0 flex-1 flex-col gap-5 overflow-x-hidden pb-6 pl-0.5 pr-8 pt-[42px] @[50rem]:overflow-y-auto",
            settingsFadeClass,
          )}
        >
          {/* Icon rides the heading; the line below runs the full width. */}
          <div className="mb-1 grid gap-1.5">
            {/* Matches "Train settings" across the rule, so the two headings read as one row. */}
            <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none">
              <HugeiconsIcon
                icon={TestTubeOutlineIcon}
                className="size-[18px] shrink-0"
              />
              Train a LoRA
            </h2>
            <p className="text-xs leading-snug text-muted-foreground">
              Teach a model a new style or subject.
            </p>
          </div>

          {/* Family + base */}
          <div className={fieldClass}>
            <FieldLabel hint="The architecture you are training. Each family brings its own bases, starting hyperparameters and VRAM floor.">
              Model family
            </FieldLabel>
            <Select value={familyName} onValueChange={setFamilyName}>
              <SelectTrigger className={selectClass} aria-label="Model family">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {families.map((f) => (
                  <SelectItem key={f.name} value={f.name}>
                    {f.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <FamilyFacts family={family} baseModel={resolvedBase} />
          </div>

          <div className={fieldClass}>
            <FieldLabel hint="The exact checkpoint the LoRA trains against. A 4-bit (bnb) base needs the least VRAM; the dense one needs the most.">
              Base model
            </FieldLabel>
            <Select
              value={effectiveBase}
              onValueChange={(v) => {
                baseDirty.current = true;  // an explicit pick survives later info refreshes
                setBaseChoice(v);
              }}
            >
              <SelectTrigger className={selectClass} aria-label="Base model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(family?.base_repos ?? []).map((repo) => (
                  <SelectItem key={repo} value={repo}>
                    {repo}
                  </SelectItem>
                ))}
                <SelectItem value={CUSTOM_BASE}>Custom repo or local path...</SelectItem>
              </SelectContent>
            </Select>
            {effectiveBase === CUSTOM_BASE && (
              <Input
                value={customBase}
                placeholder="Repo id or local folder"
                spellCheck={false}
                onChange={(e) => setCustomBase(e.target.value)}
                className="h-8 text-xs"
              />
            )}
          </div>

          {/* Dataset */}
          {/* the whole field is the drop zone, so a folder can land on the picker, the thumbnails or the
              caption grid. */}
          <div
            className={cn(
              fieldClass,
              "rounded-lg transition-colors",
              dropActive && "outline-2 outline-dashed outline-offset-4 outline-primary/60",
            )}
            onDragOver={(e) => {
              // tauri consumes os drags before the webview, as the chat composer notes.
              if (isTauri) return;
              // files only: without this a text-selection drag also arms the zone.
              if (!e.dataTransfer.types.includes("Files")) return;
              e.preventDefault();
              setDropActive(true);
            }}
            onDragLeave={(e) => {
              // dragging onto a child fires leave on the parent, so ignore inner moves.
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
                setDropActive(false);
              }
            }}
            onDrop={(e) => void onDrop(e)}
          >
            <FieldLabel hint="The set the LoRA learns from. 10-50 images is plenty, and captions are optional.">
              Training images
            </FieldLabel>
            <div className="flex items-center gap-2">
              <Select
                value={uploadMode ? UPLOAD_DATASET : dataset}
                onValueChange={(v) => {
                  if (v.startsWith(EXAMPLE_PREFIX)) {
                    const ex = pendingExamples.find((x) => x.id === v.slice(EXAMPLE_PREFIX.length));
                    if (ex) void importExample(ex);
                    return;  // the controlled value stays put while the import runs
                  }
                  setDataset(v);
                  setGridOpen(false);
                }}
                disabled={importingId !== null}
              >
                <SelectTrigger className={cn(selectClass, "flex-1")} aria-label="Training images">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* Name plus item count only; captions and license show elsewhere. */}
                  {(info?.datasets ?? []).map((d) => (
                    <SelectItem key={d.name} value={d.name}>
                      {d.name} - {datasetItemLabel(d)}
                    </SelectItem>
                  ))}
                  {pendingExamples.length > 0 && (
                    <SelectGroup>
                      <SelectLabel>Examples</SelectLabel>
                      {pendingExamples.map((ex) => (
                        <SelectItem key={ex.id} value={`${EXAMPLE_PREFIX}${ex.id}`}>
                          {shortExampleLabel(ex.label)} - {ex.image_cap} images
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                  <SelectItem value={UPLOAD_DATASET}>Upload new images...</SelectItem>
                </SelectContent>
              </Select>
              {!uploadMode && selectedDataset && (
                <>
                  {/* The set is already named, so the pick uploads straight into it. */}
                  <input
                    ref={addInputRef}
                    type="file"
                    multiple
                    accept={DATASET_FILE_ACCEPT}
                    className="hidden"
                    aria-label="More training images"
                    onChange={(e) => {
                      const files = Array.from(e.target.files ?? []);
                      e.target.value = "";
                      void uploadTo(dataset, files);
                    }}
                  />
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-8 shrink-0 gap-1.5 px-3 text-xs"
                    onClick={() => addInputRef.current?.click()}
                    disabled={uploading}
                  >
                    <HugeiconsIcon icon={Upload01Icon} className="size-3.5" />
                    {uploading ? "Uploading..." : "Add"}
                  </Button>
                  <FolderPickButton disabled={uploading} onPick={() => pickFolder(dataset)} />
                </>
              )}
            </div>
            {/* a folder pick carries the whole tree, so an images-plus-captions set arrives paired in one go. */}
            <input
              ref={folderInputRef}
              type="file"
              multiple
              {...({ webkitdirectory: "", directory: "" } as object)}
              className="hidden"
              aria-label="Training image folder"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.target.value = "";
                void uploadTo(folderTarget.current, files);
              }}
            />
            {importingId && (
              <p className="text-ui-11 text-muted-foreground">
                Importing {examples.find((e) => e.id === importingId)?.label ?? "example"}...
              </p>
            )}

            {uploadMode ? (
              <div className={cn(fieldClass, "pb-4")}>
                <Label className="text-xs font-normal text-muted-foreground">
                  Name for this set of images
                </Label>
                <div className="flex items-center gap-2">
                  <Input
                    value={uploadName}
                    placeholder="my-photos"
                    spellCheck={false}
                    onChange={(e) => setUploadName(e.target.value)}
                    className="h-8 min-w-0 flex-1 text-xs"
                    aria-label="New dataset name"
                  />
                  {/* The native input is the file picker but never the control the user sees, so the row keeps the
                      app button styling. */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={DATASET_FILE_ACCEPT}
                    className="hidden"
                    aria-label="Training image files"
                    onChange={(e) => {
                      const files = Array.from(e.target.files ?? []);
                      e.target.value = "";
                      void uploadTo(uploadName.trim(), files);
                    }}
                  />
                  {/* The pick is the confirmation, so it uploads without a second click. */}
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-8 shrink-0 gap-1.5 px-3 text-xs"
                    onClick={() => {
                      if (!uploadName.trim()) {
                        toast.error("Give the dataset a folder name, e.g. my-style-photos.");
                        return;
                      }
                      fileInputRef.current?.click();
                    }}
                    disabled={uploading}
                  >
                    <HugeiconsIcon icon={Upload01Icon} className="size-3.5" />
                    {uploading ? "Uploading..." : "Upload"}
                  </Button>
                  <FolderPickButton
                    disabled={uploading}
                    onPick={() => {
                      if (!uploadName.trim()) {
                        toast.error("Give the dataset a folder name, e.g. my-style-photos.");
                        return;
                      }
                      pickFolder(uploadName.trim());
                    }}
                  />
                </div>
                <p className="text-ui-11 leading-snug text-muted-foreground">
                  {isTauri ? "Pick files or a folder." : "Pick files or a folder, or drop them here."}{" "}
                  Images, or clips for the video families. A caption file beside one (cat.png and
                  cat.txt, or cat.mp4 and cat.txt) is read as that item's caption.
                </p>
              </div>
            ) : (
              selectedDataset && (
                <>
                  {selectedDataset.image_count > 0 && !gridOpen && (
                    <DatasetShowcase
                      dataset={dataset}
                      imageCount={selectedDataset.image_count}
                      refreshKey={gridRefresh}
                      onBrowse={() => setGridOpen(true)}
                      onChanged={() => void refreshInfo()}
                    />
                  )}
                  {/* Thumbnails, so offered only where there are images. One guard over the toggle AND the grid:
                      gating only the toggle would leave an open grid with nothing to close it once a mixed
                      folder's last image is deleted. */}
                  {selectedDataset.image_count > 0 && (
                    <>
                      <LabelingGridToggle
                        count={selectedDataset.image_count}
                        open={gridOpen}
                        onToggle={() => setGridOpen((o) => !o)}
                      />
                      {gridOpen && (
                        <DatasetLabelingGrid
                          dataset={dataset}
                          refreshKey={gridRefresh}
                          onCountsChanged={() => void refreshInfo()}
                        />
                      )}
                    </>
                  )}
                  {selectedDataset.caption_count === 0 && !gridOpen && (
                    <p className="text-ui-11 leading-snug text-muted-foreground">
                      No captions yet, so the trigger prompt describes every item.
                    </p>
                  )}
                </>
              )
            )}

            {/* The upload block pads itself; a picked dataset ends in a dense grid, so pad above there. */}
            <ExampleDatasetCards
              examples={pendingExamples}
              busyId={importingId}
              onImport={(ex) => void importExample(ex)}
              className={cn(!uploadMode && "pt-3")}
            />
          </div>

          {/* Trigger first: it describes the dataset, the name just labels the output. */}
          {fullyCaptioned ? (
            <p className="text-ui-11 leading-snug text-muted-foreground">
              Every item in {selectedDataset?.name} has a caption, so no trigger prompt
              is needed.
            </p>
          ) : (
            <div className={fieldClass}>
              <FieldLabel hint="The words you will use later to get this style back. Pick something the base model would not already know.">
                Trigger prompt
              </FieldLabel>
              <Input
                value={instancePrompt}
                placeholder="a photo in mystyle"
                onChange={(e) => setInstancePrompt(e.target.value)}
                className="h-8 text-xs"
              />
            </div>
          )}
          <div className={fieldClass}>
            <FieldLabel hint="What the finished LoRA is called in the Create tab's picker.">
              Adapter name
            </FieldLabel>
            <Input
              value={outputDir}
              placeholder="my-style"
              spellCheck={false}
              onChange={(e) => setOutputDir(e.target.value)}
              className="h-8 text-xs"
            />
          </div>

        </div>
        {/* In its own footer, as Create's Generate is. The scroll mask provides the fade, so the footer
            stays unpainted to avoid dark-mode banding. */}
        <div className="relative z-10 flex shrink-0 justify-center pt-0.5 pb-4 pl-8 pr-8">
          <Button
            type="button"
            className="relative z-10 h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
            onClick={onStart}
            disabled={starting || uploading || running || familyUntrainable}
          >
            {starting
              ? "Starting..."
              : running
                ? "Training in progress"
                : familyUntrainable
                  ? "Not supported on this GPU"
                  : "Start training"}
          </Button>
        </div>
      </div>

      {/* Right: the run area. Before a run: training settings + previous-runs history; during/after: the live view. */}
      {/* Sections carry no card of their own: spacing and a rule separate them. p-1.5 keeps the chart
          cards' outer ring from being clipped. */}
      {/* 40px off the rule, the gutter the settings column has off the page edge. */}
      {/* This pane remains a query container for its own stat and chart breakpoints. */}
      <div className="@container hover-scrollbar relative flex min-w-0 flex-1 flex-col gap-5 pb-7 pl-10 pr-1.5 pt-4 @[50rem]:overflow-y-auto @[50rem]:pt-[42px]">
        {viewRun && !hasRun ? (
          <>
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">
                  Previous run: {viewRun.adapter || viewRun.job_id.slice(0, 8)}
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => setViewRun(null)}
                >
                  Back
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3 @min-[440px]:grid-cols-4">
                <Stat label="Status" value={viewRun.status} />
                <Stat label="Steps" value={`${viewRun.step}/${viewRun.total_steps}`} />
                <Stat
                  label="Avg loss"
                  value={viewRun.avg_loss != null ? viewRun.avg_loss.toFixed(4) : "-"}
                />
                <Stat
                  label="Peak VRAM"
                  value={
                    viewRun.peak_memory_gb != null
                      ? `${viewRun.peak_memory_gb.toFixed(1)} GB`
                      : "-"
                  }
                />
              </div>
              <p className="text-ui-11 text-muted-foreground">
                {viewRun.family ? `${viewRun.family} - ` : ""}
                {viewRun.base_model || ""}
                {viewRun.ended_at
                  ? ` - ${new Date(viewRun.ended_at * 1000).toLocaleString()}`
                  : ""}
              </p>
              {/* Deploy and Resume sit together. Resume stays visible but disabled when the backend says it
                  cannot, with the reason as the tooltip, so its absence is explained rather than hidden. */}
              <div className="flex flex-wrap items-center gap-2">
                {viewRun.saved && viewRun.catalog_path && (
                  <Button
                    type="button"
                    size="sm"
                    onClick={() =>
                      onDeploy?.({
                        baseRepo: deployBaseFor(viewRun.base_model || "", viewRun.family || ""),
                        family: viewRun.family || "",
                        catalogPath: viewRun.catalog_path || "",
                        trigger: viewRun.instance_prompt || "",
                      })
                    }
                  >
                    Deploy to Create
                  </Button>
                )}
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void onResume(viewRun.job_id)}
                  disabled={
                    !viewRun.can_resume || running || resumingJobId === viewRun.job_id
                  }
                  title={
                    viewRun.can_resume
                      ? undefined
                      : viewRun.resume_blocked_reason ||
                        "This run has no checkpoint to continue from."
                  }
                >
                  {resumingJobId === viewRun.job_id
                    ? "Resuming..."
                    : resumeActionLabel(viewRun)}
                </Button>
              </div>
            </div>
            <DiffusionCharts
              lossHistory={viewLossHistory}
              gradNormHistory={viewGradNormHistory}
            />
          </>
        ) : !hasRun ? (
          <>
            <div className="flex flex-col gap-4">
              {/* mb-2 + gap-4 matches the left column's mb-1 + gap-5, so Steps starts level with Model family. */}
              <div className="mb-2 flex items-center justify-between">
                <div className="grid gap-1.5">
                  <span className="flex items-center gap-2 font-heading text-xl font-medium leading-none">
                    <HugeiconsIcon
                      icon={Settings02Icon}
                      className="size-[18px] shrink-0"
                    />
                    Train settings
                  </span>
                  <p className="text-xs leading-snug text-muted-foreground">
                    Hyperparameters for this run.
                  </p>
                </div>
                <span className="text-xs text-muted-foreground">
                  Applied on Start training
                </span>
              </div>
              {trainingSettings}
              <p className="text-ui-11 leading-snug text-muted-foreground">
                Progress and charts take over here once training starts.
              </p>
            </div>

            {prevRuns.length > 0 && (
              <div className="flex flex-col gap-2 border-t border-border/60 pt-4">
                <span className="text-sm font-semibold">Previous runs</span>
                <div className="flex flex-col divide-y divide-border/60">
                  {prevRuns.map((r) => (
                    <button
                      key={r.job_id}
                      type="button"
                      onClick={() => void openPrevRun(r.job_id)}
                      className="flex items-center justify-between gap-3 rounded-md px-1 py-2 text-left text-xs transition-colors hover:bg-muted/40"
                    >
                      <span className="min-w-0 truncate">
                        <span className="font-medium">
                          {r.adapter || r.job_id.slice(0, 8)}
                        </span>
                        <span className="text-muted-foreground">
                          {r.family ? ` ${r.family}` : ""} - {r.step}/{r.total_steps} steps
                          {r.avg_loss != null ? `, avg loss ${r.avg_loss.toFixed(3)}` : ""}
                        </span>
                      </span>
                      <span className="flex shrink-0 items-center gap-2">
                        {r.saved && (
                          <span className="rounded-full bg-primary/15 px-2 py-0.5 text-ui-10 text-primary">
                            adapter saved
                          </span>
                        )}
                        <span className="text-ui-10 uppercase tracking-wide text-muted-foreground">
                          {r.status}
                        </span>
                        <span className="text-ui-10 text-muted-foreground">
                          {r.ended_at ? new Date(r.ended_at * 1000).toLocaleString() : ""}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold capitalize">
                  {status?.status === "completed" ? "Training complete \u{1F389}" : status?.status}
                </span>
                <span className="text-xs text-muted-foreground">
                  {(status?.total_steps ?? 0) > 0
                    ? `${status?.step}/${status?.total_steps} steps`
                    : ""}
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-border">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="grid grid-cols-2 gap-3 @min-[440px]:grid-cols-4">
                <Stat
                  label="Loss"
                  value={status?.loss != null ? status.loss.toFixed(4) : "-"}
                />
                <Stat
                  label="Avg loss"
                  value={status?.avg_loss != null ? status.avg_loss.toFixed(4) : "-"}
                />
                <Stat
                  label="Speed"
                  value={
                    status?.samples_per_second != null
                      ? `${status.samples_per_second.toFixed(2)} img/s`
                      : "-"
                  }
                />
                <Stat
                  label="Peak VRAM"
                  value={
                    status?.peak_memory_gb != null
                      ? `${status.peak_memory_gb.toFixed(1)} GB`
                      : "-"
                  }
                />
              </div>
              {status?.message && (
                <p className="text-ui-11 text-muted-foreground">{status.message}</p>
              )}
              {running && (
                <Button
                  type="button"
                  variant="destructive"
                  className="w-full"
                  onClick={() => setStopDialogOpen(true)}
                  disabled={stopRequested}
                >
                  {stopRequested ? "Stopping..." : "Stop training"}
                </Button>
              )}
              {/* Terminal runs WITHOUT an adapter card still need a way back to the settings, and can be the
                  most worth resuming: a crash leaves no adapter but may have left a periodic checkpoint. */}
              {!running &&
                status &&
                terminalStatuses.includes(status.status) &&
                !completed &&
                !stoppedWithAdapter && (
                  <div className="flex flex-wrap gap-2">
                    {liveRunSummary?.can_resume && status.job_id && (
                      <Button
                        type="button"
                        className="flex-1"
                        onClick={() => void onResume(status.job_id as string)}
                        disabled={resumingJobId === status.job_id}
                      >
                        {resumingJobId === status.job_id
                          ? "Resuming..."
                          : resumeActionLabel(liveRunSummary)}
                      </Button>
                    )}
                    <Button
                      type="button"
                      variant="outline"
                      className="flex-1"
                      onClick={() => setDismissedJobId(status.job_id)}
                    >
                      Back to settings
                    </Button>
                  </div>
                )}
            </div>

            {(completed || stoppedWithAdapter) && (
              <div className="flex flex-col gap-2 border-t border-border/60 pt-4">
                <span className="text-sm font-semibold">
                  {completed ? "Adapter ready" : "Partial adapter saved"}
                </span>
                <p className="text-ui-11 text-muted-foreground">
                  {completed
                    ? "Trained"
                    : "Stopped early; the adapter as of the last finished step was saved"}
                  {status?.family ? ` (${status.family})` : ""}
                  {/* Only an adapter published into the diffusion LoRA catalog reaches the Images surface. A family
                      trained here without that catalog (video) still saves a loadable adapter, so report the
                      file and claim nothing more. */}
                  {status?.catalog_path
                    ? " and added to the LoRA picker."
                    : ". Load it from the path below."}
                  {status?.lora_path && (
                    <span className="mt-1 block break-all">Saved: {status.lora_path}</span>
                  )}
                  {status?.ema_path && (
                    <span className="mt-1 block break-all">EMA adapter: {status.ema_path}</span>
                  )}
                </p>
                <div className="mt-1 flex flex-wrap gap-2">
                  {/* A video run publishes no catalog entry, so there is nothing to deploy. */}
                  {status?.catalog_path && (
                    <Button type="button" size="sm" onClick={onDeployClick}>
                      Deploy to Create
                    </Button>
                  )}
                  {/* Only a run stopped short can be continued. Disabled until the run record lands, with the
                      backend's reason as the tooltip. */}
                  {!completed && status?.job_id && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void onResume(status.job_id as string)}
                      disabled={
                        !liveRunSummary?.can_resume ||
                        running ||
                        resumingJobId === status.job_id
                      }
                      title={
                        liveRunSummary?.can_resume
                          ? undefined
                          : liveRunSummary?.resume_blocked_reason ||
                            status.resume_blocked_reason ||
                            "Saving this run's checkpoint..."
                      }
                    >
                      {resumingJobId === status.job_id
                        ? "Resuming..."
                        : resumeActionLabel(
                            liveRunSummary ?? { checkpoint_step: status.checkpoint_step },
                          )}
                    </Button>
                  )}
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => status && setDismissedJobId(status.job_id)}
                  >
                    Train another
                  </Button>
                </div>
              </div>
            )}

            <DiffusionCharts lossHistory={lossHistory} gradNormHistory={gradNormHistory} />
          </>
        )}
      </div>

      {/* Confirm-stop dialog: Continue / Stop / Stop and save. */}
      <AlertDialog open={stopDialogOpen} onOpenChange={setStopDialogOpen}>
        <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop training?</AlertDialogTitle>
            <AlertDialogDescription>
              Save the adapter trained so far, or discard this run? Either way the current step
              finishes first.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {/* flex-wrap keeps all three buttons visible at narrow widths; a real label on the destructive
              action, since a bare "Stop" read as misaligned. */}
          <AlertDialogFooter className="flex-wrap items-center">
            <AlertDialogCancel>Continue training</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => void onStop(false)}>
              Stop without saving
            </AlertDialogAction>
            <AlertDialogAction onClick={() => void onStop(true)}>
              Stop and save
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={cn("rounded-lg border border-border/60 bg-muted/20 px-2.5 py-1.5")}>
      <div className="text-ui-10 uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-medium tabular-nums">{value}</div>
    </div>
  );
}
