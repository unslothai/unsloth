// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { TrainingSeriesPoint } from "@/features/training";
// eslint-disable-next-line no-restricted-imports -- matches images-page.tsx's token access
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  type DiffusionDatasetExample,
  type DiffusionTrainableFamily,
  type DiffusionTrainingInfo,
  type DiffusionTrainingRunDetail,
  type DiffusionTrainingRunSummary,
  type DiffusionTrainingStatus,
  getDiffusionTrainingInfo,
  getDiffusionTrainingRun,
  getDiffusionTrainingStatus,
  listDiffusionDatasetExamples,
  listDiffusionTrainingRuns,
  startDiffusionTraining,
  stopDiffusionTraining,
  uploadDiffusionDataset,
} from "../api";
import { DatasetLabelingGrid, LabelingGridToggle } from "./dataset-labeling-grid";
import { DatasetShowcase } from "./dataset-showcase";
import { DiffusionCharts } from "./diffusion-charts";
import { ExampleDatasetCards, runExampleImport } from "./example-dataset-cards";

// The families the Train tab can train, in the popularity order the user asked for. This is
// the fallback used when the backend's /info does not yet report families (older backend);
// when it does, its list wins and these labels/notes fill any gaps.
type FamilyPreset = {
  name: string;
  label: string;
  base_repos: string[];
  defaults: { rank: number; lr: number; resolution: number };
  vram_note: string;
  gated?: boolean;
};

const FAMILY_PRESETS: FamilyPreset[] = [
  {
    name: "flux.1",
    label: "FLUX.1-dev (12B)",
    base_repos: ["black-forest-labs/FLUX.1-dev"],
    defaults: { rank: 16, lr: 0.0001, resolution: 512 },
    vram_note: "Gated repo - accept the license on Hugging Face and add your HF token. QLoRA (4-bit).",
    gated: true,
  },
  {
    name: "qwen-image",
    label: "Qwen-Image (20B)",
    base_repos: ["unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "Qwen/Qwen-Image"],
    defaults: { rank: 16, lr: 0.00005, resolution: 512 },
    vram_note: "Largest model - QLoRA (4-bit) on a big GPU. Start at 512px, batch 1.",
  },
  {
    name: "z-image",
    label: "Z-Image-Turbo (6B)",
    base_repos: ["unsloth/Z-Image-Turbo-unsloth-bnb-4bit", "Tongyi-MAI/Z-Image-Turbo"],
    defaults: { rank: 16, lr: 0.0001, resolution: 768 },
    vram_note: "Lightest and fastest to train. bf16 only (fp16 is unstable for this family).",
  },
  {
    name: "sdxl",
    label: "SDXL (U-Net)",
    base_repos: ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo"],
    defaults: { rank: 16, lr: 0.0001, resolution: 1024 },
    vram_note: "The classic text-to-image base. Trains comfortably at 1024px.",
  },
];

const CUSTOM_BASE = "__custom__";
const UPLOAD_DATASET = "__upload__";
// Dataset-select option value prefix for a not-yet-imported example; picking it imports.
const EXAMPLE_PREFIX = "example:";
const DATASET_FILE_ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.txt,.caption,.jsonl";
const selectClass = "h-8 w-full rounded-md border border-input bg-background px-2 text-xs";

// Merge the backend's reported families (if any) over the presets, keeping the preset
// ordering (popularity) and filling labels/notes/defaults the backend omits.
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
      },
      vram_note: r.vram_note || p.vram_note,
      gated: r.gated ?? p.gated,
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
      },
      vram_note: r.vram_note ?? "",
      gated: r.gated ?? false,
    });
  }
  return merged;
}

// A full-page training workspace: left = configure (family, dataset, labeling, settings),
// right = live run (progress, loss/LR charts, completion + deploy). Kept mounted with the
// page so a long run survives Create/Train tab switches; polling is gated on `active`.
export function DiffusionTrainPanel({
  active,
  loadedFamily,
  loadedBaseRepo,
  onTrainingComplete,
  onDeploy,
}: {
  active: boolean;
  // The currently loaded generation model's family / base repo, to preselect a matching
  // training base when it is one we can train.
  loadedFamily?: string | null;
  loadedBaseRepo?: string | null;
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

  const [familyName, setFamilyName] = useState(families[0]?.name ?? "flux.1");
  const family = useMemo(
    () => families.find((f) => f.name === familyName) ?? families[0],
    [families, familyName],
  );
  // The raw backend family record (precision_modes / recommended_precision / supports_compile
  // live only here, not on the preset). Absent on an older backend -> the DiT speed controls
  // fall back to a sensible default list.
  const reportedFamily = useMemo(
    () => info?.families?.find((f) => f.name === familyName),
    [info?.families, familyName],
  );
  // sdxl trains the U-Net in mixed precision (no quantised base), so it uses the
  // mixed_precision control instead of base_precision. Everything else is a DiT family.
  const isDiT = familyName !== "sdxl";
  // The quantised base precisions this family can train in, with a stable fallback when the
  // backend does not report them (older backend, or a preset-only family).
  const precisionModes = useMemo<Array<"nf4" | "bf16" | "int8" | "fp8" | "auto">>(() => {
    const reported = reportedFamily?.precision_modes?.filter(
      (m): m is "nf4" | "bf16" | "int8" | "fp8" =>
        m === "nf4" || m === "bf16" || m === "int8" || m === "fp8",
    );
    if (reported && reported.length > 0) return ["auto", ...reported];
    return ["auto", "nf4", "bf16", "int8", "fp8"];
  }, [reportedFamily?.precision_modes]);
  // Whether to show the torch.compile control. Default on for DiT families when the backend
  // does not say otherwise; sdxl's U-Net path does not expose it here.
  const supportsCompile = isDiT && (reportedFamily?.supports_compile ?? true);

  const [baseChoice, setBaseChoice] = useState<string>(family?.base_repos[0] ?? "");
  const [customBase, setCustomBase] = useState("");

  const [dataset, setDataset] = useState<string>(UPLOAD_DATASET);
  const [uploadName, setUploadName] = useState("my-images");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [gridOpen, setGridOpen] = useState(false);
  const [gridRefresh, setGridRefresh] = useState(0);
  const [examples, setExamples] = useState<DiffusionDatasetExample[]>([]);
  const [importingId, setImportingId] = useState<string | null>(null);

  const [outputDir, setOutputDir] = useState("");
  const [instancePrompt, setInstancePrompt] = useState("");

  const [steps, setSteps] = useState(500);
  // Run length is set in either steps or epochs; the trainer resolves epochs -> steps once
  // the dataset size is known (num_epochs overrides train_steps on the backend).
  const [durationUnit, setDurationUnit] = useState<"steps" | "epochs">("steps");
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(family?.defaults.lr ?? 0.0001);
  const [rank, setRank] = useState(family?.defaults.rank ?? 16);
  const [resolution, setResolution] = useState(family?.defaults.resolution ?? 768);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(1);
  const [seed, setSeed] = useState(42);
  // LR schedule (PR E wired get_scheduler into the loop, so the LR chart reflects this).
  // Warmup only applies to the non-constant schedules; plain "constant" ignores it.
  const [lrScheduler, setLrScheduler] = useState<
    "constant" | "constant_with_warmup" | "cosine" | "linear"
  >("constant");
  const [lrWarmupSteps, setLrWarmupSteps] = useState(0);
  // Gradient checkpointing trades ~20-30% step time for a large activation-VRAM saving.
  const [gradCheckpoint, setGradCheckpoint] = useState(true);
  // sdxl (U-Net) trains in a mixed-precision autocast; the DiT families quantise the frozen
  // base weights instead (base_precision) and ignore this. Both are surfaced in Advanced.
  const [precision, setPrecision] = useState<"bf16" | "fp16" | "no">("bf16");
  // Quantised base precision for DiT families (nf4 QLoRA default, or a speed tier). "auto"
  // lets the backend pick the family's recommended mode. Re-seeded to the family's
  // recommendation on family change (unless the user picked one).
  const [basePrecision, setBasePrecision] = useState<
    "nf4" | "bf16" | "int8" | "fp8" | "auto"
  >("auto");
  // Whether to torch.compile the DiT transformer. "auto" defers to the backend.
  const [compileTransformer, setCompileTransformer] = useState<"off" | "on" | "auto">(
    "auto",
  );
  // Track whether the user hand-edited the numeric settings; if not, a family change
  // re-seeds them from that family's defaults.
  const settingsDirty = useRef(false);
  // Track whether the user hand-picked a base precision; if not, a family change re-seeds it
  // from that family's recommended_precision.
  const precisionDirty = useRef(false);

  const [starting, setStarting] = useState(false);
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);
  // Persisted previous runs (terminal), listed on the idle view; selecting one loads its
  // full record (config + metric logs) and re-plots its charts read-only.
  const [prevRuns, setPrevRuns] = useState<DiffusionTrainingRunSummary[]>([]);
  const [viewRun, setViewRun] = useState<DiffusionTrainingRunDetail | null>(null);
  // The confirm-stop dialog (mirrors the LLM Train tab): Continue / Stop / Stop and save.
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  // Set when the user confirms a stop; the button reads "Stopping..." until the run ends.
  // Clamped to the running state at read time (below) so a fresh run never inherits it.
  const [stopRequestedLocal, setStopRequestedLocal] = useState(false);

  const refreshInfo = useCallback(async (): Promise<DiffusionTrainingInfo | null> => {
    try {
      const i = await getDiffusionTrainingInfo();
      setInfo(i);
      return i;
    } catch {
      return null;
    }
  }, []);

  // On first activation, load the dataset list and preselect a base matching the loaded
  // generation model when it is a trainable family.
  useEffect(() => {
    if (!active) return;
    void refreshInfo().then((i) => {
      setDataset((cur) => {
        if (cur !== UPLOAD_DATASET && i?.datasets.some((d) => d.name === cur)) return cur;
        return i && i.datasets.length > 0 ? i.datasets[0].name : UPLOAD_DATASET;
      });
    });
  }, [active, refreshInfo]);

  // Load the curated example list once (for the dropdown group + the cards). Best-effort:
  // an older backend without the endpoint just yields no examples.
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

  // Examples whose folder is not on disk yet: shown in the dropdown's Examples group and as
  // cards. An example imports into a folder named after its id, so a matching dataset name
  // means it is already imported (and appears as a normal dataset instead).
  const importedNames = useMemo(
    () => new Set((info?.datasets ?? []).map((d) => d.name)),
    [info?.datasets],
  );
  const pendingExamples = useMemo(
    () => examples.filter((ex) => !importedNames.has(ex.id)),
    [examples, importedNames],
  );

  // Import a curated example, then select the resulting folder. Seeds the trigger prompt from
  // the example only when the field is meaningful (the import has no captions of its own).
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

  // If the loaded generation model is a trainable family, jump the family selector to it
  // once (only when the panel first sees a loaded family).
  const seededFromLoaded = useRef(false);
  useEffect(() => {
    if (seededFromLoaded.current) return;
    if (!loadedFamily) return;
    if (families.some((f) => f.name === loadedFamily)) {
      setFamilyName(loadedFamily);
      seededFromLoaded.current = true;
    }
  }, [loadedFamily, families]);

  // Re-seed base + numeric settings from the family's defaults on family change (unless the
  // user edited the numbers). Prefer the loaded base repo when it belongs to this family.
  useEffect(() => {
    if (!family) return;
    const preferLoaded =
      loadedBaseRepo && family.base_repos.includes(loadedBaseRepo)
        ? loadedBaseRepo
        : family.base_repos[0] ?? CUSTOM_BASE;
    setBaseChoice(preferLoaded);
    if (!settingsDirty.current) {
      setLearningRate(family.defaults.lr);
      setRank(family.defaults.rank);
      setResolution(family.defaults.resolution);
    }
    // Re-seed the DiT base precision from the family's recommendation (unless the user picked
    // one). "auto" is always a safe default when the backend has no recommendation.
    if (!precisionDirty.current) {
      const rec = reportedFamily?.recommended_precision;
      setBasePrecision(
        rec === "nf4" || rec === "bf16" || rec === "int8" || rec === "fp8"
          ? rec
          : "auto",
      );
    }
  }, [family, loadedBaseRepo, reportedFamily?.recommended_precision]);

  // The base actually used everywhere (request, deploy, select value). baseChoice can
  // briefly hold another family's repo between a family switch and the reseed effect
  // (or if that effect is skipped); a raw <select value> would then DISPLAY the first
  // option while the request still carried the stale repo -- the user saw FLUX's gated
  // error while another family looked selected. Clamp to the current family's repos.
  const effectiveBase =
    baseChoice === CUSTOM_BASE || (family?.base_repos ?? []).includes(baseChoice)
      ? baseChoice
      : family?.base_repos[0] ?? CUSTOM_BASE;

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

  // "Train another" dismisses the completed run's card locally (the backend keeps the
  // terminal "completed" status until the next start, so we can't rely on it clearing).
  const [dismissedJobId, setDismissedJobId] = useState<string | null>(null);
  const running = Boolean(status?.active) || status?.status === "running";
  const completed =
    status?.status === "completed" && status.job_id !== dismissedJobId;
  // "Stop and save" ends the run as "stopped" WITH a saved partial adapter; it must get
  // the same ready-to-deploy card as a full run (only a no-save stop has nothing to show).
  const stoppedWithAdapter =
    status?.status === "stopped" &&
    Boolean(status?.lora_path) &&
    status.job_id !== dismissedJobId;
  const pct =
    status && status.total_steps > 0
      ? Math.min(100, Math.round((status.step / status.total_steps) * 100))
      : 0;

  // The pending-stop flag only matters while a run is active; clamping at read time (rather
  // than resetting in an effect) means a fresh run never inherits a stale "Stopping..." state.
  const stopRequested = running && stopRequestedLocal;

  // Whether there is a run to show live: running, or ANY terminal run (completed /
  // stopped / error) the user has not dismissed yet. Dismissing must cover every
  // terminal status, or "Train another" after a stop (and any error) would trap the
  // run view with no way back to the settings.
  const terminalStatuses = ["completed", "stopped", "error"];
  const hasRun = Boolean(
    status &&
      status.status !== "idle" &&
      !(terminalStatuses.includes(status.status) && status.job_id === dismissedJobId),
  );

  // Notify the parent exactly once per run that produced an adapter (full completion or
  // stop-and-save) so it rescans the LoRA picker.
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
  // A dataset where every image already ships a caption needs no trigger prompt; hide the
  // field and explain why. Partial/no captions (or upload mode) still show it.
  const fullyCaptioned = Boolean(
    selectedDataset &&
      selectedDataset.image_count > 0 &&
      selectedDataset.caption_count >= selectedDataset.image_count,
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

  // Refresh the previous-runs list whenever the service is not mid-run (on mount and
  // right after a run terminates, when its record has just been persisted).
  useEffect(() => {
    if (!active) return;
    if (status?.status === "running") return;
    let cancelled = false;
    listDiffusionTrainingRuns()
      .then((r) => {
        if (!cancelled) setPrevRuns(r.runs);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
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

  const onUpload = useCallback(async () => {
    const files = Array.from(fileInputRef.current?.files ?? []);
    if (files.length === 0) {
      toast.error("Choose the images to upload first.");
      return;
    }
    const name = uploadName.trim();
    if (!name) {
      toast.error("Give the dataset a folder name, e.g. my-style-photos.");
      return;
    }
    setUploading(true);
    try {
      const res = await uploadDiffusionDataset(name, files);
      toast.success(
        `Uploaded ${res.uploaded} file${res.uploaded === 1 ? "" : "s"} - ` +
          `"${res.name}" now has ${res.image_count} images`,
      );
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshInfo();
      setDataset(res.name);
      setGridRefresh((k) => k + 1);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [uploadName, refreshInfo]);

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
      toast.error("Name the adapter (this becomes its folder under Studio outputs).");
      return;
    }
    if (selectedDataset && selectedDataset.caption_count === 0 && !instancePrompt.trim()) {
      toast.error(
        "These images have no captions - add a trigger prompt so the trainer knows " +
          "what to learn (it becomes the caption for every image).",
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
    // A previous run's confirmed stop must not leak into this run: without the reset the
    // read-time clamp (running && stopRequestedLocal) re-arms the moment the new run goes
    // active, rendering a permanently disabled "Stopping..." button.
    setStopRequestedLocal(false);
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
        // Epochs mode overrides train_steps on the backend, so send num_epochs and omit
        // train_steps (the backend default is unused when num_epochs > 0).
        train_steps: durationUnit === "epochs" ? undefined : steps,
        num_epochs: durationUnit === "epochs" ? epochs : undefined,
        learning_rate: learningRate,
        train_batch_size: batchSize,
        gradient_accumulation_steps: gradAccum,
        seed,
        gradient_checkpointing: gradCheckpoint,
        lr_scheduler: lrScheduler,
        lr_warmup_steps: lrScheduler === "constant" ? 0 : lrWarmupSteps,
        lora_rank: rank,
        mixed_precision: precision,
        // DiT families quantise the base weights (base_precision); sdxl uses mixed_precision
        // above and ignores this. Only send compile for families that support it.
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
    gradCheckpoint,
    lrScheduler,
    lrWarmupSteps,
    rank,
    precision,
    isDiT,
    basePrecision,
    supportsCompile,
    compileTransformer,
    poll,
  ]);

  // Confirm-then-stop, mirroring the LLM Train tab. `save` writes the current adapter before
  // halting ("Stop and save"); false discards it ("Stop"). Closes the dialog and marks the
  // stop as requested so the button reads "Stopping..." until the backend reports it stopped.
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

  const onDeployClick = useCallback(() => {
    if (!status?.catalog_path) {
      toast.error("The trained adapter is not available yet.");
      return;
    }
    const baseRepo = status.base_model || (effectiveBase === CUSTOM_BASE ? customBase : effectiveBase);
    if (!baseRepo) {
      toast.error("Could not determine the base model to load for this adapter.");
      return;
    }
    onDeploy?.({
      baseRepo,
      family: status.family || family?.name || "",
      catalogPath: status.catalog_path,
      trigger: instancePrompt.trim(),
    });
  }, [status, baseChoice, customBase, family, instancePrompt, onDeploy]);

  const numberField = (
    label: string,
    value: number,
    set: (n: number) => void,
    fallback: number,
    extra?: { min?: number; step?: number },
  ) => (
    <div className="grid gap-1.5">
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        min={extra?.min ?? 1}
        step={extra?.step}
        value={value}
        onChange={(e) => {
          settingsDirty.current = true;
          set(Number(e.target.value) || fallback);
        }}
        className="h-8 text-xs"
      />
    </div>
  );

  // Run length: a number paired with a compact unit select (Steps / Epochs). Epochs mode
  // trains for that many full passes over the dataset; the backend resolves it to steps.
  const durationField = (
    <div className="grid gap-1.5">
      <Label className="text-xs">{durationUnit === "epochs" ? "Epochs" : "Steps"}</Label>
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
        <select
          value={durationUnit}
          onChange={(e) => {
            settingsDirty.current = true;
            setDurationUnit(e.target.value as "steps" | "epochs");
          }}
          className="h-8 w-24 rounded-md border border-input bg-background px-2 text-xs"
          aria-label="Run length unit"
        >
          <option value="steps">Steps</option>
          <option value="epochs">Epochs</option>
        </select>
      </div>
    </div>
  );

  const precisionLabel = (m: "nf4" | "bf16" | "int8" | "fp8" | "auto"): string => {
    if (m === "auto") return "Auto (recommended)";
    if (m === "nf4") return "nf4 (4-bit QLoRA, lowest VRAM)";
    if (m === "bf16") return "bf16 (fastest, most VRAM)";
    if (m === "int8") return "int8 (8-bit)";
    return "fp8 (experimental)";
  };

  // The training settings, shown as the run area's MAIN content before a run starts
  // (settings are set once, up front); once training starts the run view (progress +
  // charts) replaces them. Laid out as a wide grid for the center column.
  const trainingSettings = (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 lg:grid-cols-3">
        {durationField}
        {numberField("LoRA rank", rank, setRank, 1)}
        {numberField("Resolution", resolution, setResolution, 512, { min: 64, step: 64 })}
        {numberField("Batch", batchSize, setBatchSize, 1)}
        {numberField("Grad accumulation", gradAccum, setGradAccum, 1)}
        {numberField("Seed", seed, setSeed, 42, { min: 0 })}
      </div>

      <div className="grid grid-cols-2 items-start gap-x-6 gap-y-4 lg:grid-cols-3">
        {numberField("Learning rate", learningRate, setLearningRate, 0.0001, {
          min: 0,
          step: 0.00001,
        })}
        <div className="grid gap-1.5">
          <Label className="text-xs">LR schedule</Label>
          <select
            value={lrScheduler}
            onChange={(e) => setLrScheduler(e.target.value as typeof lrScheduler)}
            className={selectClass}
            aria-label="LR schedule"
          >
            <option value="constant">Constant</option>
            <option value="constant_with_warmup">Constant + warmup</option>
            <option value="cosine">Cosine decay</option>
            <option value="linear">Linear decay</option>
          </select>
          <p className="text-[11px] leading-snug text-muted-foreground">
            How the learning rate evolves over the run (shown live in the LR chart).
          </p>
        </div>
        {lrScheduler !== "constant" &&
          numberField("Warmup steps", lrWarmupSteps, setLrWarmupSteps, 0, { min: 0 })}
      </div>

      <div className="grid grid-cols-2 items-start gap-x-6 gap-y-4 lg:grid-cols-3">
        <div className="grid gap-1.5">
          <Label className="text-xs">Gradient checkpointing</Label>
          <select
            value={gradCheckpoint ? "on" : "off"}
            onChange={(e) => setGradCheckpoint(e.target.value === "on")}
            className={selectClass}
            aria-label="Gradient checkpointing"
          >
            <option value="on">On (less VRAM)</option>
            <option value="off">Off (faster steps)</option>
          </select>
          <p className="text-[11px] leading-snug text-muted-foreground">
            Recomputes activations in the backward pass: a large VRAM saving for a modest
            per-step slowdown.
          </p>
        </div>

        {isDiT ? (
          <>
            <div className="grid gap-1.5">
              <Label className="text-xs">Base precision</Label>
              <select
                value={basePrecision}
                onChange={(e) => {
                  precisionDirty.current = true;
                  setBasePrecision(e.target.value as typeof basePrecision);
                }}
                className={selectClass}
                aria-label="Base precision"
              >
                {precisionModes.map((m) => (
                  <option key={m} value={m}>
                    {precisionLabel(m)}
                  </option>
                ))}
              </select>
              <p className="text-[11px] leading-snug text-muted-foreground">
                How the frozen base weights are quantised. nf4 (4-bit) uses the least VRAM;
                bf16 is fastest but needs the most. Auto picks this family&apos;s recommended
                mode.
              </p>
            </div>
            {supportsCompile && (
              <div className="grid gap-1.5">
                <Label className="text-xs">Compile transformer</Label>
                <select
                  value={compileTransformer}
                  onChange={(e) =>
                    setCompileTransformer(e.target.value as typeof compileTransformer)
                  }
                  className={selectClass}
                  aria-label="Compile transformer"
                >
                  <option value="auto">Auto</option>
                  <option value="on">On (faster after warmup)</option>
                  <option value="off">Off</option>
                </select>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  torch.compile the transformer. Adds a one-time warmup, then speeds up each
                  step.
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="grid gap-1.5">
            <Label className="text-xs">Precision</Label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value as "bf16" | "fp16" | "no")}
              className={selectClass}
              aria-label="Precision"
            >
              <option value="bf16">bf16 (default)</option>
              <option value="fp16">fp16 (older GPUs)</option>
              <option value="no">fp32 (no mixed)</option>
            </select>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Mixed-precision autocast for the U-Net. bf16 suits modern GPUs.
            </p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
      {/* Left: configure */}
      <div className="bg-card corner-squircle flex w-[380px] min-w-0 shrink-0 flex-col gap-4 overflow-y-auto overflow-x-hidden rounded-3xl p-5 ring-1 ring-foreground/10">
        <div>
          <h2 className="text-sm font-semibold">Train a LoRA</h2>
          <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
            Teach an image model a style, character, or subject from your own images. The
            finished adapter shows up in the Create tab&apos;s LoRA picker.
          </p>
        </div>

        {/* Family + base */}
        <div className="grid gap-1.5">
          <Label className="text-xs">Model family</Label>
          <select
            value={familyName}
            onChange={(e) => setFamilyName(e.target.value)}
            className={selectClass}
            aria-label="Model family"
          >
            {families.map((f) => (
              <option key={f.name} value={f.name}>
                {f.label}
              </option>
            ))}
          </select>
          {family?.vram_note && (
            <p className="text-[11px] leading-snug text-muted-foreground">{family.vram_note}</p>
          )}
        </div>

        <div className="grid gap-1.5">
          <Label className="text-xs">Base model</Label>
          <select
            value={effectiveBase}
            onChange={(e) => setBaseChoice(e.target.value)}
            className={selectClass}
            aria-label="Base model"
          >
            {(family?.base_repos ?? []).map((repo) => (
              <option key={repo} value={repo}>
                {repo}
              </option>
            ))}
            <option value={CUSTOM_BASE}>Custom repo or local path...</option>
          </select>
          {effectiveBase === CUSTOM_BASE && (
            <Input
              value={customBase}
              placeholder="my-org/my-base or /path/to/pipeline"
              spellCheck={false}
              onChange={(e) => setCustomBase(e.target.value)}
              className="h-8 text-xs"
            />
          )}
        </div>

        {/* Dataset */}
        <div className="grid gap-1.5">
          <Label className="text-xs">Training images</Label>
          <select
            value={dataset}
            onChange={(e) => {
              const v = e.target.value;
              if (v.startsWith(EXAMPLE_PREFIX)) {
                const ex = pendingExamples.find((x) => x.id === v.slice(EXAMPLE_PREFIX.length));
                if (ex) void importExample(ex);
                return; // controlled select snaps back to the current dataset while importing
              }
              setDataset(v);
              setGridOpen(false);
            }}
            className={selectClass}
            aria-label="Training images"
            disabled={importingId !== null}
          >
            {(info?.datasets ?? []).map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.image_count} image{d.image_count === 1 ? "" : "s"}
                {d.caption_count > 0 ? `, ${d.caption_count} captions` : ""})
              </option>
            ))}
            {pendingExamples.length > 0 && (
              <optgroup label="Examples (one-click import)">
                {pendingExamples.map((ex) => (
                  <option key={ex.id} value={`${EXAMPLE_PREFIX}${ex.id}`}>
                    {ex.label} ({ex.image_cap} images, {ex.license})
                  </option>
                ))}
              </optgroup>
            )}
            <option value={UPLOAD_DATASET}>Upload new images...</option>
          </select>
          {importingId && (
            <p className="text-[11px] text-muted-foreground">
              Importing {examples.find((e) => e.id === importingId)?.label ?? "example"}...
            </p>
          )}

          {dataset === UPLOAD_DATASET ? (
            <div className="grid gap-1.5 rounded-md border border-dashed border-border p-2">
              <Input
                value={uploadName}
                placeholder="my-style-photos"
                spellCheck={false}
                onChange={(e) => setUploadName(e.target.value)}
                className="h-8 text-xs"
                aria-label="New dataset name"
              />
              <div className="flex items-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept={DATASET_FILE_ACCEPT}
                  className="min-w-0 flex-1 text-xs file:mr-2 file:rounded-md file:border-0 file:bg-muted file:px-2 file:py-1 file:text-xs"
                  aria-label="Training image files"
                />
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className="h-8 shrink-0"
                  onClick={onUpload}
                  disabled={uploading}
                >
                  {uploading ? "Uploading..." : "Upload"}
                </Button>
              </div>
              <p className="text-[11px] text-muted-foreground">
                10-50 images work well. Optional captions: a .txt per image or a
                metadata.jsonl; without them the trigger prompt below captions every image.
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
                  />
                )}
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
                {selectedDataset.caption_count === 0 && !gridOpen && (
                  <p className="text-[11px] text-muted-foreground">
                    No caption files - the trigger prompt below captions every image, or
                    open Review captions to label them.
                  </p>
                )}
              </>
            )
          )}

          <ExampleDatasetCards
            examples={pendingExamples}
            busyId={importingId}
            onImport={(ex) => void importExample(ex)}
          />
        </div>

        {/* Trigger + adapter name (trigger first: it describes the dataset, the name
            just labels the output) */}
        {fullyCaptioned ? (
          <p className="text-[11px] leading-snug text-muted-foreground">
            All {selectedDataset?.image_count} images have captions - no trigger prompt needed.
            The style applies to any prompt after training.
          </p>
        ) : (
          <div className="grid gap-1.5">
            <Label className="text-xs">
              Trigger prompt (how you&apos;ll invoke the style later)
            </Label>
            <Input
              value={instancePrompt}
              placeholder="a photo in SKS style"
              onChange={(e) => setInstancePrompt(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
        )}
        <div className="grid gap-1.5">
          <Label className="text-xs">Adapter name</Label>
          <Input
            value={outputDir}
            placeholder="my-style-lora"
            spellCheck={false}
            onChange={(e) => setOutputDir(e.target.value)}
            className="h-8 text-xs"
          />
        </div>

        {/* Start lives here; Stop lives in the run card next to the live stats. */}
        <div className="mt-auto pt-2">
          <Button
            type="button"
            className="w-full"
            onClick={onStart}
            disabled={starting || uploading || running}
          >
            {starting ? "Starting..." : running ? "Training in progress" : "Start training"}
          </Button>
        </div>
      </div>

      {/* Right: the run area. Before a run it shows the training settings (set once, up
          front) plus the previous-runs history; during/after a run the live view takes
          over (progress with Stop, then the saved-adapter card ABOVE the charts).
          Selecting a previous run re-plots its persisted logs read-only. */}
      <div className="relative flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto">
        {viewRun && !hasRun ? (
          <>
            <div className="bg-card corner-squircle flex flex-col gap-3 rounded-3xl p-5 ring-1 ring-foreground/10">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">
                  Previous run: {viewRun.adapter || viewRun.job_id.slice(0, 8)}
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  onClick={() => setViewRun(null)}
                >
                  Back
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
              <p className="text-[11px] text-muted-foreground">
                {viewRun.family ? `${viewRun.family} - ` : ""}
                {viewRun.base_model || ""}
                {viewRun.ended_at
                  ? ` - ${new Date(viewRun.ended_at * 1000).toLocaleString()}`
                  : ""}
              </p>
              {viewRun.saved && viewRun.catalog_path && (
                <div>
                  <Button
                    type="button"
                    size="sm"
                    onClick={() =>
                      onDeploy?.({
                        baseRepo: viewRun.base_model || "",
                        family: viewRun.family || "",
                        catalogPath: viewRun.catalog_path || "",
                        trigger: viewRun.instance_prompt || "",
                      })
                    }
                  >
                    Deploy to Create
                  </Button>
                </div>
              )}
            </div>
            <DiffusionCharts
              lossHistory={viewLossHistory}
              gradNormHistory={viewGradNormHistory}
            />
          </>
        ) : !hasRun ? (
          <>
            <div className="bg-card corner-squircle flex flex-col gap-4 rounded-3xl p-5 ring-1 ring-foreground/10">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-sm font-semibold">
                  <HugeiconsIcon icon={Settings02Icon} className="size-4" />
                  Training settings
                </span>
                <span className="text-xs text-muted-foreground">
                  Applied when you press Start training
                </span>
              </div>
              {trainingSettings}
              <p className="text-[11px] leading-snug text-muted-foreground">
                Once training starts, live progress and the Training Loss / Gradient Norm
                charts take over this area.
              </p>
            </div>

            {prevRuns.length > 0 && (
              <div className="bg-card corner-squircle flex flex-col gap-2 rounded-3xl p-5 ring-1 ring-foreground/10">
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
                          <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[10px] text-primary">
                            adapter saved
                          </span>
                        )}
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                          {r.status}
                        </span>
                        <span className="text-[10px] text-muted-foreground">
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
            <div className="bg-card corner-squircle flex flex-col gap-3 rounded-3xl p-5 ring-1 ring-foreground/10">
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
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
                <p className="text-[11px] text-muted-foreground">{status.message}</p>
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
              {/* Terminal runs WITHOUT an adapter card (error, or a stop that discarded
                  the run) still need a way back to the settings. */}
              {!running &&
                status &&
                terminalStatuses.includes(status.status) &&
                !completed &&
                !stoppedWithAdapter && (
                  <Button
                    type="button"
                    variant="secondary"
                    className="w-full"
                    onClick={() => setDismissedJobId(status.job_id)}
                  >
                    Back to settings
                  </Button>
                )}
            </div>

            {(completed || stoppedWithAdapter) && (
              <div className="bg-card corner-squircle flex flex-col gap-2 rounded-3xl p-5 ring-1 ring-foreground/10">
                <span className="text-sm font-semibold">
                  {completed ? "Adapter ready" : "Partial adapter saved"}
                </span>
                <p className="text-[11px] text-muted-foreground">
                  {completed
                    ? "Trained"
                    : "Stopped early; the adapter as of the last finished step was saved"}
                  {status?.family ? ` (${status.family})` : ""} and added to the LoRA picker.
                  {status?.lora_path && (
                    <span className="mt-1 block break-all">Saved: {status.lora_path}</span>
                  )}
                </p>
                <div className="mt-1 flex gap-2">
                  <Button type="button" size="sm" onClick={onDeployClick}>
                    Deploy to Create
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
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

      {/* Confirm-stop dialog (mirrors the LLM Train tab): Continue / Stop / Stop and save. */}
      <AlertDialog open={stopDialogOpen} onOpenChange={setStopDialogOpen}>
        <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop training?</AlertDialogTitle>
            <AlertDialogDescription>
              Save the adapter trained so far, or discard this run? Either way the current step
              finishes first.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {/* flex-wrap keeps all three buttons visible when the sm:flex-row row is wider than
              the dialog at narrow widths (down to ~480px); it wraps instead of clipping the
              last button past the right edge. items-center + a real label on the destructive
              action: a bare "Stop" rendered as a stubby pill between two wide ones and read
              as misaligned. */}
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
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-medium tabular-nums">{value}</div>
    </div>
  );
}
