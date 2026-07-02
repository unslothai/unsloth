// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { TrainingSeriesPoint } from "@/features/training";
// eslint-disable-next-line no-restricted-imports -- matches images-page.tsx's token access
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  type DiffusionTrainableFamily,
  type DiffusionTrainingInfo,
  type DiffusionTrainingStatus,
  getDiffusionTrainingInfo,
  getDiffusionTrainingStatus,
  startDiffusionTraining,
  stopDiffusionTraining,
  uploadDiffusionDataset,
} from "../api";
import { DatasetLabelingGrid, LabelingGridToggle } from "./dataset-labeling-grid";
import { DatasetShowcase } from "./dataset-showcase";
import { DiffusionCharts } from "./diffusion-charts";
import { ExampleDatasetCards } from "./example-dataset-cards";

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

  const [baseChoice, setBaseChoice] = useState<string>(family?.base_repos[0] ?? "");
  const [customBase, setCustomBase] = useState("");

  const [dataset, setDataset] = useState<string>(UPLOAD_DATASET);
  const [uploadName, setUploadName] = useState("my-images");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [gridOpen, setGridOpen] = useState(false);
  const [gridRefresh, setGridRefresh] = useState(0);

  const [outputDir, setOutputDir] = useState("");
  const [instancePrompt, setInstancePrompt] = useState("");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [steps, setSteps] = useState(500);
  const [learningRate, setLearningRate] = useState(family?.defaults.lr ?? 0.0001);
  const [rank, setRank] = useState(family?.defaults.rank ?? 16);
  const [resolution, setResolution] = useState(family?.defaults.resolution ?? 768);
  const [batchSize, setBatchSize] = useState(1);
  const [precision, setPrecision] = useState<"bf16" | "fp16" | "no">("bf16");
  // Track whether the user hand-edited the numeric settings; if not, a family change
  // re-seeds them from that family's defaults.
  const settingsDirty = useRef(false);

  const [starting, setStarting] = useState(false);
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);

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
  }, [family, loadedBaseRepo]);

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
  const pct =
    status && status.total_steps > 0
      ? Math.min(100, Math.round((status.step / status.total_steps) * 100))
      : 0;

  // Notify the parent exactly once per completed run so it rescans the LoRA picker.
  const notifiedComplete = useRef(false);
  useEffect(() => {
    if (status?.status === "completed" && !notifiedComplete.current) {
      notifiedComplete.current = true;
      onTrainingComplete?.();
    } else if (status?.status === "running" && notifiedComplete.current) {
      notifiedComplete.current = false;
    }
  }, [status?.status, onTrainingComplete]);

  const selectedDataset =
    dataset !== UPLOAD_DATASET ? info?.datasets.find((d) => d.name === dataset) : undefined;

  // Map the backend's paired history arrays into the chart component's {step,value} series.
  const lossHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = status?.metric_history;
    if (!h) return [];
    return h.steps.map((step, i) => ({ step, value: h.loss[i] })).filter((p) => p.value != null);
  }, [status?.metric_history]);
  const lrHistory: TrainingSeriesPoint[] = useMemo(() => {
    const h = status?.metric_history;
    if (!h) return [];
    return h.steps
      .map((step, i) => ({ step, value: h.lr[i] }))
      .filter((p): p is TrainingSeriesPoint => p.value != null);
  }, [status?.metric_history]);

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
    const baseModel = (baseChoice === CUSTOM_BASE ? customBase : baseChoice).trim();
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
    if (steps < 1) return toast.error("Steps must be at least 1.");
    if (rank < 1) return toast.error("LoRA rank must be at least 1.");
    if (resolution < 64 || resolution % 8 !== 0) {
      return toast.error("Resolution must be a multiple of 8 and at least 64.");
    }
    if (batchSize < 1) return toast.error("Batch size must be at least 1.");
    if (learningRate <= 0) return toast.error("Learning rate must be greater than 0.");
    setStarting(true);
    try {
      await startDiffusionTraining({
        base_model: baseModel,
        model_family: family?.name,
        data_dir: dataset,
        output_dir: outputDir.trim(),
        instance_prompt: instancePrompt.trim() || undefined,
        resolution,
        train_steps: steps,
        learning_rate: learningRate,
        train_batch_size: batchSize,
        lora_rank: rank,
        mixed_precision: precision,
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
    baseChoice,
    customBase,
    family,
    dataset,
    selectedDataset,
    outputDir,
    instancePrompt,
    resolution,
    steps,
    learningRate,
    batchSize,
    rank,
    precision,
    poll,
  ]);

  const onStop = useCallback(async () => {
    try {
      await stopDiffusionTraining();
      toast.success("Stop requested; finishing the current step.");
      void poll();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to stop training");
    }
  }, [poll]);

  const onDeployClick = useCallback(() => {
    if (!status?.catalog_path) {
      toast.error("The trained adapter is not available yet.");
      return;
    }
    const baseRepo = status.base_model || (baseChoice === CUSTOM_BASE ? customBase : baseChoice);
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

  return (
    <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
      {/* Left: configure */}
      <div className="bg-card corner-squircle flex w-[380px] shrink-0 flex-col gap-4 overflow-y-auto rounded-3xl p-5 ring-1 ring-foreground/10">
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
            value={baseChoice}
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
          {baseChoice === CUSTOM_BASE && (
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
              setDataset(e.target.value);
              setGridOpen(false);
            }}
            className={selectClass}
            aria-label="Training images"
          >
            {(info?.datasets ?? []).map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.image_count} image{d.image_count === 1 ? "" : "s"}
                {d.caption_count > 0 ? `, ${d.caption_count} captions` : ""})
              </option>
            ))}
            <option value={UPLOAD_DATASET}>Upload new images...</option>
          </select>

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
            onImported={(res, ex) => {
              void refreshInfo();
              setDataset(res.name);
              setGridRefresh((k) => k + 1);
              if (ex.suggested_trigger && !instancePrompt.trim()) {
                setInstancePrompt(ex.suggested_trigger);
              }
            }}
          />
        </div>

        {/* Adapter name + trigger */}
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
        <div className="grid gap-1.5">
          <Label className="text-xs">Trigger prompt (how you&apos;ll invoke the style later)</Label>
          <Input
            value={instancePrompt}
            placeholder="a photo in SKS style"
            onChange={(e) => setInstancePrompt(e.target.value)}
            className="h-8 text-xs"
          />
        </div>

        {/* Collapsed training settings */}
        <button
          type="button"
          className="w-fit text-xs text-muted-foreground underline-offset-2 hover:underline"
          onClick={() => setShowAdvanced((s) => !s)}
          aria-expanded={showAdvanced}
        >
          {showAdvanced
            ? "Hide training settings"
            : "Training settings (defaults suit a first run)"}
        </button>
        {showAdvanced && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {numberField("Steps", steps, setSteps, 1)}
              {numberField("LoRA rank", rank, setRank, 1)}
              {numberField("Resolution", resolution, setResolution, 512, { min: 64, step: 64 })}
              {numberField("Batch", batchSize, setBatchSize, 1)}
            </div>
            <div className="grid grid-cols-2 gap-3">
              {numberField("Learning rate", learningRate, setLearningRate, 0.0001, {
                min: 0,
                step: 0.00001,
              })}
              <div className="grid gap-1.5">
                <Label className="text-xs">Precision</Label>
                <select
                  value={precision}
                  onChange={(e) => setPrecision(e.target.value as "bf16" | "fp16" | "no")}
                  className={selectClass}
                >
                  <option value="bf16">bf16 (default)</option>
                  <option value="fp16">fp16 (older GPUs)</option>
                  <option value="no">fp32 (no mixed)</option>
                </select>
              </div>
            </div>
          </>
        )}

        <div className="mt-auto pt-2">
          {running ? (
            <Button type="button" variant="destructive" className="w-full" onClick={onStop}>
              Stop training
            </Button>
          ) : (
            <Button
              type="button"
              className="w-full"
              onClick={onStart}
              disabled={starting || uploading}
            >
              {starting ? "Starting..." : "Start training"}
            </Button>
          )}
        </div>
      </div>

      {/* Right: run view */}
      <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto">
        {status &&
        status.status !== "idle" &&
        !(status.status === "completed" && status.job_id === dismissedJobId) ? (
          <>
            <div className="bg-card corner-squircle flex flex-col gap-3 rounded-3xl p-5 ring-1 ring-foreground/10">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold capitalize">{status.status}</span>
                <span className="text-xs text-muted-foreground">
                  {status.total_steps > 0 ? `${status.step}/${status.total_steps} steps` : ""}
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-border">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Loss" value={status.loss != null ? status.loss.toFixed(4) : "-"} />
                <Stat
                  label="Avg loss"
                  value={status.avg_loss != null ? status.avg_loss.toFixed(4) : "-"}
                />
                <Stat
                  label="Speed"
                  value={
                    status.samples_per_second != null
                      ? `${status.samples_per_second.toFixed(2)} img/s`
                      : "-"
                  }
                />
                <Stat
                  label="Peak VRAM"
                  value={
                    status.peak_memory_gb != null ? `${status.peak_memory_gb.toFixed(1)} GB` : "-"
                  }
                />
              </div>
              {status.message && (
                <p className="text-[11px] text-muted-foreground">{status.message}</p>
              )}
            </div>

            <DiffusionCharts lossHistory={lossHistory} lrHistory={lrHistory} />

            {completed && (
              <div className="bg-card corner-squircle flex flex-col gap-2 rounded-3xl p-5 ring-1 ring-foreground/10">
                <span className="text-sm font-semibold">Adapter ready</span>
                <p className="text-[11px] text-muted-foreground">
                  Trained{status.family ? ` (${status.family})` : ""} and added to the LoRA
                  picker.
                  {status.lora_path && (
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
                    onClick={() => setDismissedJobId(status.job_id)}
                  >
                    Train another
                  </Button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="max-w-sm text-center">
              <p className="text-sm font-medium">No training run yet</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Pick a family and dataset on the left, then Start training. The loss chart and
                progress appear here live.
              </p>
            </div>
          </div>
        )}
      </div>
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
