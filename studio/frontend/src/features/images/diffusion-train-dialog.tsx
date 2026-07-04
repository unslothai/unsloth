// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { toast } from "@/lib/toast";

import {
  type DiffusionTrainingInfo,
  type DiffusionTrainingStatus,
  getDiffusionTrainingInfo,
  getDiffusionTrainingStatus,
  startDiffusionTraining,
  stopDiffusionTraining,
  uploadDiffusionDataset,
} from "./api";

// The two official SDXL bases the backend allowlists for non-GGUF loads. Everything the
// dropdown offers is trainable; "custom" is the escape hatch for local SDXL checkpoints.
const SDXL_BASES: Array<{ id: string; label: string }> = [
  { id: "stabilityai/stable-diffusion-xl-base-1.0", label: "SDXL Base 1.0 (best quality)" },
  { id: "stabilityai/sdxl-turbo", label: "SDXL Turbo (fast, good for quick tests)" },
];
const CUSTOM_BASE = "__custom__";
const UPLOAD_DATASET = "__upload__";
const DATASET_FILE_ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.txt,.caption,.jsonl";

const selectClass =
  "h-8 w-full rounded-md border border-input bg-background px-2 text-xs";

// A self-contained "Train an SDXL LoRA" dialog. It posts to /api/train/diffusion/start
// and polls /status while open, so it never blocks the page and works whether or not a
// model is loaded for generation. Only SDXL is trainable today; the backend refuses
// known non-SDXL picks instantly, and the base-model dropdown keeps users on safe picks.
export function DiffusionTrainDialog({
  open,
  onOpenChange,
  defaultBaseModel,
  onTrainingComplete,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  defaultBaseModel?: string;
  // Called once when a run finishes so the page can rescan the LoRA picker.
  onTrainingComplete?: () => void;
}) {
  const [baseChoice, setBaseChoice] = useState(SDXL_BASES[0].id);
  const [customBase, setCustomBase] = useState("");
  const [info, setInfo] = useState<DiffusionTrainingInfo | null>(null);
  const [dataset, setDataset] = useState<string>(UPLOAD_DATASET);
  const [uploadName, setUploadName] = useState("my-images");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [outputDir, setOutputDir] = useState("");
  const [instancePrompt, setInstancePrompt] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [steps, setSteps] = useState(500);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [rank, setRank] = useState(16);
  const [resolution, setResolution] = useState(1024);
  const [batchSize, setBatchSize] = useState(1);
  const [precision, setPrecision] = useState<"bf16" | "fp16" | "no">("bf16");
  const [starting, setStarting] = useState(false);
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);

  // The dialog stays mounted (ImagesPage is keep-alive), so seed per-open state here:
  // the base-model choice from the currently loaded SDXL pipeline (when there is one),
  // and the dataset list from the backend.
  const refreshInfo = useCallback(async (): Promise<DiffusionTrainingInfo | null> => {
    try {
      const i = await getDiffusionTrainingInfo();
      setInfo(i);
      return i;
    } catch {
      return null; // older backends: keep the upload-only flow usable
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    if (defaultBaseModel) {
      setBaseChoice(defaultBaseModel);
    }
    void refreshInfo().then((i) => {
      // Preselect the only dataset, or the freshest-looking state: with no datasets
      // yet, the picker sits on "Upload new images".
      setDataset((cur) => {
        if (cur !== UPLOAD_DATASET && i?.datasets.some((d) => d.name === cur)) return cur;
        return i && i.datasets.length > 0 ? i.datasets[0].name : UPLOAD_DATASET;
      });
    });
  }, [open, defaultBaseModel, refreshInfo]);

  const poll = useCallback(async () => {
    try {
      setStatus(await getDiffusionTrainingStatus());
    } catch {
      // Best-effort; a failed poll should not surface an error while the dialog is open.
    }
  }, []);

  // Poll status only while the dialog is open.
  useEffect(() => {
    if (!open) return;
    void poll();
    const id = window.setInterval(() => void poll(), 1500);
    return () => window.clearInterval(id);
  }, [open, poll]);

  const active = Boolean(status?.active) || status?.status === "running";
  const completed = status?.status === "completed";
  const pct =
    status && status.total_steps > 0
      ? Math.min(100, Math.round((status.step / status.total_steps) * 100))
      : 0;

  // Notify the parent exactly once when a run reaches "completed", so it can rescan the
  // LoRA picker (a LoRA trained while a model is loaded is otherwise invisible until a
  // model swap re-runs the discovery effect).
  const [notifiedComplete, setNotifiedComplete] = useState(false);
  useEffect(() => {
    if (status?.status === "completed" && !notifiedComplete) {
      setNotifiedComplete(true);
      onTrainingComplete?.();
    } else if (status?.status === "running" && notifiedComplete) {
      setNotifiedComplete(false); // arm again for the next run
    }
  }, [status?.status, notifiedComplete, onTrainingComplete]);

  const selectedDataset =
    dataset !== UPLOAD_DATASET ? info?.datasets.find((d) => d.name === dataset) : undefined;

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
    // Mirror the backend's numeric validation so obvious mistakes are caught before the
    // request (the backend returns 400 for these; catching here gives a clearer message).
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
        data_dir: dataset,
        output_dir: outputDir.trim(),
        instance_prompt: instancePrompt.trim() || undefined,
        resolution,
        train_steps: steps,
        learning_rate: learningRate,
        train_batch_size: batchSize,
        lora_rank: rank,
        mixed_precision: precision,
        // Forward the saved Hub token so a gated/private SDXL base can be trained (the
        // image load flow already sends it, so a model you can load, you can also train).
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] max-w-lg flex-col">
        <DialogHeader>
          <DialogTitle>Train an SDXL LoRA</DialogTitle>
          <DialogDescription>
            Teach SDXL a style, character, or subject from your own images. The finished
            adapter shows up in this page&apos;s LoRA picker. Only SDXL can be trained for
            now - FLUX, Qwen-Image and Z-Image load LoRAs but can&apos;t train them yet.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 overflow-y-auto py-2 pr-1">
          {/* 1. Base model: a constrained dropdown instead of free text, so the "SDXL
              only" rule is embodied by the control. Custom stays available for local
              SDXL checkpoints; a known non-SDXL pick is refused instantly by the API. */}
          <div className="grid gap-1.5">
            <Label className="text-xs">Base model to train on</Label>
            <select
              value={baseChoice}
              onChange={(e) => setBaseChoice(e.target.value)}
              className={selectClass}
            >
              {SDXL_BASES.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.label}
                </option>
              ))}
              {defaultBaseModel && !SDXL_BASES.some((b) => b.id === defaultBaseModel) && (
                <option value={defaultBaseModel}>Loaded model: {defaultBaseModel}</option>
              )}
              <option value={CUSTOM_BASE}>Custom SDXL repo or local path...</option>
            </select>
            {baseChoice === CUSTOM_BASE && (
              <Input
                value={customBase}
                placeholder="my-org/my-sdxl-finetune or /path/to/sdxl-pipeline"
                spellCheck={false}
                onChange={(e) => setCustomBase(e.target.value)}
                className="h-8 text-xs"
              />
            )}
          </div>

          {/* 2. Training images: pick an existing dataset folder or upload straight from
              the browser - no shell access or Studio-home knowledge needed. */}
          <div className="grid gap-1.5">
            <Label className="text-xs">Training images</Label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className={selectClass}
            >
              {(info?.datasets ?? []).map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.image_count} image{d.image_count === 1 ? "" : "s"}
                  {d.caption_count > 0 ? `, ${d.caption_count} captions` : ""})
                </option>
              ))}
              <option value={UPLOAD_DATASET}>Upload new images...</option>
            </select>
            {dataset === UPLOAD_DATASET && (
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
                    aria-label="Training images"
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
                  10-50 images work well. Optional captions: a .txt per image (same
                  filename) or a metadata.jsonl; without them the trigger prompt below
                  captions every image. You can upload more into the same name later.
                </p>
              </div>
            )}
            {selectedDataset && selectedDataset.caption_count === 0 && (
              <p className="text-[11px] text-muted-foreground">
                No caption files in this dataset - the trigger prompt below will be used
                as the caption for every image.
              </p>
            )}
          </div>

          {/* 3. What to call the result + how to trigger it. */}
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

          {/* 4. Hyperparameters, collapsed: the defaults suit a first run, and hiding
              them keeps the primary flow at three decisions. */}
          <button
            type="button"
            className="w-fit text-xs text-muted-foreground underline-offset-2 hover:underline"
            onClick={() => setShowAdvanced((s) => !s)}
            aria-expanded={showAdvanced}
          >
            {showAdvanced ? "Hide training settings" : "Training settings (defaults suit a first run)"}
          </button>
          {showAdvanced && (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="grid gap-1.5">
                  <Label className="text-xs">Steps</Label>
                  <Input
                    type="number"
                    min={1}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value) || 1)}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-xs">LoRA rank</Label>
                  <Input
                    type="number"
                    min={1}
                    value={rank}
                    onChange={(e) => setRank(Number(e.target.value) || 1)}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-xs">Resolution</Label>
                  <Input
                    type="number"
                    min={64}
                    step={64}
                    value={resolution}
                    onChange={(e) => setResolution(Number(e.target.value) || 1024)}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-xs">Batch</Label>
                  <Input
                    type="number"
                    min={1}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value) || 1)}
                    className="h-8 text-xs"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label className="text-xs">Learning rate</Label>
                  <Input
                    type="number"
                    step={0.00001}
                    min={0}
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value) || 0.0001)}
                    className="h-8 text-xs"
                  />
                </div>
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

          {status && status.status !== "idle" && (
            <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-medium capitalize">{status.status}</span>
                <span className="text-muted-foreground">
                  {status.total_steps > 0 ? `${status.step}/${status.total_steps}` : ""}
                </span>
              </div>
              <div className="mb-2 h-1.5 w-full overflow-hidden rounded-full bg-border">
                <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
              </div>
              <div className="text-muted-foreground">
                {completed
                  ? "Adapter ready - find it in the LoRA picker on this page."
                  : status.message}
                {status.loss != null && !completed && <> · loss {status.loss.toFixed(4)}</>}
                {status.lora_path && (
                  <div className="mt-1 break-all text-[11px]">Saved: {status.lora_path}</div>
                )}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          {active ? (
            <Button type="button" variant="destructive" onClick={onStop}>
              Stop
            </Button>
          ) : completed ? (
            <>
              <Button type="button" variant="secondary" onClick={() => onOpenChange(false)}>
                Done - open the LoRA picker
              </Button>
              <Button type="button" onClick={onStart} disabled={starting || uploading}>
                {starting ? "Starting..." : "Train another"}
              </Button>
            </>
          ) : (
            <Button type="button" onClick={onStart} disabled={starting || uploading}>
              {starting ? "Starting..." : "Start training"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
