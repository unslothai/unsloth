// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

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
  type DiffusionTrainingStatus,
  getDiffusionTrainingStatus,
  startDiffusionTraining,
  stopDiffusionTraining,
} from "./api";

const DEFAULT_SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0";

// A self-contained "Train a LoRA" dialog for the diffusion (SDXL) trainer. It posts to
// /api/train/diffusion/start and polls /status while open, so it never blocks the page and
// works whether or not a model is loaded for generation. Only SDXL is trainable today.
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
  const [baseModel, setBaseModel] = useState(defaultBaseModel || DEFAULT_SDXL_BASE);
  const [dataDir, setDataDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [instancePrompt, setInstancePrompt] = useState("");
  const [steps, setSteps] = useState(500);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [rank, setRank] = useState(16);
  const [resolution, setResolution] = useState(1024);
  const [batchSize, setBatchSize] = useState(1);
  const [precision, setPrecision] = useState<"bf16" | "fp16" | "no">("bf16");
  const [starting, setStarting] = useState(false);
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);

  // The dialog stays mounted (ImagesPage is keep-alive), so the initial state seed does not
  // reflect a base model loaded AFTER mount. Re-seed the base-model field from the current
  // default each time the dialog opens, so "Train LoRA" after loading an SDXL checkpoint
  // starts from that checkpoint's diffusers repo, not the hard-coded default.
  useEffect(() => {
    if (open) setBaseModel(defaultBaseModel || DEFAULT_SDXL_BASE);
  }, [open, defaultBaseModel]);

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

  const onStart = useCallback(async () => {
    if (!baseModel.trim() || !dataDir.trim() || !outputDir.trim()) {
      toast.error("Base model, dataset folder, and output folder are required.");
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
        base_model: baseModel.trim(),
        data_dir: dataDir.trim(),
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
    baseModel,
    dataDir,
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
          <DialogTitle>Train a LoRA (SDXL)</DialogTitle>
          <DialogDescription>
            Fine-tune an SDXL LoRA on a folder of images. Captions come from a metadata.jsonl,
            per-image .txt sidecars, or the instance prompt below. The adapter is saved to the
            output folder shown after the run.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 overflow-y-auto py-2 pr-1">
          <div className="grid gap-1.5">
            <Label className="text-xs">Base model (SDXL repo or local path)</Label>
            <Input
              value={baseModel}
              spellCheck={false}
              onChange={(e) => setBaseModel(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Dataset folder</Label>
            <Input
              value={dataDir}
              placeholder="/path/to/images"
              spellCheck={false}
              onChange={(e) => setDataDir(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Output folder (LoRA .safetensors)</Label>
            <Input
              value={outputDir}
              placeholder="/path/to/output-lora"
              spellCheck={false}
              onChange={(e) => setOutputDir(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Instance prompt (optional; used for uncaptioned images)</Label>
            <Input
              value={instancePrompt}
              placeholder="a photo of sks style"
              onChange={(e) => setInstancePrompt(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
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
                className="h-8 rounded-md border border-input bg-background px-2 text-xs"
              >
                <option value="bf16">bf16 (default)</option>
                <option value="fp16">fp16 (older GPUs)</option>
                <option value="no">fp32 (no mixed)</option>
              </select>
            </div>
          </div>

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
                {status.message}
                {status.loss != null && <> · loss {status.loss.toFixed(4)}</>}
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
          ) : (
            <Button type="button" onClick={onStart} disabled={starting}>
              {starting ? "Starting..." : "Start training"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
