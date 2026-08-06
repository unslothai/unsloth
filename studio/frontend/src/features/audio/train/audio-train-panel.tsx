// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio Train workspace: a LoRA run on an audio base through the generic
// training API (/api/train/*), which already carries the snac/csm/bicodec/dac/
// whisper branches. Kept to the essentials the TTS notebooks tune; everything
// else rides the trainer defaults.

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { ParamSlider } from "@/features/chat";
import type { TrainingSeriesPoint } from "@/features/training";
// The generic train API is not exported from the training feature index; the audio panel drives the same endpoints the Train page owns.
// eslint-disable-next-line no-restricted-imports
import {
  getTrainingStatus,
  startTraining,
  stopTraining,
} from "@/features/training/api/train-api";
// eslint-disable-next-line no-restricted-imports
import type { TrainingStatusResponse } from "@/features/training/types/runtime";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

import { AudioCharts } from "./audio-charts";
import { AudioBaseCombobox, AudioDatasetCombobox } from "./hub-comboboxes";

/** Curated audio-trainable bases: complete trainer branches plus model_defaults
 *  yamls. Pinned above the live Hub rows rather than being the whole list. */
const TRAIN_BASES = [
  { repoId: "unsloth/orpheus-3b-0.1-ft", label: "Orpheus TTS 3B" },
  { repoId: "unsloth/csm-1b", label: "Sesame CSM 1B" },
  { repoId: "unsloth/Spark-TTS-0.5B", label: "Spark TTS 0.5B" },
  { repoId: "unsloth/Llama-OuteTTS-1.0-1B", label: "Oute TTS 1B" },
  { repoId: "unsloth/Llasa-1B", label: "Llasa 1B" },
  { repoId: "unsloth/Llasa-3B", label: "Llasa 3B" },
  { repoId: "unsloth/whisper-large-v3", label: "Whisper Large v3 (STT)" },
] as const;

const TRAIN_BASE_IDS = TRAIN_BASES.map((b) => b.repoId);
const TRAIN_BASE_LABELS = new Map<string, string>(
  TRAIN_BASES.map((b) => [b.repoId, b.label]),
);

// audio + text columns, so it maps with no overrides.
const DEFAULT_DATASET = "Etherll/kaira";
// Pinned under it: the dataset the unsloth TTS notebooks use.
const CURATED_DATASETS = [DEFAULT_DATASET, "MrDragonFox/Elise"] as const;

// Page inset plus the trigger's own pl-4, so labels start on the same line as
// "Select audio model" in the header.
const HEADER_ALIGNED_PL =
  "pl-[calc(var(--studio-media-header-left-inset,1.5rem)+1rem)]";

/** Section header, so the run reads as model -> data -> parameters. */
function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="grid gap-3">
      <div className="grid gap-0.5">
        <h3 className="text-ui-11p5 font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </h3>
        {hint ? (
          <p className="text-ui-11p5 leading-snug text-muted-foreground/80">
            {hint}
          </p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

const POLL_MS = 2000;

function seriesFrom(
  steps: number[] | undefined,
  values: number[] | undefined,
): TrainingSeriesPoint[] {
  if (!steps || !values) return [];
  const count = Math.min(steps.length, values.length);
  const out: TrainingSeriesPoint[] = [];
  for (let i = 0; i < count; i += 1) out.push({ step: steps[i], value: values[i] });
  return out;
}

export function AudioTrainPanel({
  active,
  onDeploy,
}: {
  active: boolean;
  /** Load a finished checkpoint into Create; absent = no deploy affordance. */
  onDeploy?: (outputDir: string) => void;
}) {
  const [base, setBase] = useState<string>(TRAIN_BASES[0].repoId);
  const [dataset, setDataset] = useState<string>(DEFAULT_DATASET);
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState("2e-4");
  const [loraRank, setLoraRank] = useState(16);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(4);
  const [maxSteps, setMaxSteps] = useState(0);
  const [audioColumn, setAudioColumn] = useState("");
  const [textColumn, setTextColumn] = useState("");
  const [speakerColumn, setSpeakerColumn] = useState("");
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_audio_train_advanced_open",
  );

  const [status, setStatus] = useState<TrainingStatusResponse | null>(null);
  const [starting, setStarting] = useState(false);
  const pollTimer = useRef<number | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getTrainingStatus());
    } catch {
      // A backend restart mid-poll resyncs on the next tick.
    }
  }, []);

  // Poll while the panel is visible; the run itself survives navigation.
  useEffect(() => {
    if (!active) return;
    void refreshStatus();
    const id = window.setInterval(() => void refreshStatus(), POLL_MS);
    pollTimer.current = id;
    return () => {
      window.clearInterval(id);
      pollTimer.current = null;
    };
  }, [active, refreshStatus]);

  const running = Boolean(status?.is_training_running);
  const outputDir = status?.details?.output_dir ?? null;
  const completed = status?.phase === "completed" && Boolean(outputDir);

  const handleStart = useCallback(async () => {
    const datasetId = dataset.trim();
    if (!datasetId) {
      toast.error("Name a Hugging Face dataset to train on.");
      return;
    }
    setStarting(true);
    try {
      const mapping: Record<string, string> = {};
      if (audioColumn.trim()) mapping[audioColumn.trim()] = "audio";
      if (textColumn.trim()) mapping[textColumn.trim()] = "text";
      if (speakerColumn.trim()) mapping[speakerColumn.trim()] = "speaker_id";
      const res = await startTraining({
        model_name: base,
        project_name: null,
        training_type: "LoRA/QLoRA",
        hf_token: hfApiToken(getHfToken()) ?? null,
        load_in_4bit: true,
        max_seq_length: 2048,
        hf_dataset: datasetId,
        subset: null,
        train_split: "train",
        eval_split: null,
        // The trainer rejects streaming audio datasets, so this stays off.
        dataset_streaming: false,
        dataset_slice_start: null,
        dataset_slice_end: null,
        local_datasets: [],
        local_eval_datasets: [],
        format_type: "auto",
        custom_format_mapping: Object.keys(mapping).length > 0 ? mapping : null,
        num_epochs: maxSteps > 0 ? 0 : epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        gradient_accumulation_steps: gradAccum,
        warmup_steps: 5,
        warmup_ratio: null,
        max_steps: maxSteps > 0 ? maxSteps : null,
        save_steps: 0,
        eval_steps: 0,
        weight_decay: 0.001,
        max_grad_norm: 1.0,
        random_seed: 3407,
        packing: false,
        optim: "adamw_8bit",
        lr_scheduler_type: "linear",
        use_lora: true,
        lora_r: loraRank,
        lora_alpha: loraRank * 2,
        lora_dropout: 0,
        target_modules: [
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
        ],
        gradient_checkpointing: "unsloth",
        use_rslora: false,
        use_loftq: false,
        use_dora: false,
        train_on_completions: false,
        finetune_vision_layers: true,
        finetune_language_layers: true,
        finetune_attention_modules: true,
        finetune_mlp_modules: true,
        is_dataset_image: false,
        is_dataset_audio: true,
        is_embedding: false,
        enable_wandb: false,
        wandb_token: null,
        wandb_project: null,
        enable_tensorboard: false,
        tensorboard_dir: null,
      });
      if (res.status === "error") {
        toast.error(res.error ?? res.message);
      } else {
        toast.success("Training started");
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Training could not start.",
      );
    } finally {
      setStarting(false);
      void refreshStatus();
    }
  }, [
    base,
    dataset,
    epochs,
    learningRate,
    loraRank,
    batchSize,
    gradAccum,
    maxSteps,
    audioColumn,
    textColumn,
    speakerColumn,
    refreshStatus,
  ]);

  const handleStop = useCallback(async () => {
    try {
      await stopTraining(true);
      toast.success("Stopping the run; the adapter saved so far is kept.");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not stop the run.",
      );
    } finally {
      void refreshStatus();
    }
  }, [refreshStatus]);

  const lossHistory = seriesFrom(
    status?.metric_history?.steps,
    status?.metric_history?.loss,
  );
  const gradNormHistory = seriesFrom(
    status?.metric_history?.grad_norm_steps ?? status?.metric_history?.steps,
    status?.metric_history?.grad_norm,
  );

  return (
    <div
      className={cn(
        "hover-scrollbar flex min-h-0 flex-1 flex-col gap-8 overflow-y-auto pb-10 pr-8 pt-9",
        HEADER_ALIGNED_PL,
      )}
    >
      <div className="grid max-w-2xl gap-8">
        <Section
          title="Model"
          hint="TTS bases fine-tune a voice from audio + transcript pairs; a Whisper base fine-tunes transcription instead."
        >
          <div className="grid gap-1.5">
            <span className="text-ui-13 font-medium text-foreground">
              Base model
            </span>
            <AudioBaseCombobox
              value={base}
              onValueChange={setBase}
              curated={TRAIN_BASE_IDS}
              labelFor={(id) => TRAIN_BASE_LABELS.get(id)}
              disabled={running}
              accessToken={hfApiToken(getHfToken()) ?? undefined}
            />
          </div>
        </Section>

        <Section
          title="Data"
          hint={`Needs an audio column plus a transcript column (audio/text by default; override in Advanced). ${DEFAULT_DATASET} maps with no overrides.`}
        >
          <div className="grid gap-1.5">
            <span className="text-ui-13 font-medium text-foreground">
              Hugging Face dataset
            </span>
            <AudioDatasetCombobox
              value={dataset}
              onValueChange={setDataset}
              curated={CURATED_DATASETS}
              disabled={running}
              accessToken={hfApiToken(getHfToken()) ?? undefined}
            />
          </div>
        </Section>

        <Section title="Parameters">
        <ParamSlider
          label="Epochs"
          value={epochs}
          min={1}
          max={10}
          step={1}
          onChange={setEpochs}
          disabled={running || maxSteps > 0}
        />
        <ParamSlider
          label="LoRA rank"
          value={loraRank}
          min={4}
          max={128}
          step={4}
          onChange={setLoraRank}
          disabled={running}
        />
        <div className="grid gap-1.5">
          <span className="text-ui-13 font-medium text-foreground">
            Learning rate
          </span>
          <Input
            value={learningRate}
            onChange={(event) => setLearningRate(event.target.value)}
            disabled={running}
            className="w-32"
          />
        </div>
        </Section>

        <AdvancedDisclosure open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <ParamSlider
            label="Batch size"
            value={batchSize}
            min={1}
            max={16}
            step={1}
            onChange={setBatchSize}
            disabled={running}
          />
          <ParamSlider
            label="Gradient accumulation"
            value={gradAccum}
            min={1}
            max={32}
            step={1}
            onChange={setGradAccum}
            disabled={running}
          />
          <ParamSlider
            label="Max steps (0 = train by epochs)"
            value={maxSteps}
            min={0}
            max={2000}
            step={10}
            onChange={setMaxSteps}
            disabled={running}
          />
          <div className="grid gap-1.5">
            <span className="text-ui-13 font-medium text-foreground">
              Column mapping
            </span>
            <div className="flex flex-wrap gap-2">
              <Input
                value={audioColumn}
                onChange={(event) => setAudioColumn(event.target.value)}
                placeholder="audio column"
                disabled={running}
                className="w-40"
              />
              <Input
                value={textColumn}
                onChange={(event) => setTextColumn(event.target.value)}
                placeholder="text column"
                disabled={running}
                className="w-40"
              />
              <Input
                value={speakerColumn}
                onChange={(event) => setSpeakerColumn(event.target.value)}
                placeholder="speaker column"
                disabled={running}
                className="w-40"
              />
            </div>
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              Leave blank to auto-detect (audio/speech, text/transcript,
              source/speaker_id).
            </p>
          </div>
        </AdvancedDisclosure>

        <div className="grid gap-3 border-t border-border/60 pt-5">
          {/* The whole run in one line, so Start needs no scroll back up. */}
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            LoRA r{loraRank} on{" "}
            <span className="text-foreground">
              {TRAIN_BASE_LABELS.get(base) ?? (base || "no base")}
            </span>{" "}
            over{" "}
            <span className="text-foreground">{dataset || "no dataset"}</span> —{" "}
            {maxSteps > 0
              ? `${maxSteps} steps`
              : `${epochs} epoch${epochs === 1 ? "" : "s"}`}
            , lr {learningRate}, batch {batchSize}×{gradAccum}
          </p>
          <div className="flex items-center gap-2">
            {running ? (
              <Button variant="destructive" onClick={() => void handleStop()}>
                Stop and save
              </Button>
            ) : (
              <Button
                onClick={() => void handleStart()}
                disabled={starting || !dataset.trim() || !base.trim()}
              >
                {starting ? <Spinner className="mr-2 size-4" /> : null}
                Start training
              </Button>
            )}
            {completed && outputDir && onDeploy ? (
              <Button variant="secondary" onClick={() => onDeploy(outputDir)}>
                Use in Create
              </Button>
            ) : null}
          </div>
        </div>

        {status?.message ? (
          <p className="text-ui-13 text-muted-foreground">
            {running ? <Spinner className="mr-2 inline-block size-3.5" /> : null}
            {status.message}
            {status.details?.step != null && status.details?.total_steps != null
              ? ` (step ${status.details.step}/${status.details.total_steps})`
              : null}
          </p>
        ) : null}
        {status?.error ? (
          <p className="text-ui-13 text-destructive">{status.error}</p>
        ) : null}
      </div>

      {lossHistory.length > 0 ? (
        <div className="max-w-5xl">
          <AudioCharts
            lossHistory={lossHistory}
            gradNormHistory={gradNormHistory}
          />
        </div>
      ) : null}
    </div>
  );
}
