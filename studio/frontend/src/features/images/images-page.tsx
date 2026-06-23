// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Download01Icon, ImageAdd02Icon, PaintBrush02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { SectionCard } from "@/components/section-card";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import { formatBytes } from "@/features/hub/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  type DiffusionLoadProgress,
  type DiffusionStatus,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";

interface CuratedModel {
  label: string;
  repo_id: string;
  gguf_filename: string;
  base_repo: string;
  family: string;
  // The base repo (VAE/text-encoders) is gated on HF, so loading needs a token —
  // even though the GGUF repo itself is an open redistribution.
  gated: boolean;
}

// Repo ids and GGUF filenames are the Hub-canonical (lowercase) names; verified
// to exist on HF. The base_repo provides the VAE/text-encoders the GGUF omits.
const CURATED_MODELS: CuratedModel[] = [
  {
    label: "FLUX.2 klein base 4B",
    repo_id: "unsloth/FLUX.2-klein-base-4B-GGUF",
    gguf_filename: "flux-2-klein-base-4b-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.2-klein-base-4B",
    family: "flux.2-klein",
    gated: false,
  },
  {
    label: "FLUX.2 dev",
    repo_id: "unsloth/FLUX.2-dev-GGUF",
    gguf_filename: "flux2-dev-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.2-dev",
    family: "flux.2",
    gated: true,
  },
  {
    label: "FLUX.1 dev",
    repo_id: "city96/FLUX.1-dev-gguf",
    gguf_filename: "flux1-dev-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.1-dev",
    family: "flux.1",
    gated: true,
  },
];

const RESOLUTIONS: Array<{ label: string; w: number; h: number }> = [
  { label: "Square 1024", w: 1024, h: 1024 },
  { label: "Square 768", w: 768, h: 768 },
  { label: "Portrait 832×1216", w: 832, h: 1216 },
  { label: "Landscape 1216×832", w: 1216, h: 832 },
];

interface ResultItem {
  id: number;
  src: string;
  prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
}

function downloadImage(src: string, seed: number) {
  const link = document.createElement("a");
  link.href = src;
  link.download = `unsloth-${seed}.png`;
  link.click();
}

// Title + percent/label for the chat progress component, from a progress poll.
function progressView(p: DiffusionLoadProgress): {
  title: string;
  progressPercent: number | null;
  progressLabel: string | null;
} {
  const title = p.phase === "finalizing" ? "Loading to GPU…" : "Downloading model…";
  if (p.bytes_total > 0) {
    return {
      title,
      progressPercent: p.fraction * 100,
      progressLabel: `${formatBytes(p.bytes_downloaded)} of ${formatBytes(p.bytes_total)}`,
    };
  }
  // Unknown total: show bytes counting up, no bar.
  return {
    title,
    progressPercent: null,
    progressLabel: p.bytes_downloaded > 0 ? `${formatBytes(p.bytes_downloaded)} downloaded` : null,
  };
}

// Matches the field-label style used across Studio (export/chat settings).
function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

type Busy = "loading" | "unloading" | "generating" | null;

export function ImagesPage() {
  const [modelIdx, setModelIdx] = useState(0);
  const [hfToken, setHfToken] = useState("");
  const [prompt, setPrompt] = useState(
    "a tiny ginger sloth coding in a sunlit treehouse, photorealistic",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolutionIdx, setResolutionIdx] = useState(0);
  const [steps, setSteps] = useState(24);
  const [guidance, setGuidance] = useState(3.5);
  const [seed, setSeed] = useState("");

  const [busy, setBusy] = useState<Busy>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  const [loadProgress, setLoadProgress] = useState<DiffusionLoadProgress | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  // Stable, ever-increasing ids: results are prepended, so an array index would
  // re-key every existing image on each new generation.
  const nextResultId = useRef(0);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const model = CURATED_MODELS[modelIdx];
  const resolution = RESOLUTIONS[resolutionIdx];
  const selected = useMemo(
    () => results.find((r) => r.id === selectedId) ?? results[0] ?? null,
    [results, selectedId],
  );

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getDiffusionStatus());
    } catch {
      // Status is best-effort; a failed poll shouldn't surface an error toast.
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    // Stop polling if the page unmounts mid-load.
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [refreshStatus]);

  // Poll load-progress until the background load reaches "ready" or "error".
  const pollLoadProgress = useCallback(async () => {
    try {
      const p = await getDiffusionLoadProgress();
      setLoadProgress(p);
      if (p.phase === "ready") {
        setStatus(await getDiffusionStatus());
        toast.success("Model loaded");
        setBusy(null);
        setLoadProgress(null);
        return;
      }
      if (p.phase === "error") {
        toast.error(p.error || "Failed to load model");
        setBusy(null);
        setLoadProgress(null);
        return;
      }
    } catch {
      // Transient poll failure: keep trying.
    }
    pollTimer.current = setTimeout(() => void pollLoadProgress(), 1000);
  }, []);

  const handleLoad = useCallback(async () => {
    // Cancel any prior poll loop so two can't run at once.
    if (pollTimer.current) clearTimeout(pollTimer.current);
    setBusy("loading");
    setLoadProgress(null);
    try {
      // Returns immediately — the load runs in the background; we poll for it.
      await loadDiffusionModel({
        model_path: model.repo_id,
        gguf_filename: model.gguf_filename,
        base_repo: model.base_repo,
        family_override: model.family,
        hf_token: hfToken.trim() || undefined,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to start load");
      setBusy(null);
      void refreshStatus();
      return;
    }
    void pollLoadProgress();
  }, [model, hfToken, refreshStatus, pollLoadProgress]);

  const handleUnload = useCallback(async () => {
    setBusy("unloading");
    try {
      setStatus(await unloadDiffusionModel());
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
    } finally {
      setBusy(null);
    }
  }, [refreshStatus]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    let parsedSeed: number | undefined;
    if (seed.trim()) {
      const n = Number(seed);
      if (!Number.isInteger(n) || n < 0) {
        toast.error("Seed must be a non-negative integer");
        return;
      }
      parsedSeed = n;
    }

    setBusy("generating");
    try {
      const res = await generateDiffusionImage({
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim() || undefined,
        width: resolution.w,
        height: resolution.h,
        steps,
        guidance,
        seed: parsedSeed,
      });
      const id = nextResultId.current++;
      setResults((prev) => [
        {
          id,
          src: `data:${res.mime};base64,${res.image_b64}`,
          prompt: prompt.trim(),
          width: resolution.w,
          height: resolution.h,
          steps,
          guidance,
          seed: res.seed,
        },
        ...prev,
      ]);
      setSelectedId(id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Image generation failed");
    } finally {
      setBusy(null);
    }
  }, [prompt, negativePrompt, resolution, steps, guidance, seed]);

  const statusLabel = useMemo(() => {
    if (busy === "loading") return "Loading…";
    if (status?.loaded) return `Loaded · ${status.repo_id}`;
    return "No model loaded";
  }, [busy, status]);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden p-4 sm:p-6">
      {/* ── Left control rail ─────────────────────────────── */}
      <SectionCard
        icon={<HugeiconsIcon icon={PaintBrush02Icon} className="size-5" strokeWidth={1.5} />}
        title="Image generation"
        description="Generate from local diffusion GGUFs"
        accent="indigo"
        className="w-[360px] shrink-0 gap-4 overflow-y-auto"
      >
        <Field label="Model">
          <Select value={String(modelIdx)} onValueChange={(v) => setModelIdx(Number(v))}>
            <SelectTrigger>
              <SelectValue placeholder="Pick a model" />
            </SelectTrigger>
            <SelectContent>
              {CURATED_MODELS.map((m, idx) => (
                <SelectItem key={m.repo_id} value={String(idx)}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        {model.gated && (
          <Field label="HuggingFace token">
            <Input
              type="password"
              placeholder="hf_… (required for gated repo)"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
            />
          </Field>
        )}

        <div className="flex items-center gap-2">
          <Button className="flex-1" onClick={handleLoad} disabled={busy !== null}>
            {busy === "loading" ? <Spinner className="mr-2 size-4" /> : null}
            Load
          </Button>
          <Button
            variant="outline"
            onClick={handleUnload}
            disabled={busy !== null || !status?.loaded}
          >
            {busy === "unloading" ? <Spinner className="mr-2 size-4" /> : null}
            Unload
          </Button>
        </div>
        {busy === "loading" && loadProgress ? (
          <ModelLoadDescription {...progressView(loadProgress)} />
        ) : (
          <p className="-mt-1 truncate text-xs text-muted-foreground" title={statusLabel}>
            {statusLabel}
          </p>
        )}

        <Separator />

        <Field label="Prompt">
          <Textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </Field>
        <Field label="Negative prompt">
          <Textarea
            rows={2}
            placeholder="What to avoid (optional)"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
          />
        </Field>

        <Field label="Resolution">
          <Select value={String(resolutionIdx)} onValueChange={(v) => setResolutionIdx(Number(v))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RESOLUTIONS.map((r, idx) => (
                <SelectItem key={r.label} value={String(idx)}>
                  {r.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        <Field label={`Steps · ${steps}`}>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={1} max={50} step={1} />
        </Field>
        <Field label={`Guidance · ${guidance.toFixed(1)}`}>
          <Slider
            value={[guidance]}
            onValueChange={([v]) => setGuidance(v)}
            min={0}
            max={15}
            step={0.5}
          />
        </Field>
        <Field label="Seed">
          <Input
            placeholder="Random if empty"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </Field>

        <Button onClick={handleGenerate} disabled={busy !== null || !status?.loaded}>
          {busy === "generating" ? <Spinner className="mr-2 size-4" /> : null}
          Generate
        </Button>
      </SectionCard>

      {/* ── Right: preview canvas + history strip ─────────── */}
      <div className="bg-card corner-squircle relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-3xl ring-1 ring-foreground/10">
        <div className="relative flex flex-1 items-center justify-center overflow-auto p-6">
          {selected ? (
            <>
              <img
                src={selected.src}
                alt={selected.prompt}
                className="max-h-full max-w-full rounded-xl object-contain shadow-sm"
              />
              <div className="absolute bottom-4 right-4 flex items-center gap-2">
                <span className="rounded-md bg-background/80 px-2 py-1 text-xs text-muted-foreground backdrop-blur">
                  {selected.width}×{selected.height} · seed {selected.seed}
                </span>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => downloadImage(selected.src, selected.seed)}
                >
                  <HugeiconsIcon icon={Download01Icon} className="mr-1.5 size-4" />
                  Download
                </Button>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <HugeiconsIcon icon={ImageAdd02Icon} className="size-12" strokeWidth={1.5} />
              <p className="text-sm">
                {status?.loaded
                  ? "Enter a prompt and hit Generate."
                  : "Load a model, then generate to see results here."}
              </p>
            </div>
          )}
          {busy === "generating" && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/40 backdrop-blur-sm">
              <Spinner className="size-8" />
            </div>
          )}
        </div>

        {results.length > 0 && (
          <div className="flex shrink-0 gap-2 overflow-x-auto border-t border-foreground/10 p-3">
            {results.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => setSelectedId(r.id)}
                title={`seed ${r.seed}`}
                className={cn(
                  "size-16 shrink-0 overflow-hidden rounded-lg ring-2 transition-colors",
                  r.id === selected?.id ? "ring-primary" : "ring-transparent hover:ring-border",
                )}
              >
                <img src={r.src} alt={r.prompt} className="size-full object-cover" />
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
