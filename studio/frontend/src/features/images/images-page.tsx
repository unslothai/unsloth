// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SectionCard } from "@/components/section-card";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { PaintBrush02Icon, SparklesIcon, GpuIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  fetchDiffusionStatus,
  generateDiffusionImage,
  loadDiffusionModel,
  unloadDiffusionModel,
  type DiffusionGenerateResponse,
  type DiffusionStatus,
} from "./api";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Curated short list of working diffusion GGUFs. Picked to span
// size + license so any GPU class has at least one viable option:
//   FLUX.2 klein 4B  -> ~13 GB VRAM with Q4_K_S, Apache 2.0
//   FLUX.2 klein 9B  -> ~17 GB VRAM, FLUX [klein] non-commercial (gated)
//   FLUX.2 dev       -> ~24+ GB VRAM, FLUX [dev] non-commercial (gated)
//   FLUX.1 dev       -> ~12 GB VRAM, older but widely tested (gated)
//
// Filenames mirror the Hub canonical case (lowercase 'flux-2-klein-4b')
// and base_repo is set explicitly so the backend never falls back to the
// family default. The CLI on the backend can load anything supported by
// detect_family(); this list just keeps the picker compact for the v1 UI.
const CURATED_MODELS: Array<{
  label: string;
  repo_id: string;
  default_gguf: string;
  base_repo: string;
  family: string;
  notes: string;
}> = [
  {
    label: "FLUX.2 klein base 4B (Q4_K_S, Apache 2.0)",
    repo_id: "unsloth/FLUX.2-klein-base-4B-GGUF",
    default_gguf: "flux-2-klein-base-4b-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.2-klein-base-4B",
    family: "flux.2-klein",
    notes: "13 GB VRAM, fastest. Apache 2.0, ungated.",
  },
  {
    label: "FLUX.2 klein 4B (Q4_K_S, distilled)",
    repo_id: "unsloth/FLUX.2-klein-4B-GGUF",
    default_gguf: "flux-2-klein-4b-Q4_K_S.gguf",
    // Distilled GGUF must pair with the distilled base, not the Base
    // checkpoint. The Hub model card for the GGUF lists
    // base_model: black-forest-labs/FLUX.2-klein-4B.
    base_repo: "black-forest-labs/FLUX.2-klein-4B",
    family: "flux.2-klein",
    notes: "13 GB VRAM. Distilled klein 4B. Requires HF access to FLUX.2 klein 4B.",
  },
  {
    label: "FLUX.2 klein 9B (Q4_K_S, gated)",
    repo_id: "unsloth/FLUX.2-klein-9B-GGUF",
    default_gguf: "flux-2-klein-9b-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.2-klein-9B",
    family: "flux.2-klein",
    notes: "17 GB VRAM. Higher quality distilled. Requires HF access to FLUX.2 klein 9B.",
  },
  {
    label: "FLUX.2 dev (Q4_K_S, gated)",
    repo_id: "unsloth/FLUX.2-dev-GGUF",
    default_gguf: "flux2-dev-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.2-dev",
    family: "flux.2",
    notes: "24+ GB VRAM. Requires HF access to FLUX.2 dev.",
  },
  {
    label: "FLUX.1 dev (Q4_K_S, city96, gated)",
    repo_id: "city96/FLUX.1-dev-gguf",
    default_gguf: "flux1-dev-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.1-dev",
    family: "flux.1",
    notes: "12 GB VRAM. Older but widely tested. Requires HF access to FLUX.1 dev.",
  },
];

const DEFAULT_PRESET = CURATED_MODELS[0];

const RESOLUTION_PRESETS: Array<{ label: string; w: number; h: number }> = [
  { label: "Square 1024", w: 1024, h: 1024 },
  { label: "Square 768", w: 768, h: 768 },
  { label: "Square 512", w: 512, h: 512 },
  { label: "Portrait 832x1216", w: 832, h: 1216 },
  { label: "Landscape 1216x832", w: 1216, h: 832 },
];

export function ImagesPage() {
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  const [refreshingStatus, setRefreshingStatus] = useState(false);
  const [busy, setBusy] = useState<"idle" | "loading" | "unloading" | "generating">("idle");

  const [presetIndex, setPresetIndex] = useState(0);
  const [customRepoId, setCustomRepoId] = useState("");
  const [customGguf, setCustomGguf] = useState("");
  const [customFamily, setCustomFamily] = useState<string>("auto");
  const [useCustom, setUseCustom] = useState(false);
  const [hfToken, setHfToken] = useState("");

  const [prompt, setPrompt] = useState("a tiny ginger sloth coding in a sunlit treehouse, photorealistic");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [steps, setSteps] = useState(24);
  const [guidance, setGuidance] = useState(3.5);
  const [resolutionIdx, setResolutionIdx] = useState(0);
  const [seed, setSeed] = useState<string>("");

  const [results, setResults] = useState<DiffusionGenerateResponse[]>([]);
  const lastErrorRef = useRef<string | null>(null);

  const preset = CURATED_MODELS[presetIndex] ?? DEFAULT_PRESET;
  const resolution = RESOLUTION_PRESETS[resolutionIdx];

  const refreshStatus = useCallback(async () => {
    setRefreshingStatus(true);
    try {
      const next = await fetchDiffusionStatus();
      setStatus(next);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (lastErrorRef.current !== msg) {
        lastErrorRef.current = msg;
        toast.error("Could not fetch image-model status", { description: msg });
      }
    } finally {
      setRefreshingStatus(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  const handleLoad = useCallback(async () => {
    setBusy("loading");
    try {
      const repo = useCustom ? customRepoId.trim() : preset.repo_id;
      const gguf = useCustom ? customGguf.trim() || undefined : preset.default_gguf;
      // Custom mode lets the user pin a family explicitly because
      // detect_family is substring-based and exotic repo names (custom
      // fine-tunes, third-party mirrors) frequently fail to match.
      // "auto" leaves the override blank and lets the backend infer.
      const family = useCustom
        ? customFamily === "auto"
          ? undefined
          : customFamily
        : preset.family;
      // Always pass base_repo for curated entries; custom-repo mode
      // lets the backend either infer it from the family default or
      // (when no GGUF is given) treat the repo as a full diffusers
      // checkpoint and call from_pretrained on it directly.
      const baseRepo = useCustom ? undefined : preset.base_repo;
      if (!repo) {
        toast.error("Pick a model first");
        return;
      }
      const next = await loadDiffusionModel({
        repo_id: repo,
        gguf_filename: gguf,
        base_repo: baseRepo,
        family,
        hf_token: hfToken.trim() || undefined,
      });
      setStatus(next);
      toast.success("Loaded image model", { description: next.repo_id ?? undefined });
    } catch (err) {
      toast.error("Failed to load image model", {
        description: err instanceof Error ? err.message : String(err),
      });
      // Backend clears its old pipeline before allocating the new one;
      // a failed swap leaves status.is_loaded=false while our local
      // copy still says loaded. Re-fetch so Generate disables and the
      // user does not see a stale "Loaded:" label.
      await refreshStatus();
    } finally {
      setBusy("idle");
    }
  }, [useCustom, customRepoId, customGguf, customFamily, preset, hfToken, refreshStatus]);

  const handleUnload = useCallback(async () => {
    setBusy("unloading");
    try {
      await unloadDiffusionModel();
      await refreshStatus();
    } catch (err) {
      toast.error("Failed to unload image model", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setBusy("idle");
    }
  }, [refreshStatus]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    setBusy("generating");
    try {
      // Reject non-integer or out-of-safe-integer-range seeds rather
      // than silently rounding via Number(). The backend takes an int
      // and a precision loss here would yield a different image than
      // the seed the user typed.
      const seedStr = seed.trim();
      let parsedSeed: number | undefined;
      if (seedStr) {
        if (!/^-?\d+$/.test(seedStr)) {
          toast.error("Seed must be an integer");
          return;
        }
        const candidate = Number(seedStr);
        if (
          !Number.isFinite(candidate) ||
          !Number.isSafeInteger(candidate)
        ) {
          toast.error(
            "Seed must fit in a JavaScript safe integer (<= 2^53 - 1)",
          );
          return;
        }
        parsedSeed = candidate;
      }
      const out = await generateDiffusionImage({
        prompt,
        negative_prompt: negativePrompt.trim() || undefined,
        num_inference_steps: steps,
        guidance_scale: guidance,
        width: resolution.w,
        height: resolution.h,
        seed: parsedSeed,
      });
      setResults((prev) => [out, ...prev].slice(0, 12));
    } catch (err) {
      toast.error("Image generation failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setBusy("idle");
    }
  }, [prompt, negativePrompt, steps, guidance, resolution, seed]);

  const statusLabel = useMemo(() => {
    if (!status) return refreshingStatus ? "Checking..." : "Not loaded";
    if (status.is_loading) return "Loading...";
    if (status.is_loaded) {
      const dev = status.device ? ` on ${status.device}` : "";
      return `Loaded: ${status.repo_id ?? "(unknown)"} (${status.family ?? "unknown"})${dev}`;
    }
    return "Not loaded";
  }, [status, refreshingStatus]);

  // FLUX.2 / FLUX.2 klein pipelines do NOT accept negative_prompt and
  // would 500 if we sent one through. The backend strips the field
  // defensively but hiding it client-side keeps the UI honest.
  const supportsNegativePrompt = useMemo(() => {
    const family = status?.family;
    if (!family) {
      const candidate = useCustom ? undefined : preset.family;
      if (!candidate) return true;
      return !candidate.startsWith("flux.2");
    }
    return !family.startsWith("flux.2");
  }, [status, useCustom, preset.family]);

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4 sm:p-6">
      <SectionCard
        icon={<HugeiconsIcon icon={GpuIcon} className="size-5" strokeWidth={1.5} />}
        title="Local image generation"
        description={
          "Run diffusion GGUFs from Hugging Face on your own GPU. " +
          "Pick a curated FLUX.2 model or paste any unsloth/* GGUF repo."
        }
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-2">
            <Label>Model</Label>
            <Select
              value={useCustom ? "custom" : String(presetIndex)}
              onValueChange={(v) => {
                if (v === "custom") {
                  setUseCustom(true);
                } else {
                  setUseCustom(false);
                  setPresetIndex(Number(v));
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Pick a model" />
              </SelectTrigger>
              <SelectContent>
                {CURATED_MODELS.map((m, idx) => (
                  <SelectItem key={m.repo_id} value={String(idx)}>
                    {m.label}
                  </SelectItem>
                ))}
                <SelectItem value="custom">Custom HF repo...</SelectItem>
              </SelectContent>
            </Select>
            {!useCustom && (
              <p className="text-xs text-muted-foreground">{preset.notes}</p>
            )}
          </div>

          {useCustom && (
            <div className="flex flex-col gap-2">
              <Label>HF repo id</Label>
              <Input
                value={customRepoId}
                onChange={(e) => setCustomRepoId(e.target.value)}
                placeholder="unsloth/FLUX.2-klein-4B-GGUF"
              />
              <Label>GGUF filename (optional)</Label>
              <Input
                value={customGguf}
                onChange={(e) => setCustomGguf(e.target.value)}
                placeholder="FLUX.2-klein-4B-Q4_K_S.gguf"
              />
              <Label>Pipeline family (override)</Label>
              <Select
                value={customFamily}
                onValueChange={setCustomFamily}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect from repo id</SelectItem>
                  <SelectItem value="flux.2-klein">FLUX.2 klein</SelectItem>
                  <SelectItem value="flux.2">FLUX.2</SelectItem>
                  <SelectItem value="flux.1">FLUX.1</SelectItem>
                  <SelectItem value="qwen-image">Qwen-Image</SelectItem>
                  <SelectItem value="stable-diffusion-3">Stable Diffusion 3</SelectItem>
                  <SelectItem value="stable-diffusion-xl">Stable Diffusion XL</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {"Set this when your repo name does not contain "}
                {"a recognised family substring (e.g. private fine-tunes)."}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <Label>Hugging Face token (only for gated repos)</Label>
            <Input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              autoComplete="off"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              onClick={handleLoad}
              disabled={busy !== "idle"}
              data-testid="diffusion-load"
            >
              {busy === "loading" ? <Spinner className="mr-2 size-4" /> : null}
              Load model
            </Button>
            <Button
              variant="outline"
              onClick={handleUnload}
              disabled={busy !== "idle" || !status?.is_loaded}
              data-testid="diffusion-unload"
            >
              Unload
            </Button>
            <Button
              variant="ghost"
              onClick={() => void refreshStatus()}
              disabled={refreshingStatus}
            >
              Refresh status
            </Button>
            <span
              className="ml-auto text-xs text-muted-foreground"
              data-testid="diffusion-status"
            >
              {statusLabel}
            </span>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        icon={<HugeiconsIcon icon={PaintBrush02Icon} className="size-5" strokeWidth={1.5} />}
        title="Prompt"
        description="The pipeline runs on the GPU you launched Unsloth Studio on."
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <Label htmlFor="diffusion-prompt">Prompt</Label>
            <Textarea
              id="diffusion-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              data-testid="diffusion-prompt"
            />
          </div>
          {supportsNegativePrompt ? (
            <div className="flex flex-col gap-1">
              <Label htmlFor="diffusion-negative">Negative prompt (optional)</Label>
              <Textarea
                id="diffusion-negative"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
              />
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {"FLUX.2 and FLUX.2 klein do not accept a negative prompt. "}
              {"Steer the output via the main prompt instead."}
            </p>
          )}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="flex flex-col gap-1">
              <Label>Resolution</Label>
              <Select
                value={String(resolutionIdx)}
                onValueChange={(v) => setResolutionIdx(Number(v))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RESOLUTION_PRESETS.map((r, idx) => (
                    <SelectItem key={r.label} value={String(idx)}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <Label>Steps: {steps}</Label>
              <Slider
                min={1}
                max={60}
                step={1}
                value={[steps]}
                onValueChange={(v) => setSteps(v[0] ?? steps)}
              />
            </div>
            <div className="flex flex-col gap-1">
              <Label>Guidance: {guidance.toFixed(1)}</Label>
              <Slider
                min={0}
                max={15}
                step={0.1}
                value={[guidance]}
                onValueChange={(v) => setGuidance(v[0] ?? guidance)}
              />
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <Label htmlFor="diffusion-seed">Seed (optional)</Label>
            <Input
              id="diffusion-seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="leave empty for random"
              inputMode="numeric"
            />
          </div>

          <div>
            <Button
              size="lg"
              onClick={handleGenerate}
              disabled={busy !== "idle" || !status?.is_loaded}
              data-testid="diffusion-generate"
            >
              {busy === "generating" ? <Spinner className="mr-2 size-4" /> : null}
              Generate image
            </Button>
          </div>
        </div>
      </SectionCard>

      {results.length > 0 && (
        <SectionCard
          icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" strokeWidth={1.5} />}
          title="Results"
          description="Most recent first."
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {results.map((r, idx) => (
              <figure key={idx} className="flex flex-col gap-2">
                <img
                  src={`data:${r.image_mime};base64,${r.image_b64}`}
                  alt={`Generated image ${idx + 1}`}
                  // h-auto + object-contain so portrait / landscape
                  // outputs render at their true aspect ratio instead
                  // of being cropped into a square thumbnail.
                  className="h-auto w-full rounded-md border border-border object-contain"
                  data-testid="diffusion-result-image"
                />
                <figcaption className="text-xs text-muted-foreground">
                  {r.width}x{r.height} - {r.num_inference_steps} steps - g={r.guidance_scale.toFixed(1)}
                  {r.seed !== null && r.seed !== undefined ? ` - seed ${r.seed}` : ""} -
                  {` ${(r.duration_ms / 1000).toFixed(1)}s`}
                </figcaption>
              </figure>
            ))}
          </div>
        </SectionCard>
      )}
    </div>
  );
}
