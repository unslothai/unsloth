// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import {
  cancelVideoRestyle,
  fetchVideoRestyleClip,
  fetchVideoRestyleProgress,
  startVideoRestyle,
  type VideoRestyleProgress,
} from "./api";

const POLL_MS = 1000;
const IMAGE_RE = /\.(png|jpe?g|webp|bmp)$/i;

const PHASE_LABEL: Record<string, string> = {
  edges: "Tracing edges",
  loading: "Loading Cosmos3",
  generate: "Restyling",
  decode: "Decoding video",
  export: "Saving",
};

function useObjectUrl(blob: Blob | null): string | null {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob]);
  useEffect(
    () => () => {
      if (url) {
        URL.revokeObjectURL(url);
      }
    },
    [url],
  );
  return url;
}

/** Restyle a short clip with Cosmos3-Nano-Sim-Transfer: the clip's edges keep the motion, the prompt sets the look. */
export function VideoRestylePanel({ active }: { active: boolean }) {
  const [video, setVideo] = useState<File | null>(null);
  const [frames, setFrames] = useState<File[]>([]);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("");
  const [progress, setProgress] = useState<VideoRestyleProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inputClip, setInputClip] = useState<Blob | null>(null);
  const [outputClip, setOutputClip] = useState<Blob | null>(null);
  const inputUrl = useObjectUrl(inputClip);
  const outputUrl = useObjectUrl(outputClip);
  const inputRef = useRef<HTMLVideoElement>(null);
  const outputRef = useRef<HTMLVideoElement>(null);
  const loadedFor = useRef<string | null>(null);

  const running = progress?.active === true;

  const refresh = useCallback(async () => {
    try {
      const next = await fetchVideoRestyleProgress();
      setProgress(next);
      if (next.phase === "error") {
        setError(next.error ?? "Restyle failed.");
      }
      if (next.phase === "done" && next.id && loadedFor.current !== next.id) {
        loadedFor.current = next.id;
        const [input, output] = await Promise.all([
          fetchVideoRestyleClip(next.id, "input"),
          fetchVideoRestyleClip(next.id, "output"),
        ]);
        setInputClip(input);
        setOutputClip(output);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (!active) {
      return;
    }
    const kick = window.setTimeout(() => void refresh(), 0);
    const timer = running ? window.setInterval(() => void refresh(), POLL_MS) : undefined;
    return () => {
      window.clearTimeout(kick);
      window.clearInterval(timer);
    };
  }, [active, running, refresh]);

  const submit = async () => {
    setError(null);
    setInputClip(null);
    setOutputClip(null);
    try {
      const parsedSeed = seed.trim() === "" ? null : Number.parseInt(seed, 10);
      setProgress(
        await startVideoRestyle({
          prompt,
          seed: Number.isFinite(parsedSeed) ? parsedSeed : null,
          video,
          frames,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // Keep the two players on the same frame.
  const syncPlay = (play: boolean) => {
    for (const el of [inputRef.current, outputRef.current]) {
      if (!el) continue;
      if (play) {
        el.currentTime = 0;
        void el.play();
      } else {
        el.pause();
      }
    }
  };

  const source = video ? video.name : frames.length ? `${frames.length} frames` : null;
  const step = progress?.step ?? 0;
  const total = progress?.total ?? 0;
  const phase = progress?.phase ?? null;

  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto @[50rem]:flex-row @[50rem]:overflow-hidden">
      <div className="flex w-full shrink-0 flex-col gap-4 border-b border-border/60 px-10 pt-9 pb-6 max-sm:px-5 @[50rem]:w-[408px] @[50rem]:overflow-y-auto @[50rem]:border-r @[50rem]:border-b-0">
        <div className="grid gap-1.5">
          <h2 className="font-heading text-xl font-medium leading-none text-foreground">
            Restyle a video
          </h2>
          <p className="text-xs text-muted-foreground">
            Cosmos3-Nano-Sim-Transfer keeps the clip's shapes and motion (from its edges) and
            repaints it from your prompt. Up to 121 frames at 832 x 480, 30 fps.
          </p>
        </div>

        <div className="grid gap-2">
          <Label>Input</Label>
          <div className="flex flex-wrap gap-2">
            <Button asChild={true} variant="outline" size="sm" disabled={running}>
              <label>
                Choose video
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => {
                    setVideo(e.target.files?.[0] ?? null);
                    setFrames([]);
                    e.target.value = "";
                  }}
                />
              </label>
            </Button>
            <Button asChild={true} variant="outline" size="sm" disabled={running}>
              <label>
                Choose frame folder
                <input
                  type="file"
                  multiple={true}
                  className="hidden"
                  {...({ webkitdirectory: "", directory: "" } as object)}
                  onChange={(e) => {
                    const picked = Array.from(e.target.files ?? []).filter((f) =>
                      IMAGE_RE.test(f.name),
                    );
                    setFrames(picked);
                    setVideo(null);
                    e.target.value = "";
                  }}
                />
              </label>
            </Button>
          </div>
          <span className="truncate text-xs text-muted-foreground">
            {source ?? "No input chosen"}
          </span>
        </div>

        <div className="grid gap-2">
          <Label htmlFor="restyle-prompt">Prompt</Label>
          <Textarea
            id="restyle-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Make it a red rally car at golden hour"
            rows={4}
            disabled={running}
          />
        </div>

        <div className="grid gap-2">
          <Label htmlFor="restyle-seed">Seed</Label>
          <Input
            id="restyle-seed"
            inputMode="numeric"
            value={seed}
            onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
            placeholder="Random"
            disabled={running}
          />
        </div>

        {running ? (
          <Button variant="outline" onClick={() => void cancelVideoRestyle()}>
            Cancel
          </Button>
        ) : (
          <Button disabled={!source || !prompt.trim()} onClick={() => void submit()}>
            Restyle
          </Button>
        )}

        {running && phase && (
          <div className="grid gap-1.5" data-testid="restyle-progress">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {phase === "generate" && step === 0
                  ? "Encoding the edge video"
                  : (PHASE_LABEL[phase] ?? phase)}
              </span>
              {phase === "generate" && total > 0 && (
                <span>
                  Latent frame {step} / {total}
                  {progress?.eta_seconds != null && ` · ~${Math.ceil(progress.eta_seconds)}s`}
                </span>
              )}
            </div>
            <Progress
              value={phase === "generate" && total > 0 ? (100 * step) / total : undefined}
              indeterminate={phase !== "generate"}
            />
          </div>
        )}
        {phase === "cancelled" && !running && (
          <p className="text-xs text-muted-foreground">Cancelled.</p>
        )}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      <div className="flex min-h-[60dvh] min-w-0 flex-1 flex-col items-center justify-center gap-4 p-6 @[50rem]:min-h-0">
        {inputUrl && outputUrl ? (
          <>
            <div className="grid w-full max-w-6xl grid-cols-1 gap-3 @[60rem]:grid-cols-2">
              {[
                { label: "Input", url: inputUrl, ref: inputRef },
                { label: "Restyled", url: outputUrl, ref: outputRef },
              ].map((clip) => (
                <figure key={clip.label} className="grid gap-1.5">
                  <video
                    ref={clip.ref}
                    src={clip.url}
                    muted={true}
                    loop={true}
                    playsInline={true}
                    controls={true}
                    className="w-full rounded-lg bg-black"
                  />
                  <figcaption className="text-xs text-muted-foreground">{clip.label}</figcaption>
                </figure>
              ))}
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => syncPlay(true)}>
                Play both
              </Button>
              <Button size="sm" variant="outline" onClick={() => syncPlay(false)}>
                Pause
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            The input and the restyled clip play here side by side.
          </p>
        )}
      </div>
    </div>
  );
}
