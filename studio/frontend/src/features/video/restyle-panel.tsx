// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { notifyGalleryChanged } from "@/lib/gallery-flags";
import {
  cancelVideoRestyle,
  fetchVideoRestyleClip,
  fetchVideoRestyleProgress,
  fetchVideoRestyleStatus,
  previewVideoRestyle,
  startVideoRestyle,
  type VideoRestylePreview,
  type VideoRestyleProgress,
  type VideoRestyleStatus,
} from "./api";

const POLL_MS = 1000;
const MAX_FRAMES = 121;
const IMAGE_RE = /\.(png|jpe?g|webp|bmp)$/i;

const PHASE_LABEL: Record<string, string> = {
  edges: "Tracing edges",
  loading: "Loading Cosmos3 (about a minute)",
  generate: "Restyling",
  decode: "Decoding video",
  export: "Saving",
};

const CLIPS = [
  { kind: "input", label: "Input" },
  { kind: "control", label: "Edges (what the model follows)" },
  { kind: "output", label: "Restyled" },
] as const;

// Cosmos3 makes one latent frame per 4 output frames (the first stands alone).
function outputFrames(step: number, total?: number): number {
  const done = step <= 0 ? 0 : 1 + 4 * (step - 1);
  return total ? Math.min(done, total) : done;
}

function seconds(value: number): string {
  return `${value.toFixed(1)} s`;
}

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

export function VideoRestylePanel({ active }: { active: boolean }) {
  const [video, setVideo] = useState<File | null>(null);
  const [frames, setFrames] = useState<File[]>([]);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("");
  const [progress, setProgress] = useState<VideoRestyleProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<VideoRestyleStatus | null>(null);
  const [preview, setPreview] = useState<VideoRestylePreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [outcomeStale, setOutcomeStale] = useState(false);
  const [inputClip, setInputClip] = useState<Blob | null>(null);
  const [controlClip, setControlClip] = useState<Blob | null>(null);
  const [outputClip, setOutputClip] = useState<Blob | null>(null);
  const [retrying, setRetrying] = useState(false);
  const urls = {
    input: useObjectUrl(inputClip),
    control: useObjectUrl(controlClip),
    output: useObjectUrl(outputClip),
  };
  const players = useRef<Record<string, HTMLVideoElement | null>>({});
  const loadedFor = useRef<string | null>(null);
  // The newest job progress named; a clip chosen after it marks that job's result as discarded.
  const lastJob = useRef<string | null>(null);
  const previewFor = useRef(0);
  // Bumped whenever the shown clips are cleared, so an older download cannot repopulate them.
  const clipsFor = useRef(0);

  const running = progress?.active === true || progress?.pending === true;
  if (!running && cancelling) setCancelling(false);
  const unavailable = status !== null && !status.available;

  const refresh = useCallback(async () => {
    // Taken before the progress request: a clip chosen while it is in flight voids its result.
    const ticket = clipsFor.current;
    try {
      const next = await fetchVideoRestyleProgress();
      if (next.id) lastJob.current = next.id;
      setProgress(next);
      if (next.phase === "error") {
        setError(next.error ?? "Restyle failed.");
      }
      if (ticket !== clipsFor.current) {
        // A clip was chosen meanwhile: a finished job this stale answer names was discarded with it.
        if (next.phase === "done" && next.id) loadedFor.current = next.id;
        return;
      }
      if (next.phase === "done" && next.id && loadedFor.current !== next.id) {
        loadedFor.current = next.id;
        if (next.video) notifyGalleryChanged("videos");
        let clips: Blob[];
        try {
          clips = await Promise.all(
            CLIPS.map((clip) => fetchVideoRestyleClip(next.id as string, clip.kind)),
          );
        } catch (err) {
          // Retry only for the selection that asked: a result a new clip discarded stays discarded.
          if (loadedFor.current === next.id && ticket === clipsFor.current) loadedFor.current = null;
          throw err;
        }
        const [input, control, output] = clips;
        if (ticket !== clipsFor.current) return;
        setInputClip(input);
        setControlClip(control);
        setOutputClip(output);
      }
      setRetrying(false);
    } catch (err) {
      setRetrying(true);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (!active) {
      return;
    }
    fetchVideoRestyleStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, [active]);

  const choose = async (next: { video: File | null; frames: File[] }) => {
    if (running || submitting) return;
    setVideo(next.video);
    setFrames(next.frames);
    setPreview(null);
    setError(null);
    setOutcomeStale(true);
    clipsFor.current += 1;
    loadedFor.current = lastJob.current;
    setInputClip(null);
    setControlClip(null);
    setOutputClip(null);
    if (!next.video && next.frames.length === 0) {
      return;
    }
    const ticket = ++previewFor.current;
    setPreviewing(true);
    try {
      const result = await previewVideoRestyle(next);
      if (ticket === previewFor.current) setPreview(result);
    } catch (err) {
      if (ticket === previewFor.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (ticket === previewFor.current) setPreviewing(false);
    }
  };

  useEffect(() => {
    if (!active) {
      return;
    }
    const kick = window.setTimeout(() => void refresh(), 0);
    const timer =
      running || retrying ? window.setInterval(() => void refresh(), POLL_MS) : undefined;
    return () => {
      window.clearTimeout(kick);
      window.clearInterval(timer);
    };
  }, [active, running, retrying, refresh]);

  const submit = async () => {
    if (submitting) return;
    setSubmitting(true);
    setOutcomeStale(false);
    setError(null);
    clipsFor.current += 1;
    setInputClip(null);
    setControlClip(null);
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
      // The start may have been accepted with its response lost: poll until progress answers.
      setRetrying(true);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const syncPlay = (play: boolean) => {
    for (const el of Object.values(players.current)) {
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
  const busy = running || submitting;
  const trimmed =
    preview?.source_seconds != null && preview.source_seconds > preview.seconds + 0.05;

  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto @[50rem]:flex-row @[50rem]:overflow-hidden">
      <div className="flex w-full shrink-0 flex-col gap-4 border-b border-border/60 px-10 pt-9 pb-6 max-sm:px-5 @[50rem]:w-[min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100%-var(--media-header-min-width)))] @[50rem]:overflow-y-auto @[50rem]:border-r @[50rem]:border-b-0">
        <div className="grid gap-1.5">
          <h2 className="font-heading text-xl font-medium leading-none text-foreground">
            Restyle a video
          </h2>
          <p className="text-xs text-muted-foreground">
            Cosmos3-Nano-Sim-Transfer keeps the clip's shapes and motion (from its edges) and
            repaints it from your prompt. Uses up to the first 4 seconds, at 832 x 480 and 30 fps.
          </p>
        </div>

        {unavailable && (
          <Alert data-testid="restyle-unavailable">
            <AlertTitle>Restyle is not set up here</AlertTitle>
            <AlertDescription>
              {status?.reason} It needs an NVIDIA GPU with about 40 GB of memory and has only been
              tested on Linux.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid gap-2">
          <Label>Input</Label>
          <div className="flex flex-wrap gap-2">
            <Button asChild={true} variant="outline" size="sm" disabled={busy}>
              <label>
                Choose video
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  disabled={busy}
                  onChange={(e) => {
                    void choose({ video: e.target.files?.[0] ?? null, frames: [] });
                    e.target.value = "";
                  }}
                />
              </label>
            </Button>
            <Button asChild={true} variant="outline" size="sm" disabled={busy}>
              <label>
                Choose frame folder
                <input
                  type="file"
                  multiple={true}
                  className="hidden"
                  disabled={busy}
                  {...({ webkitdirectory: "", directory: "" } as object)}
                  onChange={(e) => {
                    const picked = Array.from(e.target.files ?? [])
                      .filter((f) => IMAGE_RE.test(f.name))
                      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
                      .slice(0, MAX_FRAMES);
                    void choose({ video: null, frames: picked });
                    e.target.value = "";
                  }}
                />
              </label>
            </Button>
          </div>
          <span className="truncate text-xs text-muted-foreground">
            {source ?? "No input chosen"}
          </span>
          {previewing && (
            <span className="text-xs text-muted-foreground">Reading the clip...</span>
          )}
          {preview && (
            <div className="grid gap-1.5" data-testid="restyle-preview">
              <div className="grid grid-cols-2 gap-2">
                <img
                  src={`data:image/jpeg;base64,${preview.frame_jpeg}`}
                  alt="A frame of the input"
                  className="w-full rounded-md bg-black"
                />
                <img
                  src={`data:image/jpeg;base64,${preview.edges_jpeg}`}
                  alt="The edges the model follows for that frame"
                  className="w-full rounded-md bg-black"
                />
              </div>
              <span className="text-xs text-muted-foreground">
                Control: edges (Canny). Uses {preview.frames} frames, {seconds(preview.seconds)}
                {trimmed && ` of a ${seconds(preview.source_seconds as number)} clip`}.
              </span>
            </div>
          )}
        </div>

        <div className="grid gap-2">
          <Label htmlFor="restyle-prompt">Prompt</Label>
          <Textarea
            id="restyle-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Make it a red rally car at golden hour"
            rows={4}
            disabled={busy}
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
            disabled={busy}
          />
        </div>

        {running ? (
          <Button
            variant="outline"
            className="h-11 text-ui-14"
            disabled={cancelling}
            onClick={() => {
              setCancelling(true);
              // A pending start cannot be cancelled yet (cancelled: false): keep Cancel usable.
              void cancelVideoRestyle()
                .then((r) => {
                  if (!r.cancelled) setCancelling(false);
                })
                .catch(() => setCancelling(false));
            }}
          >
            <Spinner className="mr-2 size-4" />
            {cancelling ? "Stopping..." : "Cancel"}
          </Button>
        ) : (
          <Button
            variant="primary"
            className="h-11 text-ui-14 bg-control-accent text-control-accent-foreground hover:bg-control-accent/85 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
            disabled={!source || !prompt.trim() || submitting || previewing || unavailable}
            onClick={() => void submit()}
          >
            {submitting ? "Uploading..." : "Restyle"}
          </Button>
        )}

        {running && phase && (
          <div className="grid gap-1.5" data-testid="restyle-progress">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {cancelling
                  ? "Stopping"
                  : phase === "generate" && step === 0
                  ? "Preparing the edges for the model"
                  : (PHASE_LABEL[phase] ?? phase)}
              </span>
              {phase === "generate" && step > 0 && (
                <span>
                  Frame {outputFrames(step, progress?.num_frames)} of {progress?.num_frames ?? "?"}
                  {!cancelling &&
                    progress?.eta_seconds != null &&
                    ` · about ${Math.ceil(progress.eta_seconds)} s left`}
                </span>
              )}
            </div>
            <Progress
              value={phase === "generate" && total > 0 ? (100 * step) / total : undefined}
              indeterminate={phase !== "generate" || step === 0}
            />
          </div>
        )}
        {phase === "cancelled" && !running && !outcomeStale && (
          <p className="text-xs text-muted-foreground">Cancelled.</p>
        )}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      {/* justify-center-safe: three stacked players can be taller than the pane; plain centering would
          push them up over the form. */}
      <div className="flex min-h-[60dvh] min-w-0 flex-1 flex-col items-center justify-center-safe gap-4 p-6 @[50rem]:min-h-0 @[50rem]:overflow-y-auto">
        {urls.input && urls.control && urls.output ? (
          <>
            <div className="grid w-full max-w-7xl grid-cols-1 gap-3 @[60rem]:grid-cols-3">
              {CLIPS.map((clip) => (
                <figure key={clip.kind} className="grid gap-1.5">
                  <video
                    ref={(el) => {
                      players.current[clip.kind] = el;
                    }}
                    src={urls[clip.kind] ?? undefined}
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
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => syncPlay(true)}>
                Play all
              </Button>
              <Button size="sm" variant="outline" onClick={() => syncPlay(false)}>
                Pause
              </Button>
              {progress?.seed != null && (
                <span className="text-xs text-muted-foreground">Seed {progress.seed}</span>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            The input, its edges and the restyled clip play here side by side.
          </p>
        )}
      </div>
    </div>
  );
}
