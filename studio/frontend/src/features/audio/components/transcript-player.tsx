// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Ref, useEffect, useState } from "react";
import { fetchAudioBlob } from "../api";
import { sourceFileUrl } from "../audio-run-request";
import { decodePeaks } from "../hooks/use-audio-source";
import type { TranscriptDetails } from "../transcript-model";
import { Waveform, type WaveformControl } from "./waveform";

type PlayerState =
  | { status: "idle" | "loading" | "missing" }
  | {
      status: "ready";
      url: string;
      peaks: number[] | null;
      durationS: number | null;
    };

export function TranscriptPlayer({
  source,
  durationS,
  controlRef,
  onPositionChange,
  onAvailableChange,
}: {
  source: TranscriptDetails["source"];
  durationS: number | null;
  controlRef?: Ref<WaveformControl>;
  onPositionChange?: (seconds: number, playing: boolean) => void;
  onAvailableChange?: (available: boolean) => void;
}) {
  const [state, setState] = useState<PlayerState>({ status: "idle" });
  const kind = source?.kind ?? null;
  const id = source?.id ?? null;

  useEffect(() => {
    if (!(kind && id)) {
      setState({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    let url: string | null = null;
    setState({ status: "loading" });
    fetchAudioBlob(
      sourceFileUrl({ kind, id, name: "", durationS: null }),
      controller.signal,
    )
      .then(async (blob) => {
        const decoded = await decodePeaks(blob);
        if (controller.signal.aborted) return;
        url = URL.createObjectURL(blob);
        setState({ status: "ready", url, ...decoded });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState({ status: "missing" });
      });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [kind, id]);

  const ready = state.status === "ready" ? state : null;
  const available = ready !== null;
  useEffect(() => {
    onAvailableChange?.(available);
  }, [onAvailableChange, available]);

  if (!source) return null;
  if (state.status === "missing") {
    return (
      <p className="text-ui-11p5 leading-snug text-muted-foreground">
        The audio for this transcript is no longer available. Timestamps still
        export.
      </p>
    );
  }
  return (
    <Waveform
      peaks={ready?.peaks ?? null}
      durationS={ready?.durationS ?? durationS}
      src={ready?.url ?? null}
      label={source.name || "Transcript audio"}
      controlRef={controlRef}
      onPositionChange={onPositionChange}
    />
  );
}
