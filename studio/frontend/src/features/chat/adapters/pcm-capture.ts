// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Raw PCM capture for recorded dictation.
//
// MediaRecorder only hands over a clip after stop(), and each stop/start costs
// an encoder pipeline. Reading the microphone as Float32 samples lets the
// segmenter cut at a pause and hand that clip to the transcriber while the next
// sentence is still being recorded, with no encoder startup between segments.

import { publishDictationLevel } from "./dictation-level";
import {
  SpeechSegmenter,
  type SpeechSegment,
  type SpeechSegmenterOptions,
} from "./dictation-segmenter";

// ~21 ms at 48 kHz and ~64 ms at 16 kHz: fine enough for a pause detector,
// infrequent enough to keep the main thread free.
const BUFFER_FRAMES = 1024;
const TARGET_SAMPLE_RATE = 16_000;

export interface PcmCaptureOptions extends Omit<
  SpeechSegmenterOptions,
  "sampleRate"
> {
  onSegment: (segment: SpeechSegment) => void;
  onFrame?: (rawRms: number, level: number) => void;
}

function createCaptureContext(): AudioContext {
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  if (!Ctx) throw new Error("Web Audio is not available in this browser.");
  try {
    return new Ctx({ sampleRate: TARGET_SAMPLE_RATE });
  } catch {
    // Some engines only open at the device rate. The WAV header carries that
    // rate and the backend resamples, so the samples are still correct.
    return new Ctx();
  }
}

// One context for every recording in the app. Opening one per dictation cost a
// fresh audio thread and a resume each time the user pressed the shortcut, which
// is latency paid before a single sample arrives. It is never closed: a context
// with nothing connected is idle, and the graph is disconnected on stop.
let sharedContext: AudioContext | null = null;

function captureContext(): AudioContext {
  if (sharedContext && sharedContext.state !== "closed") return sharedContext;
  sharedContext = createCaptureContext();
  return sharedContext;
}

/** Open the shared audio context ahead of the first recording.
 *
 *  Best-effort and safe to call repeatedly: creating it is what costs, and a
 *  context made before a user gesture simply resumes on the first start. */
export function prewarmPcmCapture(): void {
  try {
    void captureContext()
      .resume()
      .catch(() => {});
  } catch {
    // Web Audio missing entirely is reported by the first real capture.
  }
}

export class PcmCapture {
  readonly sampleRate: number;
  private readonly context: AudioContext;
  private readonly source: MediaStreamAudioSourceNode;
  private readonly processor: ScriptProcessorNode;
  private readonly sink: GainNode;
  private readonly segmenter: SpeechSegmenter;
  private readonly onSegment: (segment: SpeechSegment) => void;
  private readonly onFrame?: (rawRms: number, level: number) => void;
  private running = false;
  private stopped = false;

  constructor(stream: MediaStream, options: PcmCaptureOptions) {
    this.onSegment = options.onSegment;
    this.onFrame = options.onFrame;
    this.context = captureContext();
    this.sampleRate = this.context.sampleRate;
    this.segmenter = new SpeechSegmenter({
      ...options,
      sampleRate: this.sampleRate,
    });
    this.source = this.context.createMediaStreamSource(stream);
    this.processor = this.context.createScriptProcessor(BUFFER_FRAMES, 1, 1);
    this.processor.addEventListener("audioprocess", this.handleAudioProcess);
    // A ScriptProcessor only runs while connected to a destination; route it
    // through silence rather than playing the microphone back.
    this.sink = this.context.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.context.destination);
  }

  private readonly handleAudioProcess = (event: AudioProcessingEvent): void => {
    if (!this.running || this.stopped) return;
    const input = event.inputBuffer.getChannelData(0);
    // getChannelData reuses its buffer; the segmenter copies the frame
    // before retaining it in the pre-roll or a segment.
    const segment = this.segmenter.push(input);
    publishDictationLevel(this.segmenter.level);
    this.onFrame?.(this.segmenter.rawRms, this.segmenter.level);
    if (segment) this.onSegment(segment);
  };

  start(): void {
    if (this.stopped) return;
    this.running = true;
    // A context created before a user gesture may be suspended; resume is
    // best-effort and the mic grant usually resumes it anyway.
    void this.context.resume().catch(() => {});
  }

  /** Stop capturing, flush the segment in progress, and release the audio
   *  graph. The microphone stream itself belongs to the caller, and the shared
   *  context stays open for the next recording. */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.running = false;
    const segment = this.segmenter.flush();
    if (segment) this.onSegment(segment);
    publishDictationLevel(0);
    try {
      this.processor.removeEventListener(
        "audioprocess",
        this.handleAudioProcess,
      );
      this.source.disconnect();
      this.processor.disconnect();
      this.sink.disconnect();
    } catch {
      // Already torn down.
    }
  }
}
