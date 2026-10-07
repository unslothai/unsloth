// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** For engines whose MediaRecorder cannot encode audio: WebKitGTK yields zero-byte recordings
 *  (#9543). Raw PCM to WAV, which the STT backend forwards untouched. */

export interface RecordedDataEvent {
  readonly data: Blob;
}

export interface SegmentRecorder {
  readonly state: RecordingState;
  readonly mimeType: string;
  start(timesliceMs?: number): void;
  stop(): void;
  addEventListener(
    type: "dataavailable",
    listener: (event: RecordedDataEvent) => void,
    options?: AddEventListenerOptions,
  ): void;
  addEventListener(
    type: "stop",
    listener: (event: Event) => void,
    options?: AddEventListenerOptions,
  ): void;
}

// A container the engine can only advertise if it has a working audio encoder.
const OPUS_MIME_TYPES = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus"];
// Apple WebKit also advertises only audio/mp4 but does encode; the platform tells it from WebKitGTK.
const APPLE_WEBKIT = /iPad|iPhone|iPod|Macintosh|Mac OS X/;

/** A capability test, not a Linux check, so only the broken engine takes the PCM path. */
export function mediaRecorderCanEncodeAudio(
  isTypeSupported: (type: string) => boolean = (type) =>
    typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type),
  userAgent: string = typeof navigator === "undefined"
    ? ""
    : navigator.userAgent,
): boolean {
  if (OPUS_MIME_TYPES.some((type) => isTypeSupported(type))) return true;
  return APPLE_WEBKIT.test(userAgent);
}

// Whisper's own rate: the smallest WAV that costs the backend no resample.
const TARGET_SAMPLE_RATE = 16_000;
// About 256ms per callback: few main-thread wakeups, yet a stop cuts promptly.
const BUFFER_FRAMES = 4096;
const BYTES_PER_SAMPLE = 2;
const WAV_HEADER_BYTES = 44;

/** The backend forwards a WAV untouched, so it is the cheapest and most accepted upload. */
export function encodeWav(
  samples: Float32Array,
  sampleRate: number,
): Uint8Array<ArrayBuffer> {
  const dataBytes = samples.length * BYTES_PER_SAMPLE;
  const bytes = new Uint8Array(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(bytes.buffer);
  const writeAscii = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      bytes[offset + index] = text.charCodeAt(index);
    }
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM header length
  view.setUint16(20, 1, true); // uncompressed PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * BYTES_PER_SAMPLE, true); // byte rate
  view.setUint16(32, BYTES_PER_SAMPLE, true); // block align
  view.setUint16(34, 8 * BYTES_PER_SAMPLE, true); // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let index = 0; index < samples.length; index += 1) {
    // Clamp: out-of-range samples wrap to the opposite sign as a loud click.
    const sample = Math.min(1, Math.max(-1, samples[index]));
    view.setInt16(
      WAV_HEADER_BYTES + index * BYTES_PER_SAMPLE,
      sample < 0 ? sample * 0x8000 : sample * 0x7fff,
      true,
    );
  }
  return bytes;
}

function createAudioContext(): AudioContext {
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  try {
    return new Ctx({ sampleRate: TARGET_SAMPLE_RATE });
  } catch {
    // Some engines only open at device rate; the backend resamples, and secondsWithin() covers the size.
    return new Ctx();
  }
}

/** ScriptProcessorNode, not AudioWorklet: a worklet needs a separate module URL for little gain. */
export class PcmRecorder implements SegmentRecorder {
  readonly mimeType = "audio/wav";
  readonly sampleRate: number;
  private recordingState: RecordingState = "inactive";
  private readonly context: AudioContext;
  private readonly source: MediaStreamAudioSourceNode;
  private readonly processor: ScriptProcessorNode;
  private readonly sink: GainNode;
  private readonly chunks: Float32Array[] = [];
  private frames = 0;
  private readonly dataListeners: ((event: RecordedDataEvent) => void)[] = [];
  private readonly stopListeners: {
    listener: (event: Event) => void;
    once: boolean;
  }[] = [];

  constructor(stream: MediaStream) {
    this.context = createAudioContext();
    this.sampleRate = this.context.sampleRate;
    this.source = this.context.createMediaStreamSource(stream);
    this.processor = this.context.createScriptProcessor(BUFFER_FRAMES, 1, 1);
    this.processor.addEventListener("audioprocess", (event) => {
      if (this.recordingState !== "recording") return;
      const input = event.inputBuffer.getChannelData(0);
      // getChannelData's buffer is reused for the next callback, so copy it.
      this.chunks.push(new Float32Array(input));
      this.frames += input.length;
    });
    // A ScriptProcessorNode only runs while it reaches a destination; route through a silent gain.
    this.sink = this.context.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.context.destination);
  }

  get state(): RecordingState {
    return this.recordingState;
  }

  /** Seconds that fit in `maxBytes`, less a second of slack for the buffer still in flight. */
  secondsWithin(maxBytes: number): number {
    const bytesPerSecond = this.sampleRate * BYTES_PER_SAMPLE;
    return Math.max(
      1,
      Math.floor((maxBytes - WAV_HEADER_BYTES) / bytesPerSecond) - 1,
    );
  }

  addEventListener(
    type: "dataavailable",
    listener: (event: RecordedDataEvent) => void,
    options?: AddEventListenerOptions,
  ): void;
  addEventListener(
    type: "stop",
    listener: (event: Event) => void,
    options?: AddEventListenerOptions,
  ): void;
  addEventListener(
    type: "dataavailable" | "stop",
    listener: ((event: RecordedDataEvent) => void) & ((event: Event) => void),
    options?: AddEventListenerOptions,
  ): void {
    if (type === "dataavailable") {
      this.dataListeners.push(listener);
      return;
    }
    this.stopListeners.push({ listener, once: options?.once === true });
  }

  /** `timesliceMs` is ignored; it exists only for MediaRecorder parity. */
  start(_timesliceMs?: number): void {
    if (this.recordingState !== "inactive") return;
    this.recordingState = "recording";
    // Started before a user gesture on some engines; a suspended context delivers no audioprocess callbacks at all.
    this.context.resume().catch(() => {
      // Already running, or resumed on its own once the mic was granted.
    });
  }

  stop(): void {
    if (this.recordingState === "inactive") return;
    this.recordingState = "inactive";
    const samples = new Float32Array(this.frames);
    let offset = 0;
    for (const chunk of this.chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks.length = 0;
    this.frames = 0;
    this.teardown();
    // A tap shorter than one callback yields nothing, which callers read as silence.
    const data =
      samples.length === 0
        ? new Blob([])
        : new Blob([encodeWav(samples, this.sampleRate)], {
            type: this.mimeType,
          });
    const event = { data };
    for (const listener of this.dataListeners) listener(event);
    const stopEvent = new Event("stop");
    const listeners = this.stopListeners.slice();
    // Drop one-shot listeners before dispatch: a stop handler starts the next segment and adds more.
    for (let index = this.stopListeners.length - 1; index >= 0; index -= 1) {
      if (this.stopListeners[index].once) this.stopListeners.splice(index, 1);
    }
    for (const entry of listeners) entry.listener(stopEvent);
  }

  private teardown(): void {
    this.source.disconnect();
    this.processor.disconnect();
    this.sink.disconnect();
    this.context.close().catch(() => {
      // A closing or already-closed context is harmless.
    });
  }
}

/** MediaRecorder where it encodes, otherwise PCM WAV (`mimeType` is unused there). */
export function createAudioRecorder(
  stream: MediaStream,
  mimeType?: string,
): SegmentRecorder {
  if (!mediaRecorderCanEncodeAudio()) {
    return new PcmRecorder(stream);
  }
  return new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
}
