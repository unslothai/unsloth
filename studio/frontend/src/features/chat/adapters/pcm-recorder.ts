// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Microphone capture for engines whose MediaRecorder cannot encode audio. WebKitGTK, the
 *  webview the Linux desktop build runs in, ships MediaRecorder and resolves `audio/mp4` but
 *  never builds a GStreamer audio encoding profile for it, so every recording comes back with
 *  zero bytes (#9543). Reading raw PCM off a Web Audio graph sidesteps the encoder, and WAV
 *  is the one container the STT backend forwards untouched. */

/** Only the field the recording call sites read off a `dataavailable` event. */
export interface RecordedDataEvent {
  readonly data: Blob;
}

/** The slice of MediaRecorder the recording call sites drive, so a PCM recorder can stand in
 *  for one without changing how they wire it up. */
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
// Apple's WebKit builds also advertise audio/mp4 alone, and there MediaRecorder does encode,
// so the platform is what separates them from WebKitGTK.
const APPLE_WEBKIT = /iPad|iPhone|iPod|Macintosh|Mac OS X/;

/** Whether this engine's MediaRecorder can be trusted to encode audio. Kept a capability test
 *  rather than a Linux check so every engine that does work keeps recording as it does
 *  today, and only the broken one takes the PCM path. */
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
// ~256ms of audio per callback at the target rate: infrequent enough not to keep waking the
// main thread, short enough that a stop cuts promptly.
const BUFFER_FRAMES = 4096;
const BYTES_PER_SAMPLE = 2;
const WAV_HEADER_BYTES = 44;

/** Encode mono float32 samples as a 16-bit PCM WAV. The backend forwards `RIFF....WAVE` to
 *  llama-server untouched and decodes it with no format sniffing on the Transformers path,
 *  so this is both the cheapest and the most widely accepted thing to upload. */
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
    // Clamp first: Web Audio samples are nominally -1..1 but may overshoot, and an out-of-range
    // value wraps to the opposite sign as a loud click.
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
    // Some engines only open a context at the device rate. The backend resamples whatever it is
    // given, so that is still correct, just larger on the wire, which secondsWithin() accounts
    // for.
    return new Ctx();
  }
}

/** Records a MediaStream as a single WAV blob, emitted as one `dataavailable` then `stop`, the
 *  way MediaRecorder reports a finished recording. A ScriptProcessorNode rather than an
 *  AudioWorklet: a worklet needs its processor as a separate module URL, and at one
 *  4096-frame callback per quarter second there is no main-thread cost worth that asset. */
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
      // getChannelData hands back a buffer the engine reuses for the next callback, so it has to be
      // copied rather than retained.
      this.chunks.push(new Float32Array(input));
      this.frames += input.length;
    });
    // A ScriptProcessorNode only runs while it reaches a destination, so route it through a silent
    // gain rather than playing the microphone back.
    this.sink = this.context.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.context.destination);
  }

  get state(): RecordingState {
    return this.recordingState;
  }

  /** Seconds of audio that fit in `maxBytes` once encoded, less a second of slack for the buffer
   *  still in flight when a caller stops on this. */
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

  /** `timesliceMs` is accepted for MediaRecorder parity and ignored: the samples are already
   *  buffered here rather than inside the engine, which is all a timeslice bought the callers. */
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
    // A tap too short to collect a callback reports nothing rather than a header with no samples,
    // so the callers read it as the silence an empty MediaRecorder buffer already reads as.
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
    // Drop the one-shot listeners before dispatching: a `stop` handler is where the callers start
    // the next segment, which registers more of them.
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

/** A recorder for `stream`: MediaRecorder where it encodes, PCM where it does not. `mimeType`
 *  is the MediaRecorder preference and is unused on the PCM path, which always makes WAV. */
export function createAudioRecorder(
  stream: MediaStream,
  mimeType?: string,
): SegmentRecorder {
  if (!mediaRecorderCanEncodeAudio()) {
    return new PcmRecorder(stream);
  }
  return new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
}
