// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// User agents of the engines the capability gate has to separate. WebKitGTK is
// the only one that advertises MediaRecorder, resolves audio/mp4, and then
// records nothing (#9543).
const CHROME_LINUX =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36";
const FIREFOX_LINUX =
  "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0";
const SAFARI_MAC =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
const SAFARI_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";
const WEBKITGTK_LINUX =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";

const supports =
  (...types: string[]) =>
  (type: string) =>
    types.includes(type);

// --- Fake Web Audio graph ----------------------------------------------------
// PcmRecorder only ever touches these five factory methods, so the graph can be
// stubbed rather than pulled in as a DOM implementation.

class FakeNode {
  connections: unknown[] = [];
  disconnected = false;
  connect(target: unknown): void {
    this.connections.push(target);
  }
  disconnect(): void {
    this.disconnected = true;
  }
}

class FakeGainNode extends FakeNode {
  gain = { value: 1 };
}

type AudioProcessListener = (event: {
  inputBuffer: { getChannelData: (channel: number) => Float32Array };
}) => void;

class FakeScriptProcessorNode extends FakeNode {
  listener: AudioProcessListener | null = null;
  addEventListener(_type: string, listener: AudioProcessListener): void {
    this.listener = listener;
  }
  /** Deliver one audioprocess callback carrying `samples`. */
  emit(samples: Float32Array): void {
    this.listener?.({ inputBuffer: { getChannelData: () => samples } });
  }
}

class FakeAudioContext {
  static last: FakeAudioContext | null = null;
  /** Set to reject the requested rate, as an engine that only opens a context
   *  at the device rate does. */
  static refuseRequestedRate = false;
  static requestedRate: number | undefined;

  sampleRate: number;
  destination = new FakeNode();
  processor = new FakeScriptProcessorNode();
  gain = new FakeGainNode();
  resumed = false;
  closed = false;

  constructor(options?: { sampleRate?: number }) {
    FakeAudioContext.requestedRate = options?.sampleRate;
    if (
      options?.sampleRate !== undefined &&
      FakeAudioContext.refuseRequestedRate
    ) {
      throw new Error("unsupported sample rate");
    }
    this.sampleRate = options?.sampleRate ?? 48_000;
    FakeAudioContext.last = this;
  }
  createMediaStreamSource(): FakeNode {
    return new FakeNode();
  }
  createScriptProcessor(): FakeScriptProcessorNode {
    return this.processor;
  }
  createGain(): FakeGainNode {
    return this.gain;
  }
  resume(): Promise<void> {
    this.resumed = true;
    return Promise.resolve();
  }
  close(): Promise<void> {
    this.closed = true;
    return Promise.resolve();
  }
}

Object.assign(globalThis, {
  window: { AudioContext: FakeAudioContext },
});

const { PcmRecorder, encodeWav, mediaRecorderCanEncodeAudio } = await import(
  "../src/features/chat/adapters/pcm-recorder.ts"
);

const newRecorder = () => {
  FakeAudioContext.last = null;
  return new PcmRecorder({} as unknown as MediaStream);
};

const wavOf = async (event: { data: Blob }) =>
  new DataView(await event.data.arrayBuffer());

const ascii = (view: DataView, offset: number, length: number) =>
  String.fromCharCode(
    ...Array.from({ length }, (_, i) => view.getUint8(offset + i)),
  );

// --- The capability gate -----------------------------------------------------

test("an engine that offers an Opus container keeps MediaRecorder", () => {
  assert.equal(
    mediaRecorderCanEncodeAudio(
      supports("audio/webm;codecs=opus", "audio/webm"),
      CHROME_LINUX,
    ),
    true,
  );
  assert.equal(
    mediaRecorderCanEncodeAudio(
      supports("audio/ogg;codecs=opus"),
      FIREFOX_LINUX,
    ),
    true,
  );
});

test("Apple WebKit keeps MediaRecorder despite offering audio/mp4 alone", () => {
  assert.equal(
    mediaRecorderCanEncodeAudio(supports("audio/mp4"), SAFARI_MAC),
    true,
  );
  assert.equal(
    mediaRecorderCanEncodeAudio(supports("audio/mp4"), SAFARI_IOS),
    true,
  );
});

test("WebKitGTK offers audio/mp4 alone off an Apple platform, so it takes PCM", () => {
  assert.equal(
    mediaRecorderCanEncodeAudio(supports("audio/mp4"), WEBKITGTK_LINUX),
    false,
  );
  // The same engine with no MediaRecorder at all must not be treated as able
  // to encode either.
  assert.equal(mediaRecorderCanEncodeAudio(supports(), WEBKITGTK_LINUX), false);
});

// --- WAV encoding ------------------------------------------------------------

test("encodeWav writes a header the backend's container sniff accepts", () => {
  const wav = encodeWav(new Float32Array(8), 16_000);
  const view = new DataView(wav.buffer);
  // _sniff_audio_container() in studio/backend/routes/inference.py forwards a
  // body to llama-server untranscoded on exactly these two fields.
  assert.equal(ascii(view, 0, 4), "RIFF");
  assert.equal(ascii(view, 8, 4), "WAVE");
  assert.equal(ascii(view, 12, 4), "fmt ");
  assert.equal(ascii(view, 36, 4), "data");
  assert.equal(view.getUint32(16, true), 16); // PCM header length
  assert.equal(view.getUint16(20, true), 1); // uncompressed
  assert.equal(view.getUint16(22, true), 1); // mono
  assert.equal(view.getUint32(24, true), 16_000);
  assert.equal(view.getUint32(28, true), 32_000); // byte rate
  assert.equal(view.getUint16(32, true), 2); // block align
  assert.equal(view.getUint16(34, true), 16); // bits per sample
});

test("encodeWav sizes both length fields for the samples it was given", () => {
  const wav = encodeWav(new Float32Array(100), 16_000);
  const view = new DataView(wav.buffer);
  assert.equal(wav.length, 44 + 200);
  assert.equal(view.getUint32(4, true), 36 + 200); // RIFF chunk
  assert.equal(view.getUint32(40, true), 200); // data chunk
});

test("encodeWav clamps rather than wrapping an out-of-range sample", () => {
  const wav = encodeWav(new Float32Array([0, 1, -1, 2.5, -2.5, 0.5]), 16_000);
  const view = new DataView(wav.buffer);
  const sample = (index: number) => view.getInt16(44 + index * 2, true);
  assert.equal(sample(0), 0);
  assert.equal(sample(1), 32_767);
  assert.equal(sample(2), -32_768);
  // Without the clamp these overflow int16 and come back as a loud opposite
  // sign click.
  assert.equal(sample(3), 32_767);
  assert.equal(sample(4), -32_768);
  assert.equal(sample(5), 16_383);
});

// --- PcmRecorder -------------------------------------------------------------

test("a recording is reported as one WAV dataavailable, then stop", async () => {
  const recorder = newRecorder();
  const events: string[] = [];
  let recorded: { data: Blob } | null = null;
  recorder.addEventListener("dataavailable", (event) => {
    events.push("dataavailable");
    recorded = event;
  });
  recorder.addEventListener("stop", () => events.push("stop"));

  assert.equal(recorder.state, "inactive");
  recorder.start(250);
  assert.equal(recorder.state, "recording");
  FakeAudioContext.last?.processor.emit(new Float32Array([1, -1]));
  FakeAudioContext.last?.processor.emit(new Float32Array([0, 0.5]));
  recorder.stop();

  // The callers' stop handler reads the chunks the dataavailable handler
  // pushed, so the order is load-bearing, not cosmetic.
  assert.deepEqual(events, ["dataavailable", "stop"]);
  assert.equal(recorder.state, "inactive");
  assert.ok(recorded);
  const event = recorded as { data: Blob };
  assert.equal(event.data.type, "audio/wav");
  const view = await wavOf(event);
  assert.equal(view.byteLength, 44 + 8);
  assert.equal(view.getInt16(44, true), 32_767);
  assert.equal(view.getInt16(46, true), -32_768);
  assert.equal(view.getInt16(50, true), 16_383);
});

test("samples outside a started recording are not captured", async () => {
  const recorder = newRecorder();
  let recorded: { data: Blob } | null = null;
  recorder.addEventListener("dataavailable", (event) => {
    recorded = event;
  });
  // Before start(): the graph is live from construction, so these must be
  // dropped rather than prepended to the recording.
  FakeAudioContext.last?.processor.emit(new Float32Array([0.25, 0.25]));
  recorder.start();
  FakeAudioContext.last?.processor.emit(new Float32Array([0.5]));
  recorder.stop();
  // After stop(): a late callback must not resurrect a finished recording.
  FakeAudioContext.last?.processor.emit(new Float32Array([0.75, 0.75]));

  const view = await wavOf(recorded as unknown as { data: Blob });
  assert.equal(view.byteLength, 44 + 2);
  assert.equal(view.getInt16(44, true), 16_383);
});

test("stopping releases the audio graph and the context", () => {
  const recorder = newRecorder();
  const context = FakeAudioContext.last;
  recorder.start();
  recorder.stop();
  assert.equal(context?.closed, true);
  assert.equal(context?.processor.disconnected, true);
  assert.equal(context?.gain.disconnected, true);
  // Routed through a muted gain: a ScriptProcessorNode has to reach the
  // destination to run, and the microphone must not be played back.
  assert.equal(context?.gain.gain.value, 0);
});

test("a recording that collected nothing reports no bytes", () => {
  const recorder = newRecorder();
  let recorded: { data: Blob } | null = null;
  recorder.addEventListener("dataavailable", (event) => {
    recorded = event;
  });
  recorder.start();
  recorder.stop();
  // A bare 44-byte header is a WAV with no samples, which the callers would
  // upload and the decoder could refuse; zero bytes is their silence branch.
  assert.equal((recorded as unknown as { data: Blob }).data.size, 0);
});

test("a repeated stop does not emit a second recording", () => {
  const recorder = newRecorder();
  let dataEvents = 0;
  recorder.addEventListener("dataavailable", () => {
    dataEvents += 1;
  });
  recorder.start();
  recorder.stop();
  recorder.stop();
  assert.equal(dataEvents, 1);
});

test("a once stop listener is dropped before the handlers run", () => {
  const recorder = newRecorder();
  let onceCalls = 0;
  let persistentCalls = 0;
  recorder.addEventListener("stop", () => {
    persistentCalls += 1;
    // The dictation adapter opens the next segment from inside a stop handler,
    // which registers further listeners; a one-shot must already be gone.
    recorder.addEventListener("stop", () => {
      persistentCalls += 10;
    });
  });
  recorder.addEventListener(
    "stop",
    () => {
      onceCalls += 1;
    },
    { once: true },
  );
  recorder.start();
  recorder.stop();
  assert.equal(onceCalls, 1);
  assert.equal(persistentCalls, 1);
});

test("capture asks for Whisper's rate and accepts the device rate instead", () => {
  FakeAudioContext.refuseRequestedRate = false;
  const atTargetRate = newRecorder();
  assert.equal(FakeAudioContext.requestedRate, 16_000);
  assert.equal(atTargetRate.sampleRate, 16_000);

  FakeAudioContext.refuseRequestedRate = true;
  try {
    const atDeviceRate = newRecorder();
    assert.equal(atDeviceRate.sampleRate, 48_000);
  } finally {
    FakeAudioContext.refuseRequestedRate = false;
  }
});

test("secondsWithin keeps a WAV inside the upload cap", () => {
  const recorder = newRecorder();
  const cap = 25 * 1024 * 1024;
  const seconds = recorder.secondsWithin(cap);
  // 16-bit mono at 16 kHz is 32000 B/s, so the cap is ~13.6 minutes.
  assert.equal(seconds, 818);
  // A whole recording of that length, plus the buffer still in flight when the
  // caller stops on this, has to stay under the cap.
  const inFlightBytes = 4096 * 2;
  assert.ok(44 + seconds * 32_000 + inFlightBytes < cap);
});

// --- Neither call site may construct a MediaRecorder directly again ----------

test("the recording call sites go through createAudioRecorder", () => {
  for (const path of [
    "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
    "../src/features/audio/audio-page.tsx",
  ]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.equal(
      source.includes("new MediaRecorder("),
      false,
      `${path} must not construct a MediaRecorder directly: WebKitGTK's records nothing`,
    );
    assert.equal(source.includes("createAudioRecorder("), true, path);
  }
});
