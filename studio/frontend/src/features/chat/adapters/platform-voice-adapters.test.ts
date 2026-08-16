import { waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  synthesize: vi.fn(),
  transcribe: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/integrations/platform-backend")>();
  return {
    ...actual,
    synthesizePlatformChatSpeech: mocks.synthesize,
    transcribePlatformChatAudio: mocks.transcribe,
  };
});

import { PlatformDictationAdapter } from "./platform-dictation-adapter";
import { PlatformSpeechSynthesisAdapter } from "./platform-speech-synthesis-adapter";

class FakeAudio extends EventTarget {
  static instances: FakeAudio[] = [];
  playbackRate = 1;
  volume = 1;
  pause = vi.fn();
  load = vi.fn();
  play = vi.fn(async () => undefined);
  removeAttribute = vi.fn();

  constructor() {
    super();
    FakeAudio.instances.push(this);
  }
}

class RecordingMediaRecorder extends EventTarget {
  static instances: RecordingMediaRecorder[] = [];
  static isTypeSupported() {
    return true;
  }
  state = "inactive";
  mimeType = "audio/webm";
  stop = vi.fn(() => {
    this.state = "inactive";
    this.dispatchEvent(new Event("stop"));
  });
  start = vi.fn(() => {
    this.state = "recording";
  });

  constructor() {
    super();
    RecordingMediaRecorder.instances.push(this);
  }
}

describe("Rag Platform Phase 8 voice adapters", () => {
  beforeEach(() => {
    mocks.synthesize.mockReset();
    mocks.transcribe.mockReset();
    FakeAudio.instances = [];
    RecordingMediaRecorder.instances = [];
    vi.stubGlobal("Audio", FakeAudio);
    Object.defineProperty(window, "Audio", {
      configurable: true,
      value: FakeAudio,
    });
    Object.defineProperty(window, "isSecureContext", {
      configurable: true,
      value: true,
    });
  });

  afterEach(() => vi.useRealTimers());

  it("revokes the speech object URL after playback ends", async () => {
    mocks.synthesize.mockResolvedValue(
      new Blob(["audio"], { type: "audio/mpeg" }),
    );
    const createObjectURL = vi.fn(() => "blob:speech");
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectURL,
    });

    const utterance = new PlatformSpeechSynthesisAdapter().speak("Read this");
    await waitFor(() => expect(utterance.status.type).toBe("running"));
    FakeAudio.instances.at(-1)?.dispatchEvent(new Event("ended"));

    expect(utterance.status).toMatchObject({
      type: "ended",
      reason: "finished",
    });
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:speech");
  });

  it("reports microphone permission denial and ends without transcription", async () => {
    class FakeMediaRecorder {
      static isTypeSupported() {
        return true;
      }
    }
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    const denied = new DOMException("denied", "NotAllowedError");
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(denied) },
    });

    const session = new PlatformDictationAdapter().listen();
    const speechEnd = vi.fn();
    session.onSpeechEnd(speechEnd);
    await waitFor(() => expect(session.status.type).toBe("ended"));

    expect(session.status).toMatchObject({ type: "ended", reason: "error" });
    expect(speechEnd).toHaveBeenCalledWith({ transcript: "" });
    expect(mocks.transcribe).not.toHaveBeenCalled();
  });

  it("cancels recording, aborts transcription and releases microphone tracks", async () => {
    vi.stubGlobal("MediaRecorder", RecordingMediaRecorder);
    const stopTrack = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: stopTrack }],
        }),
      },
    });
    const session = new PlatformDictationAdapter().listen();
    await waitFor(() => expect(session.status.type).toBe("running"));
    session.cancel();

    expect(session.status).toMatchObject({
      type: "ended",
      reason: "cancelled",
    });
    expect(stopTrack).toHaveBeenCalledOnce();
    expect(mocks.transcribe).not.toHaveBeenCalled();
  });

  it("automatically stops a recording at the two-minute limit", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("MediaRecorder", RecordingMediaRecorder);
    const stopTrack = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: stopTrack }],
        }),
      },
    });
    mocks.transcribe.mockResolvedValue("Transcript");
    const session = new PlatformDictationAdapter().listen();
    await Promise.resolve();
    await Promise.resolve();
    expect(session.status.type).toBe("running");

    await vi.advanceTimersByTimeAsync(120_000);
    await Promise.resolve();
    expect(
      RecordingMediaRecorder.instances.at(-1)?.stop,
    ).toHaveBeenCalledOnce();
    expect(mocks.transcribe).toHaveBeenCalledOnce();
    expect(stopTrack).toHaveBeenCalledOnce();
    expect(session.status).toMatchObject({ type: "ended", reason: "stopped" });
  });
});
