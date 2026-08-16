import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixtureText from "../../../../../../docs/rag-platform/fixtures/phase-8-chat-contract.json?raw";
import {
  PLATFORM_CHAT_AUDIO_MAX_BYTES,
  generatePlatformMindMap,
  getPlatformRecommendations,
  mapPlatformChatReference,
  platformChatCitations,
  streamPlatformChatCompletion,
  synthesizePlatformChatSpeech,
  transcribePlatformChatAudio,
  updatePlatformMessageFeedback,
} from "../chat-completion-api";
import type { PlatformChatStreamEvent } from "../chat-completion-types";
import { platformTestServer } from "./test-server";

const fixture = JSON.parse(fixtureText) as {
  request: Record<string, unknown>;
  incremental_frames: Array<Record<string, unknown>>;
  legacy_cumulative_frames: Array<Record<string, unknown>>;
  business_error: Record<string, unknown>;
};
const ok = (data: unknown) => HttpResponse.json({ code: 0, data });

function eventStream(frames: unknown[], close = true): Response {
  const encoder = new TextEncoder();
  const pieces = frames.map(
    (frame) => `data: ${JSON.stringify(frame)}\r\n\r\n`,
  );
  return new HttpResponse(
    new ReadableStream({
      start(controller) {
        const wire = pieces.join("");
        for (let index = 0; index < wire.length; index += 7) {
          controller.enqueue(encoder.encode(wire.slice(index, index + 7)));
        }
        if (close) controller.close();
      },
    }),
    { headers: { "content-type": "text/event-stream; charset=utf-8" } },
  );
}

async function collect(legacy = false): Promise<PlatformChatStreamEvent[]> {
  const events: PlatformChatStreamEvent[] = [];
  for await (const event of streamPlatformChatCompletion({
    chatId: "chat-1",
    sessionId: "session-1",
    question: "What is retrieval?",
    legacy,
  })) {
    events.push(event);
  }
  return events;
}

describe("Rag Platform Phase 8 native chat contract", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
  });
  afterEach(() => vi.unstubAllEnvs());

  it("sends explicit Chat and Session ids and normalizes fragmented native SSE", async () => {
    let payload: unknown;
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/chat/completions",
        async ({ request }) => {
          payload = await request.json();
          return eventStream(fixture.incremental_frames);
        },
      ),
    );

    const events = await collect();
    expect(payload).toEqual(fixture.request);
    expect(events).toContainEqual({ type: "reasoning-start" });
    expect(events).toContainEqual({
      type: "reasoning-delta",
      delta: "Check evidence.",
      text: "Check evidence.",
    });
    expect(events).toContainEqual(
      expect.objectContaining({ type: "reference-update" }),
    );
    expect(events.find((event) => event.type === "reference-update")).toMatchObject({
      reference: {
        chunks: [
          expect.objectContaining({
            chunkId: "chunk-1",
            documentId: "doc-1",
            filename: "Guide.pdf",
          }),
        ],
      },
    });
    expect(events).toContainEqual({
      type: "usage",
      usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 },
    });
    expect(events.at(-1)).toMatchObject({
      type: "final",
      terminal: true,
      messageId: "turn-1",
      text: "Retrieval finds context.",
    });
  });

  it("normalizes legacy retrieval aliases into visible document citations", () => {
    const reference = mapPlatformChatReference({
      chunks: [
        {
          chunk_id: "chunk-legacy",
          doc_id: "doc-legacy",
          docnm_kwd: "Legacy.pdf",
          content_with_weight: "Evidence",
          position_int: [[7, 0]],
          vector_similarity: 0.82,
        },
      ],
    });

    expect(platformChatCitations(reference)).toEqual([
      expect.objectContaining({
        chunkId: "chunk-legacy",
        documentId: "doc-legacy",
        filename: "Legacy.pdf",
        page: 7,
      }),
    ]);
  });

  it("diffs legacy cumulative answers instead of duplicating prefixes", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/completions", () =>
        eventStream(fixture.legacy_cumulative_frames),
      ),
    );
    const events = await collect(true);
    expect(events.filter((event) => event.type === "text-delta")).toEqual([
      { type: "text-delta", delta: "One", text: "One" },
      { type: "text-delta", delta: " two", text: "One two" },
    ]);
  });

  it("deduplicates retried native frames", async () => {
    const frame = { code: 0, data: { id: "turn-1", answer: "Once" } };
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/completions", () =>
        eventStream([frame, frame, { code: 0, data: true }]),
      ),
    );
    const events = await collect();
    expect(events.filter((event) => event.type === "text-delta")).toEqual([
      { type: "text-delta", delta: "Once", text: "Once" },
    ]);
  });

  it("aborts an open response without reporting a final event", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/completions", () =>
        eventStream(
          [{ code: 0, data: { id: "turn-1", answer: "Partial" } }],
          false,
        ),
      ),
    );
    const controller = new AbortController();
    const generator = streamPlatformChatCompletion(
      {
        chatId: "chat-1",
        sessionId: "session-1",
        question: "Question",
      },
      controller.signal,
    );
    await expect(generator.next()).resolves.toMatchObject({
      value: { type: "text-delta", text: "Partial" },
    });
    controller.abort();
    await expect(generator.next()).rejects.toBeDefined();
  });

  it("never reports success when the stream drops before data:true", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/completions", () =>
        eventStream(fixture.incremental_frames.slice(0, -1)),
      ),
    );
    await expect(collect()).rejects.toMatchObject({
      code: "STREAM_INTERRUPTED",
    });
  });

  it("surfaces backend business errors from the stream", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/completions", () =>
        eventStream([fixture.business_error]),
      ),
    );
    await expect(collect()).rejects.toMatchObject({ code: 100 });
  });

  it("uses exact feedback, mindmap and recommendation contracts", async () => {
    const bodies: unknown[] = [];
    platformTestServer.use(
      http.put(
        "http://platform.test/api/v1/chats/chat-1/sessions/session-1/messages/turn-1/feedback",
        async ({ request }) => {
          bodies.push(await request.json());
          return ok({ id: "session-1", chat_id: "chat-1" });
        },
      ),
      http.post(
        "http://platform.test/api/v1/chat/mindmap",
        async ({ request }) => {
          bodies.push(await request.json());
          return ok({ id: "root", children: [{ id: "branch", children: [] }] });
        },
      ),
      http.post(
        "http://platform.test/api/v1/chat/recommendation",
        async ({ request }) => {
          bodies.push(await request.json());
          return ok(["Follow up?"]);
        },
      ),
    );

    await updatePlatformMessageFeedback("chat-1", "session-1", "turn-1", {
      thumbup: false,
      feedback: "Missing detail",
    });
    await expect(
      generatePlatformMindMap("Question", ["dataset-1"]),
    ).resolves.toEqual({
      id: "root",
      label: "root",
      children: [{ id: "branch", label: "branch", children: [] }],
    });
    await expect(getPlatformRecommendations("Question")).resolves.toEqual([
      "Follow up?",
    ]);
    expect(bodies).toEqual([
      { thumbup: false, feedback: "Missing detail" },
      { question: "Question", kb_ids: ["dataset-1"] },
      { question: "Question" },
    ]);
  });

  it("handles speech blobs and transcription multipart without persisting audio", async () => {
    const captured: { transcription: FormData | null } = {
      transcription: null,
    };
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/chat/audio/speech",
        async ({ request }) => {
          expect(await request.json()).toEqual({ text: "Read this" });
          return new HttpResponse(new Uint8Array([1, 2, 3]), {
            headers: { "content-type": "audio/mpeg" },
          });
        },
      ),
      http.post(
        "http://platform.test/api/v1/chat/audio/transcription",
        async ({ request }) => {
          captured.transcription = await request.formData();
          return ok({ text: " transcript " });
        },
      ),
    );

    await expect(
      synthesizePlatformChatSpeech("Read this"),
    ).resolves.toMatchObject({
      size: 3,
    });
    await expect(
      transcribePlatformChatAudio(new Blob(["voice"], { type: "audio/webm" })),
    ).resolves.toBe("transcript");
    expect(captured.transcription?.get("stream")).toBe("false");
    expect(captured.transcription?.get("file")).toMatchObject({
      size: 9,
      type: "audio/webm",
    });
    await expect(
      transcribePlatformChatAudio(new Blob(["x"], { type: "video/mp4" })),
    ).rejects.toThrow("desteklemiyor");
    await expect(
      transcribePlatformChatAudio(
        new Blob([new Uint8Array(PLATFORM_CHAT_AUDIO_MAX_BYTES + 1)], {
          type: "audio/wav",
        }),
      ),
    ).rejects.toThrow("25 MB");
  });

  it("surfaces audio capability mismatch and supports abort", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/audio/speech", () =>
        HttpResponse.json({
          code: 102,
          message: "TTS model is not configured",
          data: false,
        }),
      ),
      http.post("http://platform.test/api/v1/chat/audio/transcription", () =>
        HttpResponse.json({
          code: 102,
          message: "STT model is not configured",
          data: false,
        }),
      ),
    );
    await expect(
      synthesizePlatformChatSpeech("Read this"),
    ).rejects.toMatchObject({
      code: 102,
    });
    await expect(
      transcribePlatformChatAudio(new Blob(["voice"], { type: "audio/wav" })),
    ).rejects.toMatchObject({ code: 102 });

    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/audio/speech", () =>
        eventStream([], false),
      ),
    );
    const controller = new AbortController();
    const request = synthesizePlatformChatSpeech(
      "Read this",
      controller.signal,
    );
    controller.abort();
    await expect(request).rejects.toMatchObject({ code: "CLIENT_ABORTED" });
  });
});
