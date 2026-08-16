import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { streamAgentCompletion, streamAgentRun } from "../agent-stream";
import { platformTestServer } from "./test-server";

const encoder = new TextEncoder();

function sse(frames: string[]) {
  return new HttpResponse(
    new ReadableStream({
      start(controller) {
        for (const frame of frames)
          controller.enqueue(encoder.encode(`data: ${frame}\n\n`));
        controller.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
}

describe("Rag Platform Phase 11 agent streams", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("normalizes native completion events and terminal frames", async () => {
    let requestBody: unknown;
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/agents/chat/completions",
        async ({ request }) => {
          requestBody = await request.json();
          return sse([
            JSON.stringify({
              event: "message",
              data: "Merhaba",
              message_id: "msg-1",
              session_id: "session-1",
            }),
            "[DONE]",
          ]);
        },
      ),
    );
    const events = [];
    for await (const event of streamAgentCompletion({
      agentId: "agent-1",
      query: "Selam",
      returnTrace: true,
    }))
      events.push(event);
    expect(requestBody).toEqual({
      agent_id: "agent-1",
      query: "Selam",
      stream: true,
      return_trace: true,
    });
    expect(events).toEqual([
      {
        type: "event",
        event: "message",
        data: "Merhaba",
        messageId: "msg-1",
        sessionId: "session-1",
      },
      { type: "done", messageId: "msg-1", sessionId: "session-1" },
    ]);
  });

  it("sends run query/body exactly and rejects an incomplete stream", async () => {
    let url = "";
    let body: unknown;
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/agents/agent-1/run",
        async ({ request }) => {
          url = request.url;
          body = await request.json();
          return sse([JSON.stringify({ event: "message", data: "partial" })]);
        },
      ),
    );
    const consume = async () => {
      for await (const event of streamAgentRun({
        agentId: "agent-1",
        userInput: "go",
        sessionId: "session-1",
        version: "v1",
      })) {
        void event;
      }
    };
    await expect(consume()).rejects.toMatchObject({
      code: "INCOMPLETE_AGENT_STREAM",
    });
    expect(new URL(url).search).toBe("?session_id=session-1&version=v1");
    expect(body).toEqual({ user_input: "go" });
  });

  it("cancels the reader when the caller aborts", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/agents/chat/completions", () =>
        sse([JSON.stringify({ event: "message", data: "one" }), "[DONE]"]),
      ),
    );
    const controller = new AbortController();
    controller.abort();
    const consume = async () => {
      for await (const event of streamAgentCompletion(
        { agentId: "agent-1", query: "x" },
        controller.signal,
      )) {
        void event;
      }
    };
    await expect(consume()).rejects.toMatchObject({ code: "CLIENT_ABORTED" });
  });
});
