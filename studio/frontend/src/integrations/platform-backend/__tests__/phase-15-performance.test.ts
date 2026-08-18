import { describe, expect, it } from "vitest";
import { mapPlatformChatReference } from "../chat-completion-api";
import { parsePlatformSseStream } from "../sse";

describe("Phase 15 bounded payload contracts", () => {
  it("consumes a large SSE sequence with reader backpressure and no frame loss", async () => {
    const encoder = new TextEncoder();
    const frameCount = 2_000;
    let index = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (index >= frameCount) {
          controller.close();
          return;
        }
        controller.enqueue(encoder.encode(`data: {"index":${index}}\n\n`));
        index += 1;
      },
    });
    let consumed = 0;
    for await (const event of parsePlatformSseStream(stream)) {
      expect(event.data).toContain(`"index":${consumed}`);
      consumed += 1;
    }
    expect(consumed).toBe(frameCount);
  });

  it("normalizes a large citation payload without dropping document aggregates", () => {
    const chunks = Array.from({ length: 1_000 }, (_, index) => ({
      id: `chunk-${index}`,
      document_id: `doc-${index % 25}`,
      document_name: `Document ${index % 25}`,
      content_with_weight: `Citation ${index}`,
      similarity: 0.9,
      positions: [[index + 1]],
    }));
    const reference = mapPlatformChatReference({
      chunks,
      doc_aggs: Array.from({ length: 25 }, (_, index) => ({
        doc_id: `doc-${index}`,
        doc_name: `Document ${index}`,
        count: 40,
      })),
    });
    expect(reference.chunks).toHaveLength(1_000);
    expect(reference.documentAggregations).toHaveLength(25);
  });
});
