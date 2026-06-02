import type { PreviewTarget } from "@/features/rag/api/rag-api";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockAuthFetch,
  mockGetAuthToken,
  mockEventSourceInstances,
  eventSourcePolyfillExport,
  authorizationHeader,
} = vi.hoisted(() => ({
  mockAuthFetch: vi.fn(),
  mockGetAuthToken: vi.fn(),
  mockEventSourceInstances: [] as MockEventSource[],
  eventSourcePolyfillExport: "EventSourcePolyfill",
  authorizationHeader: "Authorization",
}));

vi.mock("@/features/auth", () => ({
  authFetch: mockAuthFetch,
  getAuthToken: mockGetAuthToken,
}));

interface MockEventSource {
  url: string;
  options: unknown;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: (() => void) | null;
  close: ReturnType<typeof vi.fn>;
}

vi.mock("event-source-polyfill", () => ({
  [eventSourcePolyfillExport]: class {
    url: string;
    options: unknown;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: (() => void) | null = null;
    close = vi.fn();

    constructor(url: string, options?: unknown) {
      this.url = url;
      this.options = options;
      mockEventSourceInstances.push(this);
    }
  },
}));

import {
  fetchPreviewFileUrl,
  fetchPreviewTarget,
  subscribeToJobEvents,
} from "@/features/rag/api/rag-api";

function target(): PreviewTarget {
  return {
    documentId: "doc-abc",
    filename: "report.pdf",
    contentType: "application/pdf",
    mediaKind: "pdf",
    byteSize: 100,
    status: "completed",
    kbId: "kb-1",
    threadId: null,
    chunkId: "chunk-xyz",
    chunkIndex: 0,
    targetPage: 1,
    snippet: "excerpt",
    kind: "text",
    sourcePageIndex: 0,
    pageCharStart: 0,
    pageCharEnd: 7,
    lineStart: 1,
    lineEnd: 1,
    pdfRegions: [],
  };
}

beforeEach(() => {
  mockAuthFetch.mockReset();
  mockGetAuthToken.mockReset();
  mockEventSourceInstances.length = 0;
});

describe("RAG API preview target", () => {
  it("URL-encodes documentId and chunk_id", async () => {
    mockAuthFetch.mockResolvedValue(
      new Response(JSON.stringify(target()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await fetchPreviewTarget("doc id/with?slash", "chunk id/with?amp&eq=1");

    expect(mockAuthFetch).toHaveBeenCalledWith(
      "/api/rag/documents/doc%20id%2Fwith%3Fslash/preview-target?chunk_id=chunk%20id%2Fwith%3Famp%26eq%3D1",
    );
  });

  it("fetches signed preview URL without adding a bearer token query", async () => {
    mockAuthFetch.mockResolvedValue(
      new Response(
        JSON.stringify({
          url: "/api/rag/documents/doc-abc/file-signed?token=signed-preview",
          expiresAt: 1_700_000_000,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    const result = await fetchPreviewFileUrl("doc id/with?slash");

    expect(mockAuthFetch).toHaveBeenCalledWith(
      "/api/rag/documents/doc%20id%2Fwith%3Fslash/file-url",
      undefined,
    );
    expect(result.url).toContain("token=signed-preview");
    expect(result.url).not.toContain("Bearer");
    expect(result.url).not.toContain("Authorization");
  });

});

describe("RAG API job events", () => {
  it("opens SSE with Authorization header instead of token query params", () => {
    mockGetAuthToken.mockReturnValue("mock-token-123");

    const unsubscribe = subscribeToJobEvents("job id/with?slash", {});

    expect(mockEventSourceInstances).toHaveLength(1);
    const source = mockEventSourceInstances[0];
    expect(source.url).toContain(
      "/api/rag/jobs/job%20id%2Fwith%3Fslash/events",
    );
    expect(source.url).not.toContain("token=");
    expect(source.options).toEqual({
      headers: {
        [authorizationHeader]: "Bearer mock-token-123",
      },
    });

    unsubscribe();
    expect(source.close).toHaveBeenCalled();
  });

  it("omits EventSource options when there is no bearer token", () => {
    mockGetAuthToken.mockReturnValue(null);

    const unsubscribe = subscribeToJobEvents("job-abc", {});

    expect(mockEventSourceInstances).toHaveLength(1);
    const source = mockEventSourceInstances[0];
    expect(source.url).toContain("/api/rag/jobs/job-abc/events");
    expect(source.url).not.toContain("token=");
    expect(source.options).toBeUndefined();

    unsubscribe();
  });
});
