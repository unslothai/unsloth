// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  authFetch: vi.fn(),
}));

vi.mock("@/features/auth", () => ({
  authFetch: mocks.authFetch,
}));

import {
  PROJECT_WORK_CHANGED_EVENT,
  announceProjectSourcesUpdated,
  isRagClientError,
  listProjectDocuments,
  noteProjectWork,
  projectWorkCount,
  subscribeProjectSourcesUpdated,
  watchProjectFolderJob,
} from "./rag-api";

describe("RAG project coordination", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("classifies answered client failures as terminal but keeps rate limits retryable", async () => {
    mocks.authFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Project not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const error = await listProjectDocuments("missing-project").catch(
      (caught: unknown) => caught,
    );

    expect(isRagClientError(error)).toBe(true);
    expect(
      isRagClientError(
        Object.assign(new Error("rate limited"), { status: 429 }),
      ),
    ).toBe(false);
    expect(isRagClientError(new TypeError("network failure"))).toBe(false);
  });

  it("tracks project work and publishes count changes", () => {
    const changes: string[] = [];
    const listener = (event: Event) => {
      const detail = (event as CustomEvent<{ projectId: string }>).detail;
      changes.push(detail.projectId);
    };
    window.addEventListener(PROJECT_WORK_CHANGED_EVENT, listener);

    try {
      noteProjectWork("project-work", 1);
      noteProjectWork("project-work", 1);
      expect(projectWorkCount("project-work")).toBe(2);

      noteProjectWork("project-work", -1);
      noteProjectWork("project-work", -1);
      expect(projectWorkCount("project-work")).toBe(0);
      expect(changes).toEqual([
        "project-work",
        "project-work",
        "project-work",
        "project-work",
      ]);
    } finally {
      window.removeEventListener(PROJECT_WORK_CHANGED_EVENT, listener);
    }
  });

  it("notifies only subscribers for changed project sources", () => {
    const first = vi.fn();
    const second = vi.fn();
    const unsubscribeFirst = subscribeProjectSourcesUpdated("project-1", first);
    const unsubscribeSecond = subscribeProjectSourcesUpdated(
      "project-2",
      second,
    );

    try {
      announceProjectSourcesUpdated("project-1");
      expect(first).toHaveBeenCalledTimes(1);
      expect(second).not.toHaveBeenCalled();
    } finally {
      unsubscribeFirst();
      unsubscribeSecond();
    }
  });

  it("stops tracking a cancelled folder job without another poll", async () => {
    mocks.authFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "job-cancelled",
          status: "cancelled",
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    watchProjectFolderJob("project-cancelled", "job-cancelled");
    expect(projectWorkCount("project-cancelled")).toBe(1);

    await vi.waitFor(() => {
      expect(projectWorkCount("project-cancelled")).toBe(0);
    });
    expect(mocks.authFetch).toHaveBeenCalledTimes(1);
  });
});
