// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  readSrcAsync,
  registerStoreStubResolver,
} from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";
import { recordedToasts } from "./helpers/store-stubs/toast.ts";

const { storage } = installLocalStorageFake();
// The sources panel only learns about a save through a window event.
const events = new EventTarget();
Object.assign(globalThis, {
  window: Object.assign(events, {
    localStorage: storage,
    location: { protocol: "http:" },
  }),
});
registerStoreStubResolver();

const {
  PROJECT_SOURCES_UPDATED_EVENT,
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  subscribeProjectSourcesUpdated,
} = await import("../src/features/rag/api/rag-api.ts");
const { saveMarkdownAsProjectSource } = await import(
  "../src/features/rag/api/save-markdown-source.ts"
);

function collectUpdates(): string[] {
  const seen: string[] = [];
  events.addEventListener(PROJECT_SOURCES_UPDATED_EVENT, (event) => {
    seen.push(
      String((event as CustomEvent<{ projectId?: string }>).detail?.projectId),
    );
  });
  return seen;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Ingest watchers poll up to 300s across tests, so only answer this test's job. */
function jobFor(jobId: string, input: string, body: unknown): Response {
  return input.includes(`/jobs/${jobId}`)
    ? json(body)
    : json({ detail: `no such job: ${input}` }, 404);
}

test.beforeEach(() => {
  recordedToasts.length = 0;
  setAuthFetchHandler(null);
});

test("invalidating the probe does not refetch anyone's document list", () => {
  // The panel invalidates after dropping the row; a refetch there would restore it.
  const seen = collectUpdates();
  invalidateProjectSources("p1");
  assert.deepEqual(seen, []);
  announceProjectSourcesUpdated("p1");
  assert.deepEqual(seen, ["p1"]);
});

test("uploads the chat under its sanitised name and reports it once", async () => {
  const seen = collectUpdates();
  const uploaded: File[] = [];
  setAuthFetchHandler((input, init) => {
    assert.equal(input, "/api/rag/projects/p%20one/documents");
    uploaded.push((init?.body as FormData).get("file") as File);
    return json({ documentId: "d1", jobId: "j1", filename: "Chat.md" });
  });
  const ok = await saveMarkdownAsProjectSource("p one", "# Chat\n", "Chat:1");
  assert.equal(ok, true);
  assert.equal(uploaded.length, 1);
  assert.equal(uploaded[0].name, "Chat_1.md");
  assert.equal(uploaded[0].type, "text/markdown");
  assert.equal(await uploaded[0].text(), "# Chat\n");
  assert.deepEqual(
    recordedToasts.map((t) => [t.kind, t.message]),
    [["success", "Saved to project sources."]],
  );
  assert.deepEqual(seen, ["p one"]);
});

test("a quiet save stays silent so a pair can report the count itself", async () => {
  setAuthFetchHandler((input) =>
    input.includes("/jobs/")
      ? jobFor("j2", input, { id: "j2", documentId: "d2", status: "completed" })
      : json({ documentId: "d2", jobId: "j2", filename: "Chat.md" }),
  );
  assert.equal(
    await saveMarkdownAsProjectSource("p2", "# Chat\n", "Chat", {
      quiet: true,
    }),
    true,
  );
  assert.deepEqual(recordedToasts, []);
});

test("a rejected upload resolves false and says why", async () => {
  const seen = collectUpdates();
  setAuthFetchHandler(() => json({ detail: "Project not found" }, 404));
  assert.equal(
    await saveMarkdownAsProjectSource("gone", "# Chat\n", "Chat"),
    false,
  );
  assert.deepEqual(
    recordedToasts.map((t) => [t.kind, t.message, t.description]),
    [["error", "Failed to save to project sources.", "Project not found"]],
  );
  assert.deepEqual(seen, ["gone"]);
});

test("a quiet save still reports its own failure", async () => {
  setAuthFetchHandler(() => json({ detail: "RAG is unavailable" }, 503));
  assert.equal(
    await saveMarkdownAsProjectSource("p3", "# Chat\n", "Chat", {
      quiet: true,
    }),
    false,
  );
  assert.equal(recordedToasts.length, 1);
  assert.equal(recordedToasts[0].kind, "error");
});

test("an ingest that fails after the upload is not left silent", async () => {
  const seen = collectUpdates();
  // A unique filename so toasts are attributable to this save.
  setAuthFetchHandler((input) => {
    if (input.includes("/jobs/")) {
      return jobFor("j4", input, {
        id: "j4",
        documentId: "d4",
        status: "failed",
        error: "Could not parse the document",
      });
    }
    return json({ documentId: "d4", jobId: "j4", filename: "Unparsable.md" });
  });
  await saveMarkdownAsProjectSource("p4", "# Chat\n", "Unparsable");
  // Wait on the announce, not the toast: the watcher toasts then announces with no await between.
  const announcedTwice = await waitFor(() =>
    seen.filter((id) => id === "p4").length >= 2 || undefined,
  );
  assert.ok(
    announcedTwice,
    `the failed ingest never re-announced p4, so a chip left "pending" never resolves; saw ${JSON.stringify(seen)}`,
  );
  // The panel hides failed documents, so the toast is the only failure signal.
  const failure = recordedToasts.find(
    (t) => t.message === "Couldn't index Unparsable.md",
  );
  assert.equal(failure?.kind, "error");
  assert.equal(failure?.description, "Could not parse the document");
  assert.equal(seen.filter((id) => id === "p4").length, 2);
});

/** Only assert things the code does before what is polled, or the wait races it. */
async function waitFor<T>(read: () => T | undefined): Promise<T | undefined> {
  for (let attempt = 0; attempt < 300; attempt++) {
    const value = read();
    if (value !== undefined) return value;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return undefined;
}

// The panel is .tsx, so its rag-api subscription is exercised directly.

test("a mounted sources list refetches when a chat is saved into its project", async () => {
  // The list only polls while a known row indexes, so an empty panel needs the event.
  const listed: string[][] = [];
  let rows: string[] = [];
  const unsubscribe = subscribeProjectSourcesUpdated("p5", () => {
    listed.push([...rows]);
  });
  setAuthFetchHandler((input) => {
    if (input.includes("/jobs/")) {
      return jobFor("j5", input, {
        id: "j5",
        documentId: "d5",
        status: "completed",
      });
    }
    rows = ["Chat.md"];
    return json({ documentId: "d5", jobId: "j5", filename: "Chat.md" });
  });
  await saveMarkdownAsProjectSource("p5", "# Chat\n", "Chat");
  assert.deepEqual(
    listed,
    [["Chat.md"]],
    "the list was never refetched, so the saved source stays absent until remount",
  );
  unsubscribe();
});

test("another project's save leaves this list alone", () => {
  let refreshed = 0;
  const unsubscribe = subscribeProjectSourcesUpdated("mine", () => {
    refreshed += 1;
  });
  announceProjectSourcesUpdated("theirs");
  assert.equal(refreshed, 0, "every open panel refetches on any project's save");
  announceProjectSourcesUpdated("mine");
  assert.equal(refreshed, 1);
  unsubscribe();
});

test("unsubscribing stops the refetch, so an unmounted panel cannot set state", () => {
  let refreshed = 0;
  const unsubscribe = subscribeProjectSourcesUpdated("p6", () => {
    refreshed += 1;
  });
  unsubscribe();
  announceProjectSourcesUpdated("p6");
  assert.equal(refreshed, 0);
});

test("the panel subscribes, and does not resurrect a row it just deleted", async () => {
  const src = await readSrcAsync("features/rag/components/project-sources-panel.tsx");
  assert.match(
    src,
    /subscribeProjectSourcesUpdated\(projectId, \(\) => \{\n\s*void refresh\(\{ quiet: true \}\);\n\s*\}\),/,
    "the mounted list no longer refreshes when a source is saved elsewhere",
  );
  assert.ok(
    !src.includes("PROJECT_SOURCES_UPDATED_EVENT"),
    "the panel listens for the raw event again, bypassing the tested subscription",
  );
  // The row is dropped optimistically before DELETE, and there is no request sequencing.
  assert.match(
    src,
    /invalidateProjectSources\(projectId\);\n\s*await remove\(documentId\);/,
    "the delete path no longer invalidates before its own mutation",
  );
});
