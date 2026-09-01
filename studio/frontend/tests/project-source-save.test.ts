// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";
import { recordedToasts } from "./helpers/store-stubs/toast.ts";

const { storage } = installLocalStorageFake();
// A real EventTarget in place of the inert window the fake installs: the sources
// panel only learns about a save through a window event.
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

/** Reject polls from watchers left by earlier tests. */
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
  // The sources panel invalidates *before* its own delete, having already
  // dropped the row; a refetch there would put the row straight back.
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
  // Announced after the upload, so the panel refetches a list that has it.
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
  // The probe is still invalidated, so the next chat re-reads the truth.
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
  // Waiting for the post-toast announce avoids a race.
  const announcedTwice = await waitFor(() =>
    seen.filter((id) => id === "p4").length >= 2 || undefined,
  );
  assert.ok(
    announcedTwice,
    `the failed ingest never re-announced p4, so a chip left "pending" never resolves; saw ${JSON.stringify(seen)}`,
  );
  // The panel hides failed documents, so the success toast would otherwise be
  // the only thing the user ever sees about a source that never arrives.
  const failure = recordedToasts.find(
    (t) => t.message === "Couldn't index Unparsable.md",
  );
  assert.equal(failure?.kind, "error");
  assert.equal(failure?.description, "Could not parse the document");
  assert.equal(seen.filter((id) => id === "p4").length, 2);
});

async function waitFor<T>(read: () => T | undefined): Promise<T | undefined> {
  for (let attempt = 0; attempt < 300; attempt++) {
    const value = read();
    if (value !== undefined) return value;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return undefined;
}

// The panel is a .tsx component node cannot load, so the subscription it mounts
// lives in rag-api and is exercised here directly. These assert the refresh
// itself runs, not merely that an event was dispatched.

test("a mounted sources list refetches when a chat is saved into its project", async () => {
  // Model the panel: subscribe, and let the callback re-run the same lister
  // useRagDocuments.refresh calls. Before the fix, nothing here ever ran again,
  // because the list only polls while a row it already knows is indexing, and an
  // empty panel knows none.
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
    // The upload is what puts the row on the server.
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
  const src = await readFile(
    new URL(
      "../src/features/rag/components/project-sources-panel.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    src,
    /subscribeProjectSourcesUpdated\(projectId, \(\) => \{\n\s*void refresh\(\{ quiet: true \}\);\n\s*\}\),/,
    "the mounted list no longer refreshes when a source is saved elsewhere",
  );
  assert.ok(
    !src.includes("PROJECT_SOURCES_UPDATED_EVENT"),
    "the panel listens for the raw event again, bypassing the tested subscription",
  );
  // handleRemove drops the row optimistically and *then* invalidates, so an
  // invalidate that also refetched would re-list the row before the DELETE went
  // out, and the panel has no request sequencing to drop the stale answer.
  assert.match(
    src,
    /invalidateProjectSources\(projectId\);\n\s*await remove\(documentId\);/,
    "the delete path no longer invalidates before its own mutation",
  );
});
