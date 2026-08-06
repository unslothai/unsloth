// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createScopedRefreshGate,
  runScopedRefresh,
  setScopedRefreshScope,
} from "../src/features/rag/components/scoped-refresh.ts";
import {
  isLinkedFolderManaged,
  linkedFolderSourcesChanged,
} from "../src/features/rag/types/rag.ts";

const ragApi = readFileSync(
  new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
  "utf8",
);
const nativeApi = readFileSync(
  new URL("../src/features/native-intents/api.ts", import.meta.url),
  "utf8",
);
const statusChip = readFileSync(
  new URL(
    "../src/features/rag/components/document-status-chip.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("linked folders have global/scoped listing and scoped creation routes", () => {
  assert.match(
    ragApi,
    /`\?scope_type=\$\{encodeURIComponent\(scope\.type\)\}&scope_id=/,
  );
  assert.match(
    ragApi,
    /scope\.type === "knowledge_base" \? "knowledge-bases" : "projects"/,
  );
  assert.match(ragApi, /body: \{ nativePathLease, displayName \}/);
});

test("unlink, sync, rebuild, and aggregate job routes stay explicit", () => {
  assert.match(ragApi, /\?remove_index=\$\{removeIndex\}/);
  assert.match(ragApi, /action: "sync" \| "rebuild"/);
  assert.match(
    ragApi,
    /\/linked-folder-jobs\/\$\{encodeURIComponent\(jobId\)\}/,
  );
  assert.match(
    ragApi,
    /\/linked-folder-jobs\/\$\{encodeURIComponent\(jobId\)\}\/events/,
  );
});

test("the desktop folder picker returns an opaque token, never a path", () => {
  assert.match(nativeApi, /"pick_native_document_folder"/);
  assert.match(nativeApi, /token: string/);
  assert.match(nativeApi, /displayName: string/);
  assert.doesNotMatch(
    nativeApi,
    /interface NativeDocumentFolderSelection \{[^}]*path:/s,
  );
});

test("document status chips render their progress value", () => {
  assert.match(statusChip, /progress != null/);
  assert.match(statusChip, /progress \* 100/);
});

test("documents managed by linked folders cannot be treated as individual uploads", () => {
  const document = {
    id: "doc",
    filename: "notes.md",
    status: "completed",
    managed: false,
  } as const;
  assert.equal(isLinkedFolderManaged(document), false);
  assert.equal(
    isLinkedFolderManaged({ ...document, linkedFolderId: "folder" }),
    true,
  );
  assert.equal(isLinkedFolderManaged({ ...document, managed: true }), true);
});

test("a deferred list response cannot repopulate a different scope", async () => {
  let resolveKb!: (rows: string[]) => void;
  let resolveProject!: (rows: string[]) => void;
  const kbResponse = new Promise<string[]>((resolve) => {
    resolveKb = resolve;
  });
  const projectResponse = new Promise<string[]>((resolve) => {
    resolveProject = resolve;
  });
  const gate = createScopedRefreshGate("knowledge_base:kb-1");
  let visibleRows: string[] = [];

  const list = (scopeKey: string, response: Promise<string[]>) =>
    runScopedRefresh(gate, scopeKey, async (isCurrent) => {
      const rows = await response;
      if (isCurrent()) visibleRows = rows;
    });

  const oldRequest = list("knowledge_base:kb-1", kbResponse);
  setScopedRefreshScope(gate, "project:project-1");
  visibleRows = [];
  const currentRequest = list("project:project-1", projectResponse);

  resolveKb(["kb folder"]);
  await oldRequest;
  assert.deepEqual(visibleRows, []);

  resolveProject(["project folder"]);
  await currentRequest;
  assert.deepEqual(visibleRows, ["project folder"]);
});

test("refreshes deduplicate within the current scope", async () => {
  let resolve!: () => void;
  const response = new Promise<void>((done) => {
    resolve = done;
  });
  const gate = createScopedRefreshGate("project:project-1");
  let requests = 0;
  const refresh = () =>
    runScopedRefresh(gate, "project:project-1", async () => {
      requests += 1;
      await response;
    });

  const first = refresh();
  const second = refresh();
  assert.equal(requests, 1);
  assert.equal(first, second);
  resolve();
  await first;
});

test("folder polling detects completed backend work without a visible active job", () => {
  const folder = {
    id: "folder",
    displayName: "Docs",
    scopeType: "project",
    scopeId: "project",
    status: "idle",
    documentCount: 2,
    lastSyncedAt: "2026-08-06T10:00:00Z",
  } as const;
  assert.equal(linkedFolderSourcesChanged(null, [folder]), false);
  assert.equal(linkedFolderSourcesChanged([folder], [folder]), false);
  assert.equal(
    linkedFolderSourcesChanged(
      [folder],
      [
        {
          ...folder,
          documentCount: 3,
          lastSyncedAt: "2026-08-06T10:01:00Z",
        },
      ],
    ),
    true,
  );
});
