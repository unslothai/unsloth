// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// The sources panel is mounted without a key, so it is reused when the user
// switches project and an async mutation started in the old one keeps running
// against it. `useRagDocuments` is a hook and the panel is a .tsx, neither of
// which this runner can import, so these assert the guards are present in the
// source. They are cheap to delete by accident and expensive to notice: the
// symptom is one project's sources appearing under another project's controls,
// where removing them deletes from a project the user is not looking at.

const hookSource = readFileSync(
  new URL(
    "../src/features/rag/components/use-rag-documents.ts",
    import.meta.url,
  ),
  "utf8",
);
const panelSource = readFileSync(
  new URL(
    "../src/features/rag/components/project-sources-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("a refresh publishes only into the scope it was made for", () => {
  // A mutation that awaits reconciles with the `refresh` it captured, which can
  // start after the scope changed. Its ticket is then the newest, so the
  // sequence check alone lets it publish the wrong scope's rows.
  assert.match(
    hookSource,
    /if \(scopeKey !== liveScopeKeyRef\.current\) return true;\s*\n\s*setDocuments\(/,
    "refresh must compare its own scope against the mounted one before publishing",
  );
});

test("the scope check comes after the sequence check, not instead of it", () => {
  // They cover different failures: the sequence orders responses within one
  // scope, the scope check rejects a response belonging to a previous one.
  const seq = hookSource.indexOf("if (refreshSeq.current !== requestId)");
  const scope = hookSource.indexOf("if (scopeKey !== liveScopeKeyRef.current)");
  assert.ok(seq > 0 && scope > 0, "both guards must exist");
  assert.ok(scope > seq, "the scope check belongs after the sequence check");
});

test("a delete for another scope does not retire the mounted scope's load", () => {
  // Each remove() in a bulk batch bumps refreshSeq. Left ungated, a batch still
  // running for the project the user left would discard the new project's own
  // list request and leave it empty until something unrelated refreshed it.
  assert.match(
    hookSource,
    /const forCurrentScope = scopeKey === liveScopeKeyRef\.current;/,
    "remove must know whether it is acting on the mounted scope",
  );
  assert.match(
    hookSource,
    /if \(forCurrentScope\) \{\s*\n\s*refreshSeq\.current \+= 1;/,
    "only a delete for the mounted scope may invalidate refreshes",
  );
});

test("invalidating a refresh hands back the in-flight marker", () => {
  // Retiring a request means its own `finally` will not clear the marker, since
  // the sequence no longer matches. The four-second poll skips a tick while it is
  // set, and the KB dialog and thread bar call remove() with no replacement
  // refresh, so leaving it set stops those scopes polling for the session: an
  // indexing row stays "running" forever and sends gated on it never go out.
  assert.match(
    hookSource,
    /if \(forCurrentScope\) \{\s*\n\s*refreshSeq\.current \+= 1;\s*\n\s*refreshInFlight\.current = false;\s*\n\s*\}/,
    "remove must clear refreshInFlight alongside the sequence bump",
  );
});

test("a delete for another scope does not drop a row from the list on screen", () => {
  assert.match(
    hookSource,
    /if \(forCurrentScope\) \{\s*\n\s*setDocuments\(\(rows\) => \{\s*\n\s*restore = rows\.find/,
    "the optimistic drop must be confined to the mounted scope",
  );
});

test("remove is rebuilt when the scope key changes", () => {
  // forCurrentScope closes over scopeKey, so a stale callback would compare the
  // wrong value and defeat both guards above.
  const deps = hookSource.slice(hookSource.indexOf("const remove = useCallback"));
  assert.match(
    deps.slice(0, deps.indexOf("\n  );")),
    /\[scope, scopeKey\]/,
    "remove's dependency list must include scopeKey",
  );
});

test("the panel abandons its own bulk-remove continuation after a switch", () => {
  // The hook refuses to publish another scope's rows; what is left to the panel
  // is its own state -- a selection and a failure toast for a project the user
  // has left, and a refetch of a list nothing is showing.
  assert.match(
    panelSource,
    /if \(projectIdRef\.current !== batchProjectId\) return;/,
    "the bulk-remove continuation must stop when the project changed",
  );
});

test("the panel abandons its post-upload refresh after a switch", () => {
  assert.match(
    panelSource,
    /if \(projectIdRef\.current !== projectId\) return;/,
    "the post-upload refresh must stop when the project changed",
  );
});

test("both continuations report to the project they were started for", () => {
  // The announce is what tells whoever is showing that project to refetch, so it
  // must happen whatever the user navigated to -- before the guards return.
  const bulk = panelSource.slice(
    panelSource.indexOf("async function handleBulkRemove"),
  );
  const announce = bulk.indexOf("announceProjectSourcesUpdated(batchProjectId)");
  const guard = bulk.indexOf("projectIdRef.current !== batchProjectId");
  assert.ok(announce > 0 && guard > 0, "both must be present");
  assert.ok(
    announce < guard,
    "the announce must run before the continuation gives up",
  );
});
