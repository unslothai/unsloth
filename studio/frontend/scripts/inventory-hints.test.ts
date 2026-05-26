// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  INVENTORY_HINT_TTL_MS,
  clonePendingInventoryHints,
  commitInventoryHintReconciliation,
  createPendingInventoryHints,
  pruneExpiredInventoryHints,
  reconcileInventoryHints,
  rememberInventoryHint,
} from "../src/features/inventory/inventory-hints.ts";
import {
  MAX_OBSERVED_INVENTORY_KEYS_PER_KIND,
  pendingWithInventoryHints,
  useInventoryHintStore,
} from "../src/features/inventory/inventory-hint-store.ts";

type Row = {
  repo_id: string;
  size_bytes: number;
  partial?: boolean;
};

test("pending inventory hints add optimistic rows", () => {
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "model",
    repoId: "Org/Model",
    bytes: 12,
  });

  const merged = reconcileInventoryHints<Row>({
    pending,
    kind: "model",
    rows: [],
    previouslyObserved: new Set(),
  });

  assert.deepEqual(merged.rows, [
    { repo_id: "Org/Model", size_bytes: 12, partial: false },
  ]);
  assert.equal(merged.pending.model.has("org/model"), true);
});

test("server-confirmed rows clear completed hints", () => {
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "gguf",
    repoId: "Org/Gguf",
    bytes: 20,
  });
  const baseline = clonePendingInventoryHints(pending);

  const merged = reconcileInventoryHints<Row>({
    pending,
    kind: "gguf",
    rows: [{ repo_id: "Org/Gguf", size_bytes: 25 }],
    previouslyObserved: new Set(),
  });
  const committed = commitInventoryHintReconciliation({
    current: pending,
    kind: "gguf",
    baseline,
    next: merged.pending,
    staleCompletedHints: merged.staleCompletedHints,
  });

  assert.deepEqual(merged.rows, [{ repo_id: "Org/Gguf", size_bytes: 25 }]);
  assert.equal(committed.pending.gguf.has("org/gguf"), false);
  assert.deepEqual(committed.completedHintsToSuppress, []);
});

test("empty-kind inventory reconciliation avoids cloning pending maps", () => {
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "dataset",
    repoId: "Org/Data",
    bytes: 30,
  });
  const rows = [{ repo_id: "Org/Model", size_bytes: 12 }];

  const merged = reconcileInventoryHints<Row>({
    pending,
    kind: "model",
    rows,
    previouslyObserved: new Set(),
  });

  assert.equal(merged.rows, rows);
  assert.equal(merged.pending, pending);
  assert.equal(merged.pending.dataset, pending.dataset);
  assert.deepEqual(merged.staleCompletedHints, []);
  assert.deepEqual([...merged.observedKeys], ["org/model"]);
});

test("stale completed hints are returned for suppression", () => {
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "dataset",
    repoId: "Org/Data",
    bytes: 30,
  });
  const baseline = clonePendingInventoryHints(pending);

  const merged = reconcileInventoryHints<Row>({
    pending,
    kind: "dataset",
    rows: [],
    previouslyObserved: new Set(["org/data"]),
  });
  const committed = commitInventoryHintReconciliation({
    current: pending,
    kind: "dataset",
    baseline,
    next: merged.pending,
    staleCompletedHints: merged.staleCompletedHints,
  });

  assert.equal(committed.pending.dataset.has("org/data"), false);
  assert.deepEqual(committed.completedHintsToSuppress, [
    { kind: "dataset", repoId: "Org/Data", bytes: 30 },
  ]);
});

test("newer concurrent hints survive stale reconciliation", () => {
  const baseline = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "model",
    repoId: "Org/Model",
    bytes: 10,
  });
  const current = rememberInventoryHint(baseline, {
    kind: "model",
    repoId: "Org/Model",
    bytes: 50,
  });

  const merged = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [],
    previouslyObserved: new Set(["org/model"]),
  });
  const committed = commitInventoryHintReconciliation({
    current,
    kind: "model",
    baseline,
    next: merged.pending,
    staleCompletedHints: merged.staleCompletedHints,
  });

  const hint = committed.pending.model.get("org/model");
  assert.ok(hint);
  assert.equal(hint.kind, "model");
  assert.equal(hint.repoId, "Org/Model");
  assert.equal(hint.bytes, 50);
  assert.equal(typeof hint.createdAt, "number");
  assert.deepEqual(committed.completedHintsToSuppress, []);
});

test("inventory hint store commits pending and observed keys immutably", () => {
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: { model: new Set(), gguf: new Set(), dataset: new Set() },
  });
  useInventoryHintStore.getState().rememberHint({
    kind: "model",
    repoId: "Org/Model",
    bytes: 10,
  });
  const baseline = clonePendingInventoryHints(
    useInventoryHintStore.getState().pending,
  );
  const observedBefore = useInventoryHintStore.getState().observedKeys;
  const merged = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [{ repo_id: "Org/Model", size_bytes: 12 }],
    previouslyObserved: observedBefore.model,
  });

  const suppress = useInventoryHintStore
    .getState()
    .commitReconciliations([], baseline, [
      { kind: "model", reconciliation: merged },
    ]);

  const state = useInventoryHintStore.getState();
  assert.deepEqual(suppress, []);
  assert.equal(state.pending.model.has("org/model"), false);
  assert.equal(state.observedKeys.model.has("org/model"), true);
  assert.notEqual(state.observedKeys, observedBefore);
});

test("inventory hint store leaves idempotent reconciliation commits untouched", () => {
  const pending = createPendingInventoryHints();
  useInventoryHintStore.setState({
    pending,
    observedKeys: { model: new Set(), gguf: new Set(), dataset: new Set() },
  });
  const reconciliation = {
    kind: "model" as const,
    reconciliation: {
      pending,
      staleCompletedHints: [],
      observedKeys: new Set(["org/model"]),
    },
  };

  useInventoryHintStore
    .getState()
    .commitReconciliations([], pending, [reconciliation]);
  const settledState = useInventoryHintStore.getState();

  useInventoryHintStore
    .getState()
    .commitReconciliations([], pending, [reconciliation]);

  assert.equal(useInventoryHintStore.getState(), settledState);
});

test("completed download hints render optimistically until inventory confirms them", () => {
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: { model: new Set(), gguf: new Set(), dataset: new Set() },
  });
  const completedHints = [
    { kind: "model" as const, repoId: "Org/Model", bytes: 42 },
  ];

  const baseline = pendingWithInventoryHints(
    useInventoryHintStore.getState().pending,
    completedHints,
  );
  const optimistic = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [],
    previouslyObserved: new Set(),
  });

  assert.deepEqual(optimistic.rows, [
    { repo_id: "Org/Model", size_bytes: 42, partial: false },
  ]);

  const confirmed = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [{ repo_id: "Org/Model", size_bytes: 42 }],
    previouslyObserved: new Set(),
  });
  const suppress = useInventoryHintStore
    .getState()
    .commitReconciliations(completedHints, baseline, [
      { kind: "model", reconciliation: confirmed },
    ]);

  assert.deepEqual(suppress, []);
  assert.equal(
    useInventoryHintStore.getState().pending.model.has("org/model"),
    false,
  );
});

test("pending inventory hints expire without refreshing their creation time", () => {
  const now = 1_000_000;
  let pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "model",
    repoId: "Org/Old",
    bytes: 10,
    createdAt: now - INVENTORY_HINT_TTL_MS - 1,
  });
  pending = rememberInventoryHint(pending, {
    kind: "dataset",
    repoId: "Org/Fresh",
    bytes: 20,
    createdAt: now - INVENTORY_HINT_TTL_MS + 1,
  });
  pending = rememberInventoryHint(pending, {
    kind: "dataset",
    repoId: "Org/Fresh",
    bytes: 30,
    createdAt: now,
  });

  const pruned = pruneExpiredInventoryHints(pending, now);
  assert.equal(pruned.model.has("org/old"), false);
  assert.equal(pruned.dataset.get("org/fresh")?.bytes, 30);
  assert.equal(
    pruned.dataset.get("org/fresh")?.createdAt,
    now - INVENTORY_HINT_TTL_MS + 1,
  );
});

test("inventory hint store prunes expired pending hints", () => {
  const now = Date.now();
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "model",
    repoId: "Org/Old",
    bytes: 10,
    createdAt: now - INVENTORY_HINT_TTL_MS - 1,
  });
  useInventoryHintStore.setState({
    pending,
    observedKeys: { model: new Set(), gguf: new Set(), dataset: new Set() },
  });

  useInventoryHintStore.getState().pruneExpiredHints();

  assert.equal(
    useInventoryHintStore.getState().pending.model.has("org/old"),
    false,
  );
});

test("stale completed download hints are returned for suppression", () => {
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: {
      model: new Set(["org/model"]),
      gguf: new Set(),
      dataset: new Set(),
    },
  });
  const completedHints = [
    { kind: "model" as const, repoId: "Org/Model", bytes: 42 },
  ];
  const baseline = pendingWithInventoryHints(
    useInventoryHintStore.getState().pending,
    completedHints,
  );
  const reconciled = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [],
    previouslyObserved: useInventoryHintStore.getState().observedKeys.model,
  });

  const suppress = useInventoryHintStore
    .getState()
    .commitReconciliations(completedHints, baseline, [
      { kind: "model", reconciliation: reconciled },
    ]);

  assert.deepEqual(suppress, completedHints);
  assert.equal(
    useInventoryHintStore.getState().pending.model.has("org/model"),
    false,
  );
});

test("inventory hint store commits multiple kinds in one reconciliation", () => {
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: {
      model: new Set(),
      gguf: new Set(),
      dataset: new Set(["org/data"]),
    },
  });
  useInventoryHintStore.getState().rememberHint({
    kind: "model",
    repoId: "Org/Model",
    bytes: 10,
  });
  useInventoryHintStore.getState().rememberHint({
    kind: "dataset",
    repoId: "Org/Data",
    bytes: 30,
  });
  const baseline = clonePendingInventoryHints(
    useInventoryHintStore.getState().pending,
  );
  const modelReconciliation = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "model",
    rows: [{ repo_id: "Org/Model", size_bytes: 12 }],
    previouslyObserved: useInventoryHintStore.getState().observedKeys.model,
  });
  const datasetReconciliation = reconcileInventoryHints<Row>({
    pending: baseline,
    kind: "dataset",
    rows: [],
    previouslyObserved: useInventoryHintStore.getState().observedKeys.dataset,
  });

  const suppress = useInventoryHintStore
    .getState()
    .commitReconciliations([], baseline, [
      { kind: "model", reconciliation: modelReconciliation },
      { kind: "dataset", reconciliation: datasetReconciliation },
    ]);

  const state = useInventoryHintStore.getState();
  assert.equal(state.pending.model.has("org/model"), false);
  assert.equal(state.pending.dataset.has("org/data"), false);
  assert.deepEqual(suppress, [
    { kind: "dataset", repoId: "Org/Data", bytes: 30 },
  ]);
});

test("observed inventory keys evict oldest unprotected keys per kind", () => {
  const protectedKey = "org/protected";
  const oldKeys = Array.from(
    { length: MAX_OBSERVED_INVENTORY_KEYS_PER_KIND - 1 },
    (_, idx) => `org/old-${idx}`,
  );
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "model",
    repoId: "Org/Protected",
    bytes: 1,
  });
  useInventoryHintStore.setState({
    pending,
    observedKeys: {
      model: new Set([protectedKey, ...oldKeys]),
      gguf: new Set(),
      dataset: new Set(),
    },
  });
  const baseline = clonePendingInventoryHints(pending);

  useInventoryHintStore.getState().commitReconciliations([], baseline, [
    {
      kind: "model",
      reconciliation: {
        pending: baseline,
        staleCompletedHints: [],
        observedKeys: new Set(["org/new"]),
      },
    },
  ]);

  const observed = useInventoryHintStore.getState().observedKeys.model;
  assert.equal(observed.size, MAX_OBSERVED_INVENTORY_KEYS_PER_KIND);
  assert.equal(observed.has(protectedKey), true);
  assert.equal(observed.has("org/new"), true);
  assert.equal(observed.has("org/old-0"), false);
});
