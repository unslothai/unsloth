// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for src/features/export/api/export-api.ts.
//
// The real module reaches authFetch -> features/auth/index.ts -> login-page.tsx, and
// node --experimental-strip-types cannot parse JSX, so importing the export store for
// a unit test would pull in the whole React tree. Cutting the chain here keeps the
// state machine testable without touching the shared bundler-resolver. Calls are
// recorded on `calls` and answered from `responses`, so a test can drive runExport
// end to end without a server.

export const calls = [];
export const responses = new Map();

function record(name) {
  return async (...args) => {
    calls.push({ name, args });
    const answer = responses.get(name);
    if (typeof answer === "function") return answer(...args);
    if (answer instanceof Error) throw answer;
    return answer ?? { success: true, message: "ok", details: null };
  };
}

export function resetStub() {
  calls.length = 0;
  responses.clear();
}

export const loadCheckpoint = record("loadCheckpoint");
export const exportGGUF = record("exportGGUF");
export const exportMerged = record("exportMerged");
export const exportLoRA = record("exportLoRA");
export const cleanupExport = record("cleanupExport");
export const cancelExport = record("cancelExport");
export const getExportStatus = record("getExportStatus");

export function isRecoverableTransportError() {
  return false;
}
