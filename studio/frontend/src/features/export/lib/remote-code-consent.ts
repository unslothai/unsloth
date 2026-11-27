// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function exportRemoteCodeConsent({
  sourceMode,
  modelSource,
  source,
  checkpointPath,
}: {
  sourceMode: "checkpoint" | "model";
  modelSource: "hf" | "local";
  source: string;
  checkpointPath: string | null;
}): {
  scanTarget: string;
  requiresTrustRemoteCode: boolean;
  trustByDefault: boolean;
} {
  if (sourceMode === "checkpoint") {
    // Scan the checkpoint folder (as the export gate does); base-only scans miss copied code.
    return {
      scanTarget: checkpointPath ?? source,
      requiresTrustRemoteCode: false,
      trustByDefault: false,
    };
  }
  return {
    scanTarget: source,
    // An HF source can need trust_remote_code via its YAML default with no auto_map to review.
    requiresTrustRemoteCode: modelSource === "hf",
    trustByDefault: modelSource !== "hf",
  };
}
