// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import type { HfModelResult } from "@/features/hub/hooks/use-hub-model-search";
// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import { formatBytes } from "@/features/hub/lib/format";
// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import { publicModelId } from "@/features/hub/lib/model-identity";
// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import { detectLicense } from "@/features/hub/lib/model-capabilities";

const HUB_REPO_RE = /^[\w.-]+\/[\w.-]+$/;
const PARAMS = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

/** The Hub repo behind a model id or HF cache snapshot path; null for other local paths. */
export function hubRepoIdFor(modelId: string): string | null {
  const repoId = publicModelId(modelId);
  return HUB_REPO_RE.test(repoId) ? repoId : null;
}

function formatDate(value: string | undefined): string | null {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;
}

/** Label/value rows for the Model info group; unknown fields are left out, except the license. */
export function modelInfoFacts(model: HfModelResult): [string, string][] {
  const access = [
    model.gated ? "Gated" : null,
    model.private ? "Private" : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const rows: [string, string | null][] = [
    ["License", detectLicense(model.tags) ?? "Not specified"],
    ["Parameters", model.totalParams ? PARAMS.format(model.totalParams) : null],
    // Summed from the stored dtypes, so a quantized checkpoint reports its own size.
    [
      "Size (estimated)",
      model.estimatedSizeBytes ? formatBytes(model.estimatedSizeBytes) : null,
    ],
    ["Library", model.libraryName ?? null],
    ["Task", model.pipelineTag ? model.pipelineTag.replace(/-/g, " ") : null],
    ["Created", formatDate(model.createdAt)],
    ["Updated", formatDate(model.updatedAt)],
    ["Access", access || null],
  ];
  return rows.filter((row): row is [string, string] => row[1] !== null);
}
