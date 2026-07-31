// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MermaidErrorComponentProps } from "streamdown";

function hasSlashComment(chart: string): boolean {
  return /(^|[^:])\/\/.*/m.test(chart);
}

export function MermaidError({
  error,
  chart,
  retry,
}: MermaidErrorComponentProps) {
  return (
    <div className="my-4 rounded-lg border border-red-300 bg-red-50 p-3 text-red-800">
      <p className="text-sm font-semibold">Mermaid render failed</p>
      <p className="mt-1 break-words font-mono text-xs">{error}</p>
      {hasSlashComment(chart) ? (
        <p className="mt-1 text-xs">Hint: Mermaid comments use `%%`, not `//`.</p>
      ) : null}
      <button
        type="button"
        onClick={retry}
        className="mt-2 rounded border border-red-300 px-2 py-1 text-xs hover:bg-red-100"
      >
        Retry
      </button>
    </div>
  );
}
