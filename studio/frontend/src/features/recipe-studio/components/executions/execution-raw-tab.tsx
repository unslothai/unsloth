// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { ReactElement } from "react";

type ExecutionRawTabProps = {
  rawExecution: Record<string, unknown> | null;
};

export function ExecutionRawTab({
  rawExecution,
}: ExecutionRawTabProps): ReactElement {
  return (
    <div className="mt-3 rounded-xl border p-3">
      <p className="mb-2 text-sm font-semibold">Raw execution</p>
      <pre className="max-h-96 overflow-auto rounded-md bg-muted/40 p-3 text-xs">
        {JSON.stringify(rawExecution, null, 2)}
      </pre>
    </div>
  );
}
