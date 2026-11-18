// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ErrorDetails } from "@/components/ui/error-details";
import { modelErrorParts } from "@/lib/model-error-message";

export function ModelLoadError({
  message,
  onDismiss,
}: { message: string; onDismiss: () => void }) {
  const { summary, details } = modelErrorParts(message);
  return (
    <div className="min-w-0 max-w-full basis-full rounded-xl border border-red-500/40 bg-red-500/5 p-3 text-ui-12 text-red-600 dark:text-red-400">
      <div className="flex items-start gap-3">
        <p
          role="alert"
          className="min-w-0 flex-1 whitespace-pre-wrap [overflow-wrap:anywhere]"
        >
          {summary}
        </p>
        <button
          type="button"
          aria-label="Dismiss model load error"
          onClick={onDismiss}
          className="shrink-0 rounded-full p-0.5 opacity-70 hover:opacity-100 focus-visible:outline-2"
        >
          <HugeiconsIcon
            icon={Cancel01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </div>
      {details && (
        <div className="mt-2">
          <ErrorDetails message={details} />
        </div>
      )}
    </div>
  );
}
