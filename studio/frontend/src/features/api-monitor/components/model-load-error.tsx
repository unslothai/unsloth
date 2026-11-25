// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ErrorDetails } from "@/components/ui/error-details";
import { useT } from "@/i18n";
import { modelErrorParts } from "@/lib/model-error-message";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function ModelLoadError({
  message,
  onDismiss,
}: { message: string; onDismiss: () => void }) {
  const t = useT();
  const { summary, details } = modelErrorParts(message);
  return (
    <div className="min-w-0 max-w-full basis-full rounded-xl border border-red-500/40 bg-red-500/5 p-3 text-red-600 text-ui-12 dark:text-red-400">
      <div className="flex items-start gap-3">
        <p
          role="alert"
          className="min-w-0 flex-1 whitespace-pre-wrap [overflow-wrap:anywhere]"
        >
          {summary}
        </p>
        <button
          type="button"
          aria-label={t("common.close")}
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
