// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

type FieldLabelProps = {
  label: string;
  htmlFor?: string;
  hint?: string;
};

export function FieldLabel({
  label,
  htmlFor,
  hint,
}: FieldLabelProps): ReactElement {
  return (
    <div className="flex min-w-0 items-center gap-1 text-xs font-semibold uppercase text-muted-foreground">
      {htmlFor ? (
        <label className="min-w-0 cursor-pointer" htmlFor={htmlFor}>
          <span className="break-words">{label}</span>
        </label>
      ) : (
        <span className="min-w-0 break-words">{label}</span>
      )}
      {hint && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="inline-flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground/80 transition hover:text-foreground"
              aria-label={`More info: ${label}`}
              title={`More info about ${label}`}
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-4" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-64 break-words text-xs leading-relaxed">
            {hint}
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}
