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
    <label
      className="flex items-center gap-1.5 text-xs font-semibold uppercase text-muted-foreground"
      htmlFor={htmlFor}
    >
      <span>{label}</span>
      {hint && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="inline-flex size-3.5 items-center justify-center rounded-full text-muted-foreground/80 hover:text-foreground"
              aria-label={`More info: ${label}`}
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent>{hint}</TooltipContent>
        </Tooltip>
      )}
    </label>
  );
}

