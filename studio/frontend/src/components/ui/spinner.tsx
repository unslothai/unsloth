// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Loader2Icon } from "lucide-react";
import { cn } from "@/lib/utils";

/** App-wide spinner inheriting the current text color. `label` overrides the announcement
 * where "loading" is not what it means (a sidebar chat is generating). */
function Spinner({
  className,
  label = "Loading",
  "data-testid": dataTestId,
}: {
  className?: string;
  label?: string;
  "data-testid"?: string;
}) {
  return (
    <Loader2Icon
      role="status"
      aria-label={label}
      data-testid={dataTestId}
      className={cn("size-4 shrink-0 animate-spin", className)}
    />
  );
}

export { Spinner };
