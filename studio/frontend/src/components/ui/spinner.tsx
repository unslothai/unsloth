// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Loader2Icon } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * App-wide spinner: a clean circular arc with a rounded cap (lucide
 * Loader2 / LoaderCircle), animated, inheriting the current text color.
 */
function Spinner({ className }: { className?: string }) {
  return (
    <Loader2Icon
      role="status"
      aria-label="Loading"
      className={cn("size-4 shrink-0 animate-spin", className)}
    />
  );
}

export { Spinner };
