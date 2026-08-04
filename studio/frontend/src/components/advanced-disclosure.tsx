// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

/** Inline "Advanced" disclosure at the foot of a settings column, in place of a docked panel,
 *  so load-time tuning sits with the settings it affects. Open state is the caller's, so it can
 *  be persisted. `prominent` gives it a bordered row where the controls deserve advertising. */
export function AdvancedDisclosure({
  open,
  onOpenChange,
  prominent = false,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  prominent?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 border-t border-border/60 pt-3">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
        className={cn(
          "flex items-center gap-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          prominent
            ? "h-9 rounded-full border border-border bg-background px-3.5 hover:bg-accent/50"
            : "rounded-lg px-1 py-1.5 hover:bg-muted/60",
        )}
      >
        <HugeiconsIcon
          icon={Settings02Icon}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <span className="min-w-0 flex-1 text-xs font-medium text-foreground">
          Advanced
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open && (
        <div className="flex flex-col gap-3">
          <p className="text-ui-11 leading-snug text-muted-foreground">
            Load-time tuning. Changes apply on the next load; Reapply reloads
            the current model.
          </p>
          {children}
        </div>
      )}
    </div>
  );
}
