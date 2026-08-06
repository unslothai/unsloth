


import type { ReactNode } from "react";

import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

/** Inline "Advanced" disclosure at the foot of a settings column, in place of a docked panel,
 *  so load-time tuning sits with the settings it affects. Open state is the caller's, so it can
 *  be persisted. */
export function AdvancedDisclosure({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) {
  return (
    // mt-2 over pt-4 sits the rule between the field above and this row, not up against it.
    <div className="mt-2 flex flex-col gap-3 border-t border-border/60 pt-4 pb-4">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
        className="flex items-center gap-2 rounded-lg px-1 py-1.5 text-left transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {/* Icon and label sized to the slider rows above, so it reads as one of them. */}
        <HugeiconsIcon
          icon={Settings02Icon}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
        <span className="min-w-0 flex-1 text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground">
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
