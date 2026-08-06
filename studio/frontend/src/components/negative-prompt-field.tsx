


import { ChevronDown } from "lucide-react";

import { InfoHint } from "@/components/ui/info-hint";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// The hint and chevron only appear with the row, so a rarely used field stays quiet.
const REVEAL_ON_ROW_HOVER =
  "pointer-events-none inline-flex opacity-0 transition-opacity group-hover/negative:pointer-events-auto group-hover/negative:opacity-100 group-focus-within/negative:pointer-events-auto group-focus-within/negative:opacity-100";

/** Collapsible negative prompt, shared by the Images and Video Create panels. */
export function NegativePromptField({
  value,
  onChange,
  open,
  onOpenChange,
  hint,
}: {
  value: string;
  onChange: (value: string) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  hint: string;
}) {
  return (
    <div className="group/negative -mt-1 flex flex-col gap-1.5 pb-2">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          aria-expanded={open}
          className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Negative prompt
        </button>
        <InfoHint>{hint}</InfoHint>
        <button
          type="button"
          // The labelled button above is the accessible toggle; this is decoration.
          aria-hidden={true}
          tabIndex={-1}
          onClick={() => onOpenChange(!open)}
          className={cn(
            REVEAL_ON_ROW_HOVER,
            open && "pointer-events-auto opacity-100",
          )}
        >
          <ChevronDown
            className={cn(
              "size-3 text-muted-foreground transition-transform",
              open && "rotate-180",
            )}
          />
        </button>
      </div>
      {open && (
        <Textarea
          rows={2}
          placeholder="What to avoid (optional)"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}
