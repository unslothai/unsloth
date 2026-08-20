


import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** The link out to another page's workspace (Images, Video, Audio, image training).
 *  Kept out of the mode strip and parked past a divider so it reads as leaving. */
export function MediaPageLink({
  to,
  label,
  icon,
  tooltip,
  onNavigate,
  labelClassName,
  arrowClassName,
}: {
  to: "/images" | "/video" | "/audio";
  label: string;
  icon: IconSvgElement;
  /** Needed on a translated page: the default prefix below is English. */
  tooltip?: string;
  /** Runs before the route change, for a destination whose mode lives in a store. */
  onNavigate?: () => void;
  /** Responsive callers can visually collapse the label while the button keeps its accessible name. */
  labelClassName?: string;
  /** Kept separate from the label because the outbound arrow is the first compact affordance to drop. */
  arrowClassName?: string;
}) {
  const navigate = useNavigate();
  return (
    <>
      {/* first:hidden, not a prop: the control to its left is conditional on the Images page,
          and a divider with nothing before it reads as a stray rule. */}
      <span
        aria-hidden="true"
        className="mx-0.5 h-4 w-px shrink-0 bg-border/70 first:hidden"
      />
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={label}
            onClick={() => {
              onNavigate?.();
              navigate({ to });
            }}
            className="flex h-[34px] min-w-0 items-center gap-1.5 rounded-full pl-2.5 pr-2 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HugeiconsIcon icon={icon} className="size-4 shrink-0" />
            <span className={cn("min-w-0 truncate", labelClassName)}>{label}</span>
            <HugeiconsIcon
              icon={ArrowRight02Icon}
              className={cn("size-3.5 shrink-0 opacity-60", arrowClassName)}
            />
          </button>
        </TooltipTrigger>
        <TooltipContent>{tooltip ?? `Go to ${label}`}</TooltipContent>
      </Tooltip>
    </>
  );
}
