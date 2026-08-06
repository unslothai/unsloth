


import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/** The link between the two media pages (Images and Video). Kept out of either page's mode
 *  strip, which switches modes within a page, and parked at the far right past a divider so
 *  it reads as leaving rather than as another mode. */
export function MediaPageLink({
  to,
  label,
  icon,
}: {
  to: "/images" | "/video";
  label: string;
  icon: IconSvgElement;
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
            onClick={() => {
              navigate({ to });
            }}
            className="flex h-[34px] shrink-0 items-center gap-1.5 rounded-full pl-2.5 pr-2 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HugeiconsIcon icon={icon} className="size-4 shrink-0" />
            <span>{label}</span>
            <HugeiconsIcon
              icon={ArrowRight02Icon}
              className="size-3.5 shrink-0 opacity-60"
            />
          </button>
        </TooltipTrigger>
        <TooltipContent>Go to {label}</TooltipContent>
      </Tooltip>
    </>
  );
}
