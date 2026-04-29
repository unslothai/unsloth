"use client";

import { openLink } from "@/lib/open-link";
import {
  memo,
  useState,
  useRef,
  useEffect,
  useCallback,
  type ComponentProps,
  type FC,
} from "react";
import { useMessage } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import { Badge, badgeVariants, type BadgeProps } from "./badge";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";

// ── Helpers ──────────────────────────────────────────────────

const extractDomain = (url: string): string => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

const getDomainInitial = (url: string): string => {
  const domain = extractDomain(url);
  return domain.charAt(0).toUpperCase();
};

// ── Sub-components ───────────────────────────────────────────

function SourceIcon({
  url,
  className,
  size = 3,
  ...props
}: ComponentProps<"span"> & { url: string; size?: number }) {
  const [hasError, setHasError] = useState(false);
  const domain = extractDomain(url);
  const SIZE_CLASSES: Record<number, string> = { 3: "size-3", 4: "size-4", 5: "size-5" };
  const sizeClass = SIZE_CLASSES[size] ?? "size-3";

  if (hasError) {
    return (
      <span
        data-slot="source-icon-fallback"
        className={cn(
          `flex ${sizeClass} shrink-0 items-center justify-center rounded-sm bg-muted font-medium text-[10px]`,
          className,
        )}
        {...props}
      >
        {getDomainInitial(url)}
      </span>
    );
  }

  return (
    <img
      data-slot="source-icon"
      src={`https://www.google.com/s2/favicons?domain=${domain}&sz=32`}
      alt=""
      className={cn(`${sizeClass} shrink-0 rounded-sm`, className)}
      onError={() => setHasError(true)}
      {...(props as ComponentProps<"img">)}
    />
  );
}

function SourceTitle({ className, ...props }: ComponentProps<"span">) {
  return (
    <span
      data-slot="source-title"
      className={cn("max-w-37.5 truncate", className)}
      {...props}
    />
  );
}

export type SourceProps = Omit<BadgeProps, "asChild"> &
  ComponentProps<"a"> & {
    asChild?: boolean;
  };

function Source({
  className,
  variant,
  size,
  asChild = false,
  href,
  onClick,
  ...props
}: SourceProps) {
  return (
    <Badge
      asChild
      variant={variant}
      size={size}
      className={cn(
        "cursor-pointer outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
        className,
      )}
    >
      <a
        data-slot="source"
        href={href}
        rel="noopener noreferrer"
        onClick={(e) => {
          if (href && openLink(href)) {
            e.preventDefault();
          }
          onClick?.(e);
        }}
        {...(props as ComponentProps<"a">)}
      />
    </Badge>
  );
}

// ── Source badge with hover card ─────────────────────────────

interface SourceData {
  url: string;
  title: string;
  description?: string;
}

const SourceBadge: FC<{ source: SourceData }> = ({ source }) => {
  const domain = extractDomain(source.url);
  const displayTitle = source.title || domain;

  return (
    <HoverCard openDelay={300} closeDelay={100}>
      <HoverCardTrigger asChild>
        <span className="inline-block">
          <Source href={source.url}>
            <SourceIcon url={source.url} />
            <SourceTitle>{displayTitle}</SourceTitle>
          </Source>
        </span>
      </HoverCardTrigger>
      <HoverCardContent side="top" align="start" className="w-72 p-3">
        <div className="flex gap-2.5">
          <SourceIcon url={source.url} size={4} className="mt-0.5 shrink-0" />
          <div className="min-w-0 space-y-1">
            <p className="text-sm font-semibold leading-tight truncate">
              {source.title || domain}
            </p>
            <p className="text-xs text-muted-foreground truncate">{domain}</p>
            {source.description && (
              <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">
                {source.description}
              </p>
            )}
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
};

// ── Grouped sources with 2-row collapse ─────────────────────

const SourcesGroup: FC = () => {
  const message = useMessage();
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);

  // Extract source parts from the message
  const sources: SourceData[] = [];
  if (message.content) {
    for (const part of message.content) {
      if (
        part.type === "source" &&
        "sourceType" in part &&
        part.sourceType === "url" &&
        "url" in part &&
        part.url
      ) {
        sources.push({
          url: part.url as string,
          title: (part as { title?: string }).title || "",
          description: (part as { metadata?: { description?: string } })
            .metadata?.description,
        });
      }
    }
  }

  // Measure how many badges fit in 2 rows
  const measure = useCallback(() => {
    const container = containerRef.current;
    if (!container || sources.length === 0) return;

    const children = Array.from(container.children) as HTMLElement[];
    if (children.length === 0) return;

    // Find the top of the first child as baseline
    const firstTop = children[0].offsetTop;
    let rowCount = 1;
    let prevTop = firstTop;
    let cutoff = children.length;

    for (let i = 1; i < children.length; i++) {
      const childTop = children[i].offsetTop;
      if (childTop > prevTop) {
        rowCount++;
        prevTop = childTop;
        if (rowCount > 2) {
          cutoff = i;
          break;
        }
      }
    }

    setVisibleCount(rowCount > 2 ? cutoff : null);
  }, [sources.length]);

  useEffect(() => {
    measure();
  }, [measure]);

  // Re-measure on resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    return () => observer.disconnect();
  }, [measure]);

  if (sources.length === 0) return null;

  const shouldCollapse = visibleCount !== null && visibleCount < sources.length;
  const displayedSources =
    expanded || !shouldCollapse ? sources : sources.slice(0, visibleCount);
  const hiddenCount = sources.length - (visibleCount ?? sources.length);

  return (
    <div className="relative mt-2">
      {/* Hidden measurement container — renders all badges to measure row positions */}
      <div
        ref={containerRef}
        aria-hidden
        className="flex w-full flex-wrap gap-1 invisible absolute pointer-events-none"
      >
        {sources.map((source) => (
          <span key={source.url} className="inline-block">
            <Source href={source.url}>
              <SourceIcon url={source.url} />
              <SourceTitle>{source.title || extractDomain(source.url)}</SourceTitle>
            </Source>
          </span>
        ))}
      </div>

      {/* Visible container */}
      <div className="flex flex-wrap gap-1">
        {displayedSources.map((source) => (
          <SourceBadge key={source.url} source={source} />
        ))}
        {shouldCollapse && !expanded && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className={cn(
              badgeVariants({ variant: "outline", size: "default" }),
              "cursor-pointer text-muted-foreground hover:text-foreground",
            )}
          >
            +{hiddenCount} more
          </button>
        )}
        {shouldCollapse && expanded && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className={cn(
              badgeVariants({ variant: "outline", size: "default" }),
              "cursor-pointer text-muted-foreground hover:text-foreground",
            )}
          >
            Show less
          </button>
        )}
      </div>
    </div>
  );
};

// ── Individual source (renders null — SourcesGroup handles all) ──

const SourcesNoop: FC<Record<string, unknown>> = () => null;

// ── Exports ──────────────────────────────────────────────────

const Sources = memo(SourcesNoop) as unknown as FC<Record<string, unknown>> & {
  Root: typeof Source;
  Icon: typeof SourceIcon;
  Title: typeof SourceTitle;
  Group: typeof SourcesGroup;
};

Sources.displayName = "Sources";
Sources.Root = Source;
Sources.Icon = SourceIcon;
Sources.Title = SourceTitle;
Sources.Group = SourcesGroup;

export {
  Sources,
  SourcesGroup,
  Source,
  SourceIcon,
  SourceTitle,
  badgeVariants as sourceVariants,
};
