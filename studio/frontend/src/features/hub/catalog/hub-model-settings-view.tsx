// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Full-page settings for one model, opened from the Hub.
//
// The same controls exist in the chat picker's popover, but a popover is a poor
// place to work through every knob a model has. This gives them a page, and
// states plainly that whatever is saved here is what an API load will use --
// the settings are mirrored server-side by ModelConfigPage's save.

import { ModelConfigPage, type ModelPickTarget } from "@/features/model-picker";
import type { PerModelConfig } from "@/features/model-picker";
import { ArrowLeft01Icon, Globe02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";

export function HubModelSettingsView({
  target,
  loadedConfig = null,
  loadedContextLength = null,
  onBack,
  onRun,
  compact = false,
}: {
  target: ModelPickTarget;
  /** Non-null when this model is the loaded one, so the page can show live values. */
  loadedConfig?: PerModelConfig | null;
  loadedContextLength?: number | null;
  onBack: () => void;
  /** Apply + load with these settings. */
  onRun: (config: PerModelConfig) => void;
  compact?: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrolled, setScrolled] = useState(false);
  // Mirrors HubDetailView so this view sits at the same measure as the rest of
  // the Hub rather than introducing a third column width.
  const measure = compact
    ? "mx-auto w-full max-w-[860px] px-5 sm:px-5"
    : "mx-auto w-full max-w-[1100px] px-5 sm:px-8";

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const next = el.scrollTop > 0;
      setScrolled((current) => (current === next ? current : next));
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        aria-hidden="true"
        data-scrolled={scrolled || undefined}
        className="hub-scroll-fade pointer-events-none absolute inset-x-0 top-0 z-10 h-7"
      />
      <div
        ref={scrollRef}
        data-hub-scroll="true"
        className={cn(
          "min-h-0 flex-1 overflow-x-hidden overflow-y-auto [overflow-anchor:none] [scrollbar-width:thin]",
          compact
            ? "mr-2 [scrollbar-gutter:stable]"
            : "[scrollbar-gutter:stable_both-edges]",
        )}
      >
        <div
          className="hub-detail-bar sticky top-0 z-20"
          data-scrolled={scrolled || undefined}
        >
          <div className={`${measure} py-3`}>
            <button
              type="button"
              onClick={onBack}
              className="-ml-1.5 inline-flex h-8 cursor-pointer select-none items-center gap-1.5 rounded-full pl-1.5 pr-2.5 text-ui-12p5 font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground dark:hover:bg-white/[0.06]"
            >
              <HugeiconsIcon
                icon={ArrowLeft01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              Back to Hub
            </button>
          </div>
        </div>

        <div className={cn(measure, "pb-20")}>
          <header className="flex flex-col gap-1 pb-5">
            <h1 className="min-w-0 break-words text-ui-24 font-semibold leading-[1.1] tracking-[-0.022em] text-foreground">
              {target.displayName}
            </h1>
            <p className="min-w-0 break-all text-ui-12 text-muted-foreground">
              {target.id}
              {target.ggufVariant ? ` · ${target.ggufVariant}` : ""}
            </p>
          </header>

          <div className="mb-5 flex items-start gap-2.5 rounded-xl border border-border/60 bg-card px-4 py-3">
            <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-muted/40">
              <HugeiconsIcon
                icon={Globe02Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </span>
            <p className="min-w-0 text-ui-12 leading-[1.5] text-muted-foreground">
              {/* Only a GGUF is mirrored to the server, because API auto-switch
                  indexes GGUFs only, so promising the API case for anything
                  else describes a load that cannot happen. */}
              {target.isGguf
                ? "Saved settings apply everywhere this model loads, including when an OpenAI-compatible API request asks for it."
                : "Saved settings apply everywhere Studio loads this model."}{" "}
              Turn on{" "}
              <span className="font-medium text-foreground">
                Remember for this model
              </span>{" "}
              below to keep them.
            </p>
          </div>

          <div className="rounded-xl border border-border/60 bg-card px-4 py-4">
            <ModelConfigPage
              key={`${target.id}::${target.ggufVariant ?? ""}`}
              target={target}
              onRun={onRun}
              loadedConfig={loadedConfig}
              loadedContextLength={loadedContextLength}
              variant="page"
              // The page heading above already names the model; the built-in
              // "Run settings" block would print it a second time.
              showHeader={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
