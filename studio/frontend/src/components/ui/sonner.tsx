// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
  CancelCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useTheme } from "@/features/settings/stores/theme-store";
import { createLoadingToastIcon } from "@/lib/toast";
import { cornerInsetScope } from "@/lib/toast-offset";
import { Toaster as Sonner, type ToasterProps } from "sonner";

// Sonner's setPointerCapture blocks text selection; swallow pointerdown on toast text only.
const handleToastPointerDownCapture = (
  event: React.PointerEvent<HTMLDivElement>,
) => {
  const target = event.target as Element | null;
  if (typeof target?.closest !== "function") return;
  if (!target.closest("[data-sonner-toast]")) return;
  if (
    target.closest("button,[data-button],[data-close-button],[data-cancel]")
  ) {
    return;
  }
  event.stopPropagation();
};

const Toaster = ({ ...props }: ToasterProps) => {
  // Use the resolved mode so data-sonner-theme matches the class on <html>.
  const { resolved } = useTheme();

  return (
    // display:contents adds no box; only carries the selection-fix handler.
    // biome-ignore lint/a11y/noStaticElementInteractions: capture-only guard, not interactive
    <div
      ref={cornerInsetScope}
      style={{ display: "contents" }}
      onPointerDownCapture={handleToastPointerDownCapture}
    >
      <Sonner
        theme={resolved}
        className="toaster group"
        duration={5000}
        icons={{
          success: (
            <HugeiconsIcon
              icon={CheckmarkCircle02Icon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          info: (
            <HugeiconsIcon
              icon={InformationCircleIcon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          warning: (
            <HugeiconsIcon
              icon={Alert02Icon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          error: (
            <HugeiconsIcon
              icon={CancelCircleIcon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          loading: createLoadingToastIcon(),
        }}
        style={
          {
            "--normal-bg": "var(--popover)",
            "--normal-text": "var(--popover-foreground)",
            "--normal-border": "transparent",
            // Just above the compact-box tier; at --radius-xl a two-line toast reads as a capsule.
            "--border-radius": "calc(var(--radius-lg) + 2px)",
            // Pin the close button inside the toast's top-right corner.
            // Sonner defaults to the left/outside edge, so keep the horizontal
            // override here and the top offset in index.css.
            "--toast-close-button-start": "auto",
            "--toast-close-button-end": "12px",
            "--toast-close-button-transform": "none",
          } as React.CSSProperties
        }
        swipeDirections={[]}
        toastOptions={{
          classNames: {
            // an open modal dialog sets pointer-events:none on body, which toasts would otherwise inherit.
            toast: "cn-toast pointer-events-auto",
            description: "!text-muted-foreground",
          },
        }}
        {...props}
      />
    </div>
  );
};

export { Toaster };
