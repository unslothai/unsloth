// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
  MultiplicationSignCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useTheme } from "@/features/settings/stores/theme-store";
import { createLoadingToastIcon } from "@/lib/toast";
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
              icon={MultiplicationSignCircleIcon}
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
            "--border-radius": "calc(var(--radius) + 8px)",
            // Sonner defaults to the outside edge; the top offset lives in index.css.
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
