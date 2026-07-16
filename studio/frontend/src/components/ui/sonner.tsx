// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
  MultiplicationSignCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Spinner } from "@/components/ui/spinner";
import { useTheme } from "@/features/settings/stores/theme-store";
import { Toaster as Sonner, type ToasterProps } from "sonner";

// Make toast text selectable. Sonner's onPointerDown calls setPointerCapture(),
// which steals the drag and blocks text selection. dismissible:false would stop
// it but also kills the close button. So we swallow pointerdown on toast text
// (never on its buttons) before sonner sees it.
const handleToastPointerDownCapture = (
  event: React.PointerEvent<HTMLDivElement>,
) => {
  // closest() lives on Element, so this also covers SVG icon targets; guard
  // against non-Element targets defensively.
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
  // Use the resolved mode so sonner's data-sonner-theme always matches the
  // class the theme store puts on <html>.
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
          // App-wide arc spinner so loading toasts match the "Downloading model" toast.
          loading: <Spinner className="size-4 text-muted-foreground" />,
        }}
        style={
          {
            "--normal-bg": "var(--popover)",
            "--normal-text": "var(--popover-foreground)",
            // No border line; elevation comes from the composer's drop shadow.
            "--normal-border": "transparent",
            "--border-radius": "var(--radius)",
            // Pin the close button inside the toast's top-right corner.
            // Sonner defaults to the left/outside edge, so keep the horizontal
            // override here and the top offset in index.css.
            "--toast-close-button-start": "auto",
            "--toast-close-button-end": "8px",
            "--toast-close-button-transform": "none",
          } as React.CSSProperties
        }
        // No swipe gestures; text selection handled by the wrapper above.
        swipeDirections={[]}
        toastOptions={{
          classNames: {
            toast: "cn-toast",
            description: "!text-muted-foreground",
          },
        }}
        {...props}
      />
    </div>
  );
};

export { Toaster };
