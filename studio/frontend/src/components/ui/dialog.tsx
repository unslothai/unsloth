// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { AppPortalGate } from "@/components/app-readiness";

import { Dialog as DialogPrimitive } from "radix-ui";
import type * as React from "react";
import { createContext, useContext } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

const DialogPortalContainerContext = createContext<HTMLElement | null>(null);

export function useDialogPortalContainer(): HTMLElement | null {
  return useContext(DialogPortalContainerContext);
}

function Dialog({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Root>) {
  return <DialogPrimitive.Root data-slot="dialog" {...props} />;
}

function DialogTrigger({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Trigger>) {
  return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />;
}

function DialogPortal({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Portal>) {
  return (
    <AppPortalGate>
      <DialogPrimitive.Portal data-slot="dialog-portal" {...props} />
    </AppPortalGate>
  );
}

function DialogClose({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Close>) {
  return <DialogPrimitive.Close data-slot="dialog-close" {...props} />;
}

function DialogOverlay({
  className,
  position = "fixed",
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Overlay> & {
  position?: "fixed" | "absolute";
}) {
  return (
    <DialogPrimitive.Overlay
      data-slot="dialog-overlay"
      data-viewport-backdrop={position === "fixed" ? true : undefined}
      className={cn(
        "data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 bg-black/30 supports-backdrop-filter:backdrop-blur-[2px] duration-100  inset-0 isolate z-50",
        position === "fixed" ? "fixed" : "absolute",
        className,
      )}
      {...props}
    />
  );
}

function DialogContent({
  className,
  children,
  showCloseButton = true,
  container,
  position = "fixed",
  overlayClassName,
  overlayPosition,
  onInteractOutside,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & {
  showCloseButton?: boolean;
  container?: HTMLElement | null;
  position?: "fixed" | "absolute";
  overlayClassName?: string;
  overlayPosition?: "fixed" | "absolute";
}) {
  const resolvedContainer = container ?? null;
  return (
    <DialogPortalContainerContext.Provider value={resolvedContainer}>
      <DialogPortal container={resolvedContainer ?? undefined}>
        <DialogOverlay
          className={overlayClassName}
          position={overlayPosition ?? position}
        />
        <DialogPrimitive.Content
          data-slot="dialog-content"
          className={cn(
            // max-h + scroll keeps tall dialogs reachable; call sites may override (twMerge).
            "bg-background data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 data-closed:zoom-out-95 data-open:zoom-in-95 ring-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-edge-gain,1)),transparent)] grid max-h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] max-w-[calc(100%-2rem)] gap-6 overflow-y-auto scroll-rounded rounded-4xl px-7 pt-8 pb-7 text-sm ring-1 duration-100 sm:max-w-md top-1/2 left-1/2 z-50 w-full -translate-x-1/2 -translate-y-1/2",
            // Fixed dialogs center below the z-70 desktop titlebar and go full screen at phone width.
            position === "fixed"
              // translate-none, not translate-x-0: any translate makes this the close button's
              // containing block.
              ? "fixed top-[calc(50%+var(--studio-window-chrome-top,0px)/2)] max-sm:top-[var(--studio-window-chrome-top,0px)] max-sm:left-0 max-sm:h-[calc(100dvh-var(--studio-window-chrome-top,0px))] max-sm:w-dvw max-sm:max-h-none max-sm:max-w-none max-sm:translate-none max-sm:rounded-none max-sm:ring-0"
              : "absolute",
            className,
          )}
          onInteractOutside={(event) => {
            onInteractOutside?.(event);
            // toasts render outside every dialog, so using one must not dismiss the dialog under it.
            if ((event.target as Element | null)?.closest?.("[data-sonner-toaster]")) {
              event.preventDefault();
            }
          }}
          {...props}
        >
          {children}
          {showCloseButton && (
            <DialogPrimitive.Close data-slot="dialog-close" asChild>
              <Button
                variant="ghost"
                // Fixed on phones so the close stays put while scrolling; offset below the titlebar.
                className={cn(
                  "absolute top-5 right-5 z-10",
                  position === "fixed" &&
                    "max-sm:fixed max-sm:top-[calc(1.25rem+var(--studio-window-chrome-top,0px))]",
                )}
                size="icon-sm"
              >
                <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} />
                <span className="sr-only">Close</span>
              </Button>
            </DialogPrimitive.Close>
          )}
        </DialogPrimitive.Content>
      </DialogPortal>
    </DialogPortalContainerContext.Provider>
  );
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="dialog-header"
      className={cn("gap-2 flex flex-col", className)}
      {...props}
    />
  );
}

function DialogFooter({
  className,
  showCloseButton = false,
  children,
  ...props
}: React.ComponentProps<"div"> & {
  showCloseButton?: boolean;
}) {
  return (
    <div
      data-slot="dialog-footer"
      className={cn(
        "flex flex-col-reverse gap-2 sm:flex-row sm:justify-end",
        className,
      )}
      {...props}
    >
      {children}
      {showCloseButton && (
        <DialogPrimitive.Close asChild>
          <Button variant="outline">Close</Button>
        </DialogPrimitive.Close>
      )}
    </div>
  );
}

function DialogTitle({
  className,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Title>) {
  return (
    <DialogPrimitive.Title
      data-slot="dialog-title"
      className={cn("font-heading text-lg leading-none font-semibold", className)}
      {...props}
    />
  );
}

function DialogDescription({
  className,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Description>) {
  return (
    <DialogPrimitive.Description
      data-slot="dialog-description"
      className={cn(
        "text-muted-foreground *:[a]:hover:text-foreground text-sm *:[a]:underline *:[a]:underline-offset-3",
        className,
      )}
      {...props}
    />
  );
}

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogPortalContainerContext,
  DialogTitle,
  DialogTrigger,
};
