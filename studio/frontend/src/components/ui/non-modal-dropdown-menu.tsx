// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps, ReactNode, RefObject } from "react";
import { useEffect, useRef, useState } from "react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MenuDismissGuard } from "@/lib/menu-dismiss-guard";

/** Non-modal: a modal menu writes inherited `pointer-events` on <body>, restyling everything. */
export function NonModalDropdownMenu({
  trigger,
  children,
  onCloseAutoFocus,
  ...contentProps
}: {
  trigger: (ref: RefObject<HTMLButtonElement | null>) => ReactNode;
  children: ReactNode;
} & ComponentProps<typeof DropdownMenuContent>) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  // Gates the guard: the content outlives the close, and a lingering guard swallows the next click.
  const [open, setOpen] = useState(false);
  // Radix pins the content once the trigger scrolls out; close on a scroll of a trigger ancestor.
  const closedByScroll = useRef(false);
  useEffect(() => {
    if (!open) return;
    const onScroll = (event: Event) => {
      const trigger = triggerRef.current;
      const target = event.target;
      if (!trigger || !(target instanceof Node) || !target.contains(trigger)) return;
      closedByScroll.current = true;
      setOpen(false);
    };
    document.addEventListener("scroll", onScroll, { capture: true, passive: true });
    return () => document.removeEventListener("scroll", onScroll, { capture: true });
  }, [open]);
  return (
    <DropdownMenu modal={false} open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild={true}>
        {trigger(triggerRef)}
      </DropdownMenuTrigger>
      <DropdownMenuContent
        {...contentProps}
        // Focus alone: returning it would scroll the trigger back and undo the closing scroll.
        onCloseAutoFocus={(event) => {
          onCloseAutoFocus?.(event);
          if (!closedByScroll.current) return;
          closedByScroll.current = false;
          if (event.defaultPrevented) return;
          event.preventDefault();
          // Preventing Radix's restore skips its check that focus moved elsewhere, so do it here.
          const active = document.activeElement;
          const claimedByTheUser =
            active !== null &&
            active !== document.body &&
            active !== document.documentElement &&
            active.closest("[data-slot='dropdown-menu-content']") === null;
          if (claimedByTheUser) return;
          triggerRef.current?.focus({ preventScroll: true });
        }}
      >
        {/* Arming survives this unmount: `arm` registers on `document`. */}
        {open ? <MenuDismissGuard triggerRef={triggerRef} /> : null}
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
