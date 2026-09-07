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

/** A dropdown that leaves `<body>` alone: modal writes an inherited `pointer-events` there, so
 *  every open restyles the whole subtree. `trigger` takes the ref so a per-row menu owns one. */
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
  // Gates the guard. The content animates out, so it outlives the close, and a mount-scoped
  // guard left watching `document` swallows the next click the user makes.
  const [open, setOpen] = useState(false);
  // Nothing locks scroll here, and Radix pins the content to the viewport edge once the trigger
  // scrolls out, leaving a menu acting on a row the user cannot see. Close on a scroll of a
  // trigger ancestor; the menu's own viewport does not contain it, so it scrolls untouched.
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
