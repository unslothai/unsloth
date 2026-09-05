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

/** A dropdown that does not take the document with it when it opens: modal writes an inherited
 *  `pointer-events` on `<body>`, so every open invalidates style for the whole mounted subtree.
 *  `trigger` takes the ref so each mount owns its own, which a menu rendered per row needs. */
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
  // `DropdownMenuContent` animates out, so Radix keeps it mounted for the whole exit animation.
  // The guard is mount-scoped, so an ungated one goes on watching `document` after the menu has
  // closed and swallows the next click the user makes. Gate it on the open state instead.
  const [open, setOpen] = useState(false);
  // Nothing locks scroll behind a non-modal menu, so the list its trigger sits in can move while
  // it is open. Radix keeps the content pinned to the viewport edge once the trigger has scrolled
  // out, which leaves a menu acting on a row the user can no longer see. Close on any scroll of an
  // ancestor of the trigger; the menu's own viewport does not contain it and scrolls untouched.
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
        // Returning focus to the trigger scrolls it back into view, which would undo the very
        // scroll that closed the menu. Keep the focus, drop the scroll.
        onCloseAutoFocus={(event) => {
          onCloseAutoFocus?.(event);
          if (!closedByScroll.current) return;
          closedByScroll.current = false;
          if (event.defaultPrevented) return;
          event.preventDefault();
          triggerRef.current?.focus({ preventScroll: true });
        }}
      >
        {/* Arming survives this unmount: `arm` registers on `document`, so the click owed by the
            dismissing press is still swallowed after the guard itself has gone. */}
        {open ? <MenuDismissGuard triggerRef={triggerRef} /> : null}
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
