// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps, ReactNode, RefObject } from "react";
import { useRef, useState } from "react";

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
  return (
    <DropdownMenu modal={false} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild={true}>
        {trigger(triggerRef)}
      </DropdownMenuTrigger>
      <DropdownMenuContent {...contentProps}>
        {/* Arming survives this unmount: `arm` registers on `document`, so the click owed by the
            dismissing press is still swallowed after the guard itself has gone. */}
        {open ? <MenuDismissGuard triggerRef={triggerRef} /> : null}
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
