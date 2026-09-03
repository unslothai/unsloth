// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps, ReactNode, RefObject } from "react";
import { useRef } from "react";

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
  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild={true}>
        {trigger(triggerRef)}
      </DropdownMenuTrigger>
      <DropdownMenuContent {...contentProps}>
        <MenuDismissGuard triggerRef={triggerRef} />
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
