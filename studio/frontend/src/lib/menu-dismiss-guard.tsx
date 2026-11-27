// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FC, RefObject } from "react";

import { useDismissingClickGuard } from "@/lib/menu-dismiss";

/** Mount-scoped: exit-animated content can outlive open state, so animated menus need open gating. */
export const MenuDismissGuard: FC<{
  triggerRef: RefObject<HTMLElement | null>;
}> = ({ triggerRef }) => {
  useDismissingClickGuard(triggerRef);
  return null;
};
