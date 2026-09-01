// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FC, RefObject } from "react";

import { useDismissingClickGuard } from "@/lib/menu-dismiss";

/**
 * Marker mounted inside non-modal menu content. The lifetime is mount-scoped: exit-animated
 * content can outlive the open state, so animated menus must add explicit open-state gating.
 */
export const MenuDismissGuard: FC<{
  triggerRef: RefObject<HTMLElement | null>;
}> = ({ triggerRef }) => {
  useDismissingClickGuard(triggerRef);
  return null;
};
