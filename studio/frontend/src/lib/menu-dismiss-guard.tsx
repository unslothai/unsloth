// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FC } from "react";

import { useDismissingClickGuard } from "@/lib/menu-dismiss";

/**
 * Renders nothing; its only job is to be mounted for exactly as long as a non-modal menu's
 * content is. Radix mounts content on open and unmounts it on close, so a child of the content
 * is the cheapest correct signal for "a menu is open", and it needs no `onOpenChange` wiring at
 * the twelve call sites.
 *
 * See menu-dismiss.ts for why the guard has to watch `pointerdown` rather than take Radix's
 * `onPointerDownOutside`.
 */
export const MenuDismissGuard: FC = () => {
  useDismissingClickGuard();
  return null;
};
