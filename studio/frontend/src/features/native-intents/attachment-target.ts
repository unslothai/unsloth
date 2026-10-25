// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

export const NativeAttachmentDragContext = createContext(false);

export const NativeAttachmentTargetContext = createContext<string | null>(null);

// The runtime can reuse its local thread ID for successive unsent drafts.
// Keep their ownership distinct even when a consumer remounts or re-keys.
export const NativeAttachmentOwnerScopeContext = createContext<string | null>(null);

export function useNativeAttachmentTargetKey(): string | null {
  return useContext(NativeAttachmentTargetContext);
}

export function useNativeAttachmentOwnerScope(): string | null {
  return useContext(NativeAttachmentOwnerScopeContext);
}
