// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TauriUpdateController } from "@/hooks/use-tauri-update";
import { createContext, useContext } from "react";

export const TauriUpdateContext = createContext<TauriUpdateController | null>(
  null,
);

export function useTauriUpdateController(): TauriUpdateController | null {
  return useContext(TauriUpdateContext);
}
