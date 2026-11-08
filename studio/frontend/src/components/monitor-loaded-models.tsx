// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLoadedModels } from "@/features/loaded-models/use-loaded-models";
import type { LoadedModelEntry } from "@/features/loaded-models";
import { useEffect } from "react";

/** Lazy: keeps the chat runtime out of the startup graph. */
export default function MonitorLoadedModels({
  onEntries,
}: {
  onEntries: (entries: LoadedModelEntry[]) => void;
}) {
  const { polledEntries } = useLoadedModels(true, true);
  useEffect(() => onEntries(polledEntries), [onEntries, polledEntries]);
  return null;
}
