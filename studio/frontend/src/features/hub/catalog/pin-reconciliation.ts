// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  parsedPinsForRepo,
  removePinnedArtifactIfPresent,
  usePinnedModelsStore,
} from "@/stores/pinned-models";
import {
  listCachedGguf,
  listCachedModels,
  listLocalModels,
} from "../inventory/api";
import {
  buildLocalPinCleanupEvidence,
  localPinInventoryNeeds,
  pinsToRemoveAfterLocalCacheDelete,
} from "./pin-cleanup";
import { downloadedGgufQuantsInCacheCopies } from "./remaining-gguf-copies";

export async function downloadedGgufQuantsAfterCacheDelete({
  repoId,
  hfToken,
}: {
  repoId: string;
  hfToken?: string;
}): Promise<Set<string> | null> {
  const inventories = await Promise.all([
    listCachedGguf(hfToken),
    listLocalModels(),
  ]).catch(() => null);
  if (!inventories) {
    return null;
  }
  const [remainingGgufRows, remainingLocalRows] = inventories;
  const evidence = buildLocalPinCleanupEvidence(
    repoId,
    remainingGgufRows,
    [],
    remainingLocalRows.models,
  );
  if (evidence.ggufState === "absent") {
    return new Set();
  }
  if (evidence.ggufState === "uncertain") {
    return null;
  }
  return downloadedGgufQuantsInCacheCopies(
    repoId,
    evidence.ggufCacheCopies,
    hfToken,
  );
}

export async function reconcilePinsAfterCacheCopyDelete({
  repoId,
  hfToken,
}: {
  repoId: string;
  hfToken?: string;
}): Promise<void> {
  const initialPins = parsedPinsForRepo(
    usePinnedModelsStore.getState().pinned,
    repoId,
  );
  if (initialPins.length === 0) {
    return;
  }
  const inventoryNeeds = localPinInventoryNeeds(initialPins);
  const inventories = await Promise.all([
    inventoryNeeds.gguf ? listCachedGguf(hfToken) : Promise.resolve([]),
    inventoryNeeds.models ? listCachedModels(hfToken) : Promise.resolve([]),
    listLocalModels(),
  ]).catch(() => null);
  if (!inventories) {
    return;
  }
  const [remainingGgufRows, remainingModelRows, remainingLocalRows] =
    inventories;
  const evidence = buildLocalPinCleanupEvidence(
    repoId,
    remainingGgufRows,
    remainingModelRows,
    remainingLocalRows.models,
  );
  const currentPins = parsedPinsForRepo(
    usePinnedModelsStore.getState().pinned,
    repoId,
  ).filter((pin) =>
    pin.quant === null ? inventoryNeeds.models : inventoryNeeds.gguf,
  );
  const representedGgufQuants =
    currentPins.some((pin) => pin.quant !== null) &&
    evidence.ggufState === "represented"
      ? await downloadedGgufQuantsInCacheCopies(
          repoId,
          evidence.ggufCacheCopies,
          hfToken,
        )
      : null;
  for (const pin of pinsToRemoveAfterLocalCacheDelete(
    currentPins,
    evidence,
    representedGgufQuants,
  )) {
    removePinnedArtifactIfPresent(pin.repoId, pin.quant);
  }
}
