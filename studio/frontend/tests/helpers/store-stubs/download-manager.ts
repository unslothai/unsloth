// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Real logic, re-exported: only the barrel needs stubbing, because it also
// exports the React panel, which bare Node cannot resolve. Importing the
// modules directly keeps the classification helpers under test genuine.
export type { ManagedDownload } from "../../../src/features/hub/download-manager/download-manager-types.ts";
export {
  downloadInventoryHintKind,
  downloadRequestInventoryKind,
  scopedDownloadInventoryKind,
} from "../../../src/features/hub/download-manager/download-manager-types.ts";

/** Not exercised by inventory tests: the hint clear is a store write. */
export function clearCompletedInventoryHint(): void {
  // No hint store in these tests.
}

export const useDownloadManagerStore = Object.assign(
  () => ({ jobs: {} }),
  {
    getState: () => ({ jobs: {} }),
    subscribe: () => () => {},
  },
);
