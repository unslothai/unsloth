// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only the barrel needs stubbing: it also exports the React panel, which bare Node cannot resolve. The helpers under test stay real.
export type { ManagedDownload } from "../../../src/features/hub/download-manager/download-manager-types.ts";
export {
  downloadInventoryHintKind,
  downloadRequestInventoryKind,
  scopedDownloadInventoryKind,
} from "../../../src/features/hub/download-manager/download-manager-types.ts";

export function clearCompletedInventoryHint(): void {
}

export const useDownloadManagerStore = Object.assign(
  () => ({ jobs: {} }),
  {
    getState: () => ({ jobs: {} }),
    subscribe: () => () => {},
  },
);
