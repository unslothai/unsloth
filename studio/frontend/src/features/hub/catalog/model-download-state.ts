// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SelectedModelView } from "../types";

export type ModelDownloadState = {
  isDownloaded: boolean;
  isPartial: boolean;
  partialTransport: string | null;
  partialResumable: boolean;
};

export function modelDownloadState(
  model: SelectedModelView,
): ModelDownloadState {
  return {
    isDownloaded: model.isDownloaded,
    isPartial: model.isPartial ?? false,
    partialTransport: model.partialTransport ?? null,
    partialResumable: model.partialResumable === true,
  };
}
