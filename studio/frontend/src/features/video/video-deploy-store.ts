// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import type { AdapterDeployRequest } from "@/features/images/stores/image-workflow-store";

/** A video-family adapter deployed from a Train panel: Video takes it on its next active render and loads the
 * family's base with the adapter baked in. Images hands its video adapters over through here. */
interface VideoDeployState {
  pendingDeploy: AdapterDeployRequest | null;
  requestDeploy: (request: AdapterDeployRequest) => void;
  takeDeploy: () => AdapterDeployRequest | null;
}

export const useVideoDeployStore = create<VideoDeployState>((set, get) => ({
  pendingDeploy: null,
  requestDeploy: (pendingDeploy) => set({ pendingDeploy }),
  takeDeploy: () => {
    const request = get().pendingDeploy;
    if (request) set({ pendingDeploy: null });
    return request;
  },
}));
