// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TransformersUpgradeInfo {
  model_type: string;
  pypi_version?: string | null;
  supported_in_pypi?: boolean;
  supported_in_main?: boolean;
}

export type TransformersUpgradePhase = "consent" | "installing" | "error";

export interface TransformersUpgradeCheck {
  upgrade: TransformersUpgradeInfo | null;
  /** A declined install still has a path via the model's own code. */
  requiresTrustRemoteCode: boolean;
  latestTierActive: boolean;
  /** The latest sidecar forces 16-bit. */
  forces16Bit: boolean;
  /** The checkpoint resumes only in 4-bit, which installing would remove. Set only for resumes. */
  installBreaksExactResume: boolean;
}

/** Same four fields the remote-code scan takes, so both read the same config.json. */
export interface ModelCachePin {
  preferLocalCache?: boolean;
  modelLocalPath?: string | null;
  modelSnapshotPath?: string | null;
  modelSnapshotRepoId?: string | null;
}
