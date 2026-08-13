// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Wire shape of `transformers_upgrade` from /api/inference/validate. */
export interface TransformersUpgradeInfo {
  /** config.json model_type unknown to installed transformers. */
  model_type: string;
  /** Latest transformers release on PyPI at check time. */
  pypi_version?: string | null;
  /** Latest PyPI release ships this model_type (installable after consent). */
  supported_in_pypi?: boolean;
  /** Only transformers main ships it (dev-only; not installable). */
  supported_in_main?: boolean;
}

export type TransformersUpgradePhase = "consent" | "installing" | "error";

/** What `/api/inference/transformers-upgrade-check` says about one model. */
export interface TransformersUpgradeCheck {
  /** Set when no installed transformers ships the architecture but a newer one does. */
  upgrade: TransformersUpgradeInfo | null;
  /** The model ships its own modeling code, so a declined install still has a path. */
  requiresTrustRemoteCode: boolean;
  /** The latest sidecar already routes this model. */
  latestTierActive: boolean;
  /** A run started now loads 16-bit, not bnb 4-bit: the latest sidecar forces it. */
  forces16Bit: boolean;
}
