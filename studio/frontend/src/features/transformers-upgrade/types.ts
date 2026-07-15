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
