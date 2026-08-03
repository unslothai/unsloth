// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorConfig } from "../../types";

export function isValidatorConsentRequired(config: ValidatorConfig): boolean {
  return (
    (config.validator_type === "tool" && config.tool_acknowledged !== true) ||
    (config.validator_type === "custom" && config.custom_acknowledged !== true)
  );
}
