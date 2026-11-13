// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../../types";

export function isTextFormatValidator(config: NodeConfig): boolean {
  return (
    config.kind === "validator" &&
    (config.validator_type === "json" || config.validator_type === "markdown")
  );
}

export function isTextFormatValidatorTarget(config: NodeConfig): boolean {
  return (
    config.kind === "llm" ||
    config.kind === "expression" ||
    config.kind === "sampler"
  );
}

export function isSemanticRelation(
  source: NodeConfig,
  target: NodeConfig,
): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    return true;
  }
  if (source.kind === "tool_config" && target.kind === "llm") {
    return true;
  }
  if (
    source.kind === "llm" &&
    source.llm_type === "code" &&
    target.kind === "validator"
  ) {
    return true;
  }
  if (isTextFormatValidator(target) && isTextFormatValidatorTarget(source)) {
    return true;
  }
  return (
    source.kind === "validator" &&
    !isTextFormatValidator(source) &&
    target.kind === "llm" &&
    target.llm_type === "code"
  );
}
