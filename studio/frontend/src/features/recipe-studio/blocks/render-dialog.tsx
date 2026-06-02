// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import type { NodeConfig, SamplerConfig } from "../types";
import { getBlockDefinitionForConfig } from "./definitions";
import { ExpressionDialog } from "../dialogs/expression/expression-dialog";
import { LlmDialog } from "../dialogs/llm/llm-dialog";
import { ModelConfigDialog } from "../dialogs/models/model-config-dialog";
import { ModelProviderDialog } from "../dialogs/models/model-provider-dialog";
import { SeedDialog } from "../dialogs/seed/seed-dialog";
import { CategoryDialog } from "../dialogs/samplers/category-dialog";
import { DatetimeDialog } from "../dialogs/samplers/datetime-dialog";
import { BernoulliDialog } from "../dialogs/samplers/bernoulli-dialog";
import { GaussianDialog } from "../dialogs/samplers/gaussian-dialog";
import { PersonDialog } from "../dialogs/samplers/person-dialog";
import { SubcategoryDialog } from "../dialogs/samplers/subcategory-dialog";
import { TimedeltaDialog } from "../dialogs/samplers/timedelta-dialog";
import { UniformDialog } from "../dialogs/samplers/uniform-dialog";
import { UuidDialog } from "../dialogs/samplers/uuid-dialog";
import { MarkdownNoteDialog } from "../dialogs/markdown-note/markdown-note-dialog";
import { ToolProfileDialog } from "../dialogs/tool-profile/tool-profile-dialog";
import { ValidatorDialog } from "../dialogs/validators/validator-dialog";

export function renderBlockDialog(
  config: NodeConfig | null,
  open: boolean,
  categoryOptions: SamplerConfig[],
  modelConfigAliases: string[],
  modelProviderOptions: string[],
  localProviderNames: Set<string>,
  toolProfileAliases: string[],
  datetimeOptions: string[],
  onUpdate: (id: string, patch: Partial<NodeConfig>) => void,
): ReactElement | null {
  const definition = getBlockDefinitionForConfig(config);
  if (!definition || !config) {
    return null;
  }

  const update = (patch: Partial<NodeConfig>) => onUpdate(config.id, patch);

  switch (definition.dialogKey) {
    case "seed":
      return config.kind === "seed" ? (
        <SeedDialog config={config} onUpdate={update} open={open} />
      ) : null;
    case "category":
      return config.kind === "sampler" && config.sampler_type === "category" ? (
        <CategoryDialog key={config.id} config={config} onUpdate={update} />
      ) : null;
    case "subcategory":
      return config.kind === "sampler" && config.sampler_type === "subcategory" ? (
        <SubcategoryDialog
          config={config}
          categoryOptions={categoryOptions}
          onUpdate={update}
        />
      ) : null;
    case "uniform":
      return config.kind === "sampler" && config.sampler_type === "uniform" ? (
        <UniformDialog config={config} onUpdate={update} />
      ) : null;
    case "gaussian":
      return config.kind === "sampler" && config.sampler_type === "gaussian" ? (
        <GaussianDialog config={config} onUpdate={update} />
      ) : null;
    case "bernoulli":
      return config.kind === "sampler" && config.sampler_type === "bernoulli" ? (
        <BernoulliDialog config={config} onUpdate={update} />
      ) : null;
    case "datetime":
      return config.kind === "sampler" && config.sampler_type === "datetime" ? (
        <DatetimeDialog config={config} onUpdate={update} />
      ) : null;
    case "timedelta":
      return config.kind === "sampler" && config.sampler_type === "timedelta" ? (
        <TimedeltaDialog
          config={config}
          datetimeOptions={datetimeOptions}
          onUpdate={update}
        />
      ) : null;
    case "uuid":
      return config.kind === "sampler" && config.sampler_type === "uuid" ? (
        <UuidDialog config={config} onUpdate={update} />
      ) : null;
    case "person":
      return config.kind === "sampler" &&
        (config.sampler_type === "person" ||
          config.sampler_type === "person_from_faker") ? (
        <PersonDialog config={config} onUpdate={update} />
      ) : null;
    case "llm":
      return config.kind === "llm" ? (
        <LlmDialog
          config={config}
          modelConfigAliases={modelConfigAliases}
          modelProviderOptions={modelProviderOptions}
          toolProfileAliases={toolProfileAliases}
          onUpdate={update}
        />
      ) : null;
    case "model_provider":
      return config.kind === "model_provider" ? (
        <ModelProviderDialog config={config} onUpdate={update} />
      ) : null;
    case "model_config":
      return config.kind === "model_config" ? (
        <ModelConfigDialog
          config={config}
          providerOptions={modelProviderOptions}
          localProviderNames={localProviderNames}
          onUpdate={update}
        />
      ) : null;
    case "tool_config":
      return config.kind === "tool_config" ? (
        <ToolProfileDialog config={config} onUpdate={update} />
      ) : null;
    case "expression":
      return config.kind === "expression" ? (
        <ExpressionDialog config={config} onUpdate={update} />
      ) : null;
    case "validator":
      return config.kind === "validator" ? (
        <ValidatorDialog config={config} onUpdate={update} />
      ) : null;
    case "markdown_note":
      return config.kind === "markdown_note" ? (
        <MarkdownNoteDialog config={config} onUpdate={update} />
      ) : null;
  }
}
