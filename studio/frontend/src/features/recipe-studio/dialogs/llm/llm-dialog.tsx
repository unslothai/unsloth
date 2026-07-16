// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useRef } from "react";
import type { LlmConfig } from "../../types";
import { LlmGeneralTab } from "./general-tab";
import { LlmScoresTab } from "./scores-tab";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

type LlmDialogProps = {
  config: LlmConfig;
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  toolProfileAliases: string[];
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmDialog({
  config,
  modelConfigAliases,
  modelProviderOptions,
  toolProfileAliases,
  onUpdate,
}: LlmDialogProps): ReactElement {
  const modelAliasAnchorRef = useRef<HTMLDivElement>(null);

  if (config.llm_type !== "judge") {
    return (
      <LlmGeneralTab
        config={config}
        modelConfigAliases={modelConfigAliases}
        modelProviderOptions={modelProviderOptions}
        toolProfileAliases={toolProfileAliases}
        modelAliasAnchorRef={modelAliasAnchorRef}
        onUpdate={onUpdate}
      />
    );
  }

  return (
    <Tabs defaultValue="general" className="w-full">
      <TabsList className="w-full">
        <TabsTrigger value="general">General</TabsTrigger>
        {config.llm_type === "judge" && <TabsTrigger value="scores">Scores</TabsTrigger>}
      </TabsList>
      <TabsContent value="general" className="pt-3">
        <LlmGeneralTab
          config={config}
          modelConfigAliases={modelConfigAliases}
          modelProviderOptions={modelProviderOptions}
          toolProfileAliases={toolProfileAliases}
          modelAliasAnchorRef={modelAliasAnchorRef}
          onUpdate={onUpdate}
        />
      </TabsContent>
      {config.llm_type === "judge" && (
        <TabsContent value="scores" className="pt-3">
          <LlmScoresTab config={config} onUpdate={onUpdate} />
        </TabsContent>
      )}
    </Tabs>
  );
}
