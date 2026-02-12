import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { type ReactElement, useRef } from "react";
import type { LlmConfig } from "../../types";
import { LlmGeneralTab } from "./general-tab";
import { LlmMcpToolsTab } from "./mcp-tools-tab";
import { LlmScoresTab } from "./scores-tab";

type LlmDialogProps = {
  config: LlmConfig;
  modelConfigAliases: string[];
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmDialog({
  config,
  modelConfigAliases,
  onUpdate,
}: LlmDialogProps): ReactElement {
  const modelAliasAnchorRef = useRef<HTMLDivElement>(null);

  return (
    <Tabs defaultValue="general" className="w-full">
      <TabsList className="w-full">
        <TabsTrigger value="general">General</TabsTrigger>
        {config.llm_type === "judge" && <TabsTrigger value="scores">Scores</TabsTrigger>}
        <TabsTrigger value="tools">MCPs / Tools</TabsTrigger>
      </TabsList>
      <TabsContent value="general" className="pt-3">
        <LlmGeneralTab
          config={config}
          modelConfigAliases={modelConfigAliases}
          modelAliasAnchorRef={modelAliasAnchorRef}
          onUpdate={onUpdate}
        />
      </TabsContent>
      {config.llm_type === "judge" && (
        <TabsContent value="scores" className="pt-3">
          <LlmScoresTab config={config} onUpdate={onUpdate} />
        </TabsContent>
      )}
      <TabsContent value="tools" className="pt-3">
        <LlmMcpToolsTab config={config} onUpdate={onUpdate} />
      </TabsContent>
    </Tabs>
  );
}
