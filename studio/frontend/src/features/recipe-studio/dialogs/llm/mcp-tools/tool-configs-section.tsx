import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChipInput } from "../../../components/chip-input";
import type { LlmToolConfig } from "../../../types";
import { addUnique, collectToolSuggestions } from "./helpers";
import { FieldLabel } from "../../shared/field-label";

type ToolConfigsSectionProps = {
  toolConfigs: LlmToolConfig[];
  providerNameSuggestions: string[];
  toolsByProvider: Record<string, string[]>;
  loadingTools: boolean;
  onFetchTools: () => void;
  onAddToolConfig: () => void;
  onUpdateToolConfig: (index: number, patch: Partial<LlmToolConfig>) => void;
  onRemoveToolConfig: (index: number) => void;
};

export function ToolConfigsSection({
  toolConfigs,
  providerNameSuggestions,
  toolsByProvider,
  loadingTools,
  onFetchTools,
  onAddToolConfig,
  onUpdateToolConfig,
  onRemoveToolConfig,
}: ToolConfigsSectionProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <FieldLabel
          label="Tool configs"
          hint="Map a tool alias to MCP providers and allowed tools."
        />
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="xs"
            variant="outline"
            disabled={loadingTools}
            onClick={onFetchTools}
          >
            {loadingTools ? "Loading..." : "Fetch MCP tools"}
          </Button>
          <Button type="button" size="xs" variant="outline" onClick={onAddToolConfig}>
            <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
            Add tool config
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Define aliases/providers here. Active alias is selected above.
      </p>
      {toolConfigs.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Add at least one tool config to map alias to providers.
        </p>
      )}
      {toolConfigs.map((toolConfig, index) => (
        <div
          key={toolConfig.id}
          className="space-y-3 border-b border-border/40 pb-4 last:border-b-0"
        >
          <div className="grid gap-2">
            <FieldLabel
              label="Tool alias"
              hint="Alias referenced by LLM column tool_alias."
            />
            <Input
              value={toolConfig.tool_alias}
              placeholder="context7_tools"
              onChange={(event) =>
                onUpdateToolConfig(index, {
                  // biome-ignore lint/style/useNamingConvention: api schema
                  tool_alias: event.target.value,
                })
              }
            />
          </div>

          <div className="grid gap-2">
            <FieldLabel
              label="Providers"
              hint="MCP provider names this alias can call."
            />
            <ChipInput
              values={toolConfig.providers}
              suggestions={providerNameSuggestions}
              onAdd={(value) =>
                onUpdateToolConfig(index, {
                  providers: addUnique(toolConfig.providers, value),
                })
              }
              onRemove={(providerIndex) =>
                onUpdateToolConfig(index, {
                  providers: toolConfig.providers.filter(
                    (_, currentIndex) => currentIndex !== providerIndex,
                  ),
                })
              }
              placeholder="Type provider name and press Enter"
            />
          </div>

          <div className="grid gap-2">
            <FieldLabel
              label="Allow tools (optional)"
              hint="Optional allowlist of tool names."
            />
            <ChipInput
              // biome-ignore lint/style/useNamingConvention: api schema
              values={toolConfig.allow_tools ?? []}
              suggestions={collectToolSuggestions(toolConfig.providers, toolsByProvider)}
              onAdd={(value) =>
                onUpdateToolConfig(index, {
                  // biome-ignore lint/style/useNamingConvention: api schema
                  allow_tools: addUnique(toolConfig.allow_tools ?? [], value),
                })
              }
              onRemove={(toolIndex) =>
                onUpdateToolConfig(index, {
                  // biome-ignore lint/style/useNamingConvention: api schema
                  allow_tools: (toolConfig.allow_tools ?? []).filter(
                    (_, currentIndex) => currentIndex !== toolIndex,
                  ),
                })
              }
              placeholder="Type tool name and press Enter"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="grid gap-2">
              <FieldLabel
                label="Max turns"
                hint="Max tool-calling turns before forcing completion."
              />
              <Input
                value={toolConfig.max_tool_call_turns ?? ""}
                onChange={(event) =>
                  onUpdateToolConfig(index, {
                    // biome-ignore lint/style/useNamingConvention: api schema
                    max_tool_call_turns: event.target.value,
                  })
                }
              />
            </div>
            <div className="grid gap-2">
              <FieldLabel
                label="Timeout sec"
                hint="Timeout per tool call."
              />
              <Input
                value={toolConfig.timeout_sec ?? ""}
                onChange={(event) =>
                  onUpdateToolConfig(index, {
                    // biome-ignore lint/style/useNamingConvention: api schema
                    timeout_sec: event.target.value,
                  })
                }
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button
              type="button"
              size="xs"
              variant="ghost"
              onClick={() => onRemoveToolConfig(index)}
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
