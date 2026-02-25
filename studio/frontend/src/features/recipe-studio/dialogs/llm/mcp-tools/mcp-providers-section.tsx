import { Delete02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { LlmMcpProviderConfig, McpEnvVar } from "../../../types";
import { FieldLabel } from "../../shared/field-label";

type McpProvidersSectionProps = {
  providers: LlmMcpProviderConfig[];
  onAddProvider: () => void;
  onUpdateProviderAt: (
    index: number,
    patch: Partial<LlmMcpProviderConfig>,
  ) => void;
  onRemoveProvider: (index: number) => void;
  onAddProviderArg: (providerIndex: number) => void;
  onUpdateProviderArg: (
    providerIndex: number,
    argIndex: number,
    value: string,
  ) => void;
  onRemoveProviderArg: (providerIndex: number, argIndex: number) => void;
  onAddProviderEnv: (providerIndex: number) => void;
  onUpdateProviderEnv: (
    providerIndex: number,
    envIndex: number,
    patch: Partial<McpEnvVar>,
  ) => void;
  onRemoveProviderEnv: (providerIndex: number, envIndex: number) => void;
};

export function McpProvidersSection({
  providers,
  onAddProvider,
  onUpdateProviderAt,
  onRemoveProvider,
  onAddProviderArg,
  onUpdateProviderArg,
  onRemoveProviderArg,
  onAddProviderEnv,
  onUpdateProviderEnv,
  onRemoveProviderEnv,
}: McpProvidersSectionProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <FieldLabel
          label="MCP servers"
          hint="Server definitions used by tool configs for tool calls."
        />
        <Button type="button" size="xs" variant="outline" onClick={onAddProvider}>
          <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
          Add MCP server
        </Button>
      </div>

      {providers.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Add MCP servers to be referenced by tool config providers.
        </p>
      )}

      {providers.map((provider, providerIndex) => {
        const args = provider.args && provider.args.length > 0 ? provider.args : [""];
        const envVars =
          provider.env && provider.env.length > 0
            ? provider.env
            : [{ key: "", value: "" }];

        return (
          <div
            key={provider.id}
            className="space-y-3 border-b border-border/40 pb-4 last:border-b-0"
          >
            <div className="grid gap-2">
              <FieldLabel
                label="Name"
                hint="Unique provider name referenced by tool configs."
              />
              <Input
                value={provider.name}
                placeholder="MCP server name"
                onChange={(event) =>
                  onUpdateProviderAt(providerIndex, { name: event.target.value })
                }
              />
            </div>

            <Tabs
              value={provider.provider_type}
              onValueChange={(value) =>
                onUpdateProviderAt(providerIndex, {
                  // biome-ignore lint/style/useNamingConvention: ui schema
                  provider_type: value === "stdio" ? "stdio" : "streamable_http",
                })
              }
            >
              <TabsList className="w-full">
                <TabsTrigger value="stdio">STDIO</TabsTrigger>
                <TabsTrigger value="streamable_http">Streamable HTTP</TabsTrigger>
              </TabsList>
            </Tabs>

            {provider.provider_type === "stdio" ? (
              <div className="space-y-3">
                <div className="grid gap-2">
                  <FieldLabel
                    label="Command to launch"
                    hint="Executable used to start stdio MCP server."
                  />
                  <Input
                    value={provider.command ?? ""}
                    placeholder="npx"
                    onChange={(event) =>
                      onUpdateProviderAt(providerIndex, {
                        command: event.target.value,
                      })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <FieldLabel
                    label="Arguments"
                    hint="CLI args passed to MCP command."
                  />
                  {args.map((arg, argIndex) => (
                    <div key={`${provider.id}-arg-${argIndex}`} className="flex gap-2">
                      <Input
                        value={arg}
                        placeholder={argIndex === 0 ? "-y" : "argument"}
                        onChange={(event) =>
                          onUpdateProviderArg(providerIndex, argIndex, event.target.value)
                        }
                      />
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        onClick={() => onRemoveProviderArg(providerIndex, argIndex)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={() => onAddProviderArg(providerIndex)}
                  >
                    <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                  </Button>
                </div>

                <div className="space-y-2">
                  <FieldLabel
                    label="Environment variables"
                    hint="Key/value env vars for MCP process."
                  />
                  {envVars.map((item, envIndex) => (
                    <div
                      key={`${provider.id}-env-${envIndex}`}
                      className="grid grid-cols-[1fr_1fr_auto] gap-2"
                    >
                      <Input
                        value={item.key}
                        placeholder="Key"
                        onChange={(event) =>
                          onUpdateProviderEnv(providerIndex, envIndex, {
                            key: event.target.value,
                          })
                        }
                      />
                      <Input
                        value={item.value}
                        placeholder="Value"
                        onChange={(event) =>
                          onUpdateProviderEnv(providerIndex, envIndex, {
                            value: event.target.value,
                          })
                        }
                      />
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        onClick={() => onRemoveProviderEnv(providerIndex, envIndex)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={() => onAddProviderEnv(providerIndex)}
                  >
                    <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="grid gap-2">
                  <FieldLabel
                    label="Endpoint"
                    hint="Streamable HTTP MCP server URL."
                  />
                  <Input
                    value={provider.endpoint ?? ""}
                    placeholder="https://example.com/mcp"
                    onChange={(event) =>
                      onUpdateProviderAt(providerIndex, {
                        endpoint: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="grid gap-2">
                  <FieldLabel
                    label="API key env (optional)"
                    hint="Env var name for endpoint auth token."
                  />
                  <Input
                    value={provider.api_key_env ?? ""}
                    placeholder="MCP_API_KEY"
                    onChange={(event) =>
                      onUpdateProviderAt(providerIndex, {
                        // biome-ignore lint/style/useNamingConvention: api schema
                        api_key_env: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="grid gap-2">
                  <FieldLabel
                    label="API key (optional)"
                    hint="Inline token for endpoint auth."
                  />
                  <Input
                    value={provider.api_key ?? ""}
                    placeholder="api key"
                    onChange={(event) =>
                      onUpdateProviderAt(providerIndex, {
                        // biome-ignore lint/style/useNamingConvention: api schema
                        api_key: event.target.value,
                      })
                    }
                  />
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <Button
                type="button"
                size="xs"
                variant="ghost"
                onClick={() => onRemoveProvider(providerIndex)}
              >
                Remove
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
