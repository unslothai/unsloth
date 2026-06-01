// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import {
  type McpServerConfig,
  createMcpServer,
  listMcpServers,
  updateMcpServer,
} from "./api/mcp-servers-api";
import { ChatMcpServersDialog } from "./chat-mcp-servers-dialog";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

type McpPreset = { id: string; displayName: string; url: string };

// Keyless remote MCP presets (rate-limited free tiers, no API key).
// Hugging Face runs anonymously; add a token via "Add custom MCP".
const MCP_PRESETS: readonly McpPreset[] = [
  {
    id: "context7",
    displayName: "Context7",
    url: "https://mcp.context7.com/mcp",
  },
  { id: "exa", displayName: "Exa", url: "https://mcp.exa.ai/mcp" },
  {
    id: "huggingface",
    displayName: "Hugging Face",
    url: "https://huggingface.co/mcp",
  },
] as const;

// mcp_servers has no UNIQUE(url); dedupe by normalized URL so a preset
// toggle reuses its row instead of creating duplicates.
function normalizeMcpUrl(url: string): string {
  return (url || "").trim().toLowerCase().replace(/\/+$/, "");
}

export function McpComposerButton() {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );

  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [pendingUrl, setPendingUrl] = useState<string | null>(null);

  // mcp_enabled only applies on the local tool-capable send path; grey out otherwise.
  const usable = modelLoaded && supportsTools;

  const refresh = useCallback(async () => {
    try {
      setServers(await listMcpServers());
    } catch {
      // Keep prior state if the list call fails.
    }
  }, []);

  // Initial load so the pill reflects already-enabled servers (also refreshed on open).
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Keep the per-chat flag in step with whether any server is enabled.
  const reconcileFlag = useCallback(
    (rows: McpServerConfig[]) => {
      const anyEnabled = rows.some((s) => s.is_enabled);
      if (anyEnabled && !mcpEnabledForChat) setMcpEnabledForChat(true);
      else if (!anyEnabled && mcpEnabledForChat) setMcpEnabledForChat(false);
    },
    [mcpEnabledForChat, setMcpEnabledForChat],
  );

  const presetUrls = new Set(MCP_PRESETS.map((p) => normalizeMcpUrl(p.url)));
  const enabledUrls = new Set(
    servers.filter((s) => s.is_enabled).map((s) => normalizeMcpUrl(s.url)),
  );
  // Non-preset servers, shown below the presets so they stay toggleable.
  const customServers = servers.filter(
    (s) => !presetUrls.has(normalizeMcpUrl(s.url)),
  );
  const enabledCount = servers.filter((s) => s.is_enabled).length;
  const active = mcpEnabledForChat && enabledCount > 0;

  async function toggleServer(args: {
    url: string;
    displayName: string;
    checked: boolean;
    existing?: McpServerConfig;
  }) {
    const norm = normalizeMcpUrl(args.url);
    if (pendingUrl === norm) return; // guard rapid double-clicks
    setPendingUrl(norm);
    try {
      if (args.checked) {
        // Reuse a row matched by URL, else create one.
        const fresh = await listMcpServers();
        const match = fresh.find((s) => normalizeMcpUrl(s.url) === norm);
        if (match) {
          if (!match.is_enabled) {
            await updateMcpServer(match.id, { isEnabled: true });
          }
        } else {
          await createMcpServer({
            displayName: args.displayName,
            url: args.url,
            isEnabled: true,
          });
        }
        setMcpEnabledForChat(true);
      } else if (args.existing) {
        await updateMcpServer(args.existing.id, { isEnabled: false });
      }
      const rows = await listMcpServers();
      setServers(rows);
      reconcileFlag(rows);
    } catch (err) {
      toast.error("Failed to update MCP server", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setPendingUrl(null);
    }
  }

  return (
    <>
      {usable ? (
        <DropdownMenu
          open={menuOpen}
          onOpenChange={(open) => {
            setMenuOpen(open);
            if (open) void refresh();
          }}
        >
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              className="composer-pill-btn"
              data-active={active ? "true" : "false"}
              aria-label="MCP servers"
            >
              <HugeiconsIcon
                icon={McpServerIcon}
                className="size-3.5"
                strokeWidth={2}
              />
              <span>MCP</span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-60">
            <DropdownMenuLabel>MCP servers</DropdownMenuLabel>
            {MCP_PRESETS.map((preset) => {
              const norm = normalizeMcpUrl(preset.url);
              const existing = servers.find(
                (s) => normalizeMcpUrl(s.url) === norm,
              );
              return (
                <DropdownMenuCheckboxItem
                  key={preset.id}
                  checked={enabledUrls.has(norm)}
                  disabled={pendingUrl === norm}
                  onSelect={(e) => e.preventDefault()}
                  onCheckedChange={(checked) =>
                    void toggleServer({
                      url: preset.url,
                      displayName: preset.displayName,
                      checked,
                      existing,
                    })
                  }
                >
                  {preset.displayName}
                </DropdownMenuCheckboxItem>
              );
            })}
            {customServers.length > 0 ? <DropdownMenuSeparator /> : null}
            {customServers.map((server) => (
              <DropdownMenuCheckboxItem
                key={server.id}
                checked={server.is_enabled}
                disabled={pendingUrl === normalizeMcpUrl(server.url)}
                onSelect={(e) => e.preventDefault()}
                onCheckedChange={(checked) =>
                  void toggleServer({
                    url: server.url,
                    displayName: server.display_name,
                    checked,
                    existing: server,
                  })
                }
              >
                <span className="truncate">{server.display_name}</span>
              </DropdownMenuCheckboxItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => {
                setMenuOpen(false);
                setDialogOpen(true);
              }}
            >
              Add custom MCP
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            {/* Not disabled, so the tooltip still fires on hover. */}
            <button
              type="button"
              className="composer-pill-btn cursor-not-allowed opacity-40"
              data-active="false"
              aria-disabled={true}
              aria-label="MCP servers"
            >
              <HugeiconsIcon
                icon={McpServerIcon}
                className="size-3.5"
                strokeWidth={2}
              />
              <span>MCP</span>
            </button>
          </TooltipTrigger>
          <TooltipContent>
            MCP works with local tool-capable models
          </TooltipContent>
        </Tooltip>
      )}
      <ChatMcpServersDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          if (!next) {
            // Resync after managing servers.
            void listMcpServers().then((rows) => {
              setServers(rows);
              reconcileFlag(rows);
            });
          }
        }}
        openToCreate={true}
      />
    </>
  );
}
