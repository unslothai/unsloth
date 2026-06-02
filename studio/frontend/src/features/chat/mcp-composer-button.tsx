// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { CheckIcon } from "lucide-react";
import { type FC, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import {
  DropdownMenu,
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

// Matches the Thinking pill chevron so the affordance reads the same.
const ArrowDownStandardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden={true}
  >
    <path d="M5.99977 9.00005L11.9998 15L17.9998 9" />
  </svg>
);

type McpPreset = {
  id: string;
  displayName: string; // stored row name
  url: string;
  label?: string; // dropdown text, if different from displayName
  hint?: string; // shown when the row is highlighted
  disablesWebSearch?: boolean; // turn the built-in Search pill off when enabled
};

// Keyless remote MCP presets (rate-limited free tiers, no API key).
// Hugging Face runs anonymously; add a token via "Add custom MCP".
const MCP_PRESETS: readonly McpPreset[] = [
  {
    id: "context7",
    displayName: "Context7",
    url: "https://mcp.context7.com/mcp",
    label: "Context7 (Realtime Docs)",
  },
  {
    id: "exa",
    displayName: "Exa",
    url: "https://mcp.exa.ai/mcp",
    label: "Exa (Semantic Search)",
    hint: "Enabling Exa will disable default search",
    disablesWebSearch: true,
  },
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

// Static, so it is not rebuilt on every render.
const PRESET_URLS = new Set(MCP_PRESETS.map((p) => normalizeMcpUrl(p.url)));

export function McpComposerButton() {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);

  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [pendingUrl, setPendingUrl] = useState<string | null>(null);
  const [hintKey, setHintKey] = useState<string | null>(null);

  // mcp_enabled only applies on the local tool-capable send path; grey out otherwise.
  const usable = modelLoaded && supportsTools;

  // Keep the per-chat flag in step with whether any server is enabled. Reads the
  // store directly so the callback stays stable (no refetch loop on mount).
  const reconcileFlag = useCallback(
    (rows: McpServerConfig[]) => {
      const anyEnabled = rows.some((s) => s.is_enabled);
      const current = useChatRuntimeStore.getState().mcpEnabledForChat;
      if (anyEnabled && !current) setMcpEnabledForChat(true);
      else if (!anyEnabled && current) setMcpEnabledForChat(false);
    },
    [setMcpEnabledForChat],
  );

  const refresh = useCallback(async () => {
    try {
      const rows = await listMcpServers();
      setServers(rows);
      reconcileFlag(rows);
    } catch {
      // Keep prior state if the list call fails.
    }
  }, [reconcileFlag]);

  // Initial load reconciles the pill with already-enabled servers (also on open).
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const enabledUrls = new Set(
    servers.filter((s) => s.is_enabled).map((s) => normalizeMcpUrl(s.url)),
  );
  // Non-preset servers, shown below the presets so they stay toggleable.
  const customServers = servers.filter(
    (s) => !PRESET_URLS.has(normalizeMcpUrl(s.url)),
  );
  const enabledCount = servers.filter((s) => s.is_enabled).length;
  const active = mcpEnabledForChat && enabledCount > 0;

  async function toggleServer(args: {
    url: string;
    displayName: string;
    checked: boolean;
    existing?: McpServerConfig;
    disablesWebSearch?: boolean;
  }) {
    const norm = normalizeMcpUrl(args.url);
    if (pendingUrl === norm) return; // guard rapid double-clicks
    setPendingUrl(norm);
    try {
      if (args.checked) {
        // Reuse the already-loaded row, else create one.
        if (args.existing) {
          if (!args.existing.is_enabled) {
            await updateMcpServer(args.existing.id, { isEnabled: true });
          }
        } else {
          await createMcpServer({
            displayName: args.displayName,
            url: args.url,
            isEnabled: true,
          });
        }
        setMcpEnabledForChat(true);
        // Exa is a search server; turn off the built-in Web Search to avoid overlap.
        if (args.disablesWebSearch) setToolsEnabled(false);
      } else if (args.existing) {
        await updateMcpServer(args.existing.id, { isEnabled: false });
      }
      await refresh();
    } catch (err) {
      toast.error("Failed to update MCP server", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setPendingUrl(null);
    }
  }

  // One dropdown row. Enabled rows get a green underlay and a tick that
  // becomes an X on hover so a click removes them. A hint shows as a tooltip
  // driven by row hover; the tooltip anchor is pointer-events-none so the whole
  // row stays clickable (a Radix TooltipTrigger would swallow the select).
  const renderRow = (opts: {
    key: string;
    label: string;
    url: string;
    displayName: string;
    enabled: boolean;
    existing?: McpServerConfig;
    hint?: string;
    disablesWebSearch?: boolean;
  }) => (
    <DropdownMenuItem
      key={opts.key}
      disabled={pendingUrl === normalizeMcpUrl(opts.url)}
      onSelect={(e) => {
        e.preventDefault();
        void toggleServer({
          url: opts.url,
          displayName: opts.displayName,
          checked: !opts.enabled,
          existing: opts.existing,
          disablesWebSearch: opts.disablesWebSearch,
        });
      }}
      onPointerEnter={opts.hint ? () => setHintKey(opts.key) : undefined}
      onPointerLeave={
        opts.hint
          ? () => setHintKey((k) => (k === opts.key ? null : k))
          : undefined
      }
      className={
        opts.enabled ? "relative text-primary font-medium" : "relative"
      }
    >
      <span className="truncate">{opts.label}</span>
      {opts.enabled ? <CheckIcon className="ml-auto" /> : null}
      {opts.hint ? (
        <Tooltip open={hintKey === opts.key}>
          <TooltipTrigger asChild={true}>
            <span
              aria-hidden={true}
              className="pointer-events-none absolute inset-y-0 right-0 w-0"
            />
          </TooltipTrigger>
          <TooltipContent side="right">{opts.hint}</TooltipContent>
        </Tooltip>
      ) : null}
    </DropdownMenuItem>
  );

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
              <ArrowDownStandardIcon className="size-[15px]" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="top"
            align="start"
            sideOffset={2}
            avoidCollisions={true}
            className="unsloth-plus-menu w-[232px]"
          >
            <DropdownMenuLabel>MCP Servers</DropdownMenuLabel>
            {MCP_PRESETS.map((preset) => {
              const norm = normalizeMcpUrl(preset.url);
              return renderRow({
                key: preset.id,
                label: preset.label ?? preset.displayName,
                url: preset.url,
                displayName: preset.displayName,
                enabled: enabledUrls.has(norm),
                existing: servers.find((s) => normalizeMcpUrl(s.url) === norm),
                hint: preset.hint,
                disablesWebSearch: preset.disablesWebSearch,
              });
            })}
            {customServers.length > 0 ? <DropdownMenuSeparator /> : null}
            {customServers.map((server) =>
              renderRow({
                key: server.id,
                label: server.display_name,
                url: server.url,
                displayName: server.display_name,
                enabled: server.is_enabled,
                existing: server,
              }),
            )}
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
          // Resync after managing servers.
          if (!next) void refresh();
        }}
        openToCreate={true}
      />
    </>
  );
}
