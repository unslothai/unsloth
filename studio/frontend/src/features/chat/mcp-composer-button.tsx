// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tick02Icon } from "@/lib/tick-icon";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { XIcon } from "lucide-react";
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
// Hugging Face runs anonymously; add a token via "Manage MCP servers".
const MCP_PRESETS: readonly McpPreset[] = [
  {
    id: "unsloth-docs",
    displayName: "Unsloth Docs",
    url: "https://unsloth.ai/docs/~gitbook/mcp",
  },
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

// mcp_servers has no UNIQUE(url); dedupe by normalized URL so a preset toggle
// reuses its row instead of duplicating.
function normalizeMcpUrl(url: string): string {
  return (url || "").trim().toLowerCase().replace(/\/+$/, "");
}

// Static, so it is not rebuilt on every render.
const PRESET_URLS = new Set(MCP_PRESETS.map((p) => normalizeMcpUrl(p.url)));

export function McpComposerButton({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
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

  // Grey out only when a loaded model lacks tool support; with no model yet,
  // MCP can still be pre-selected, like the other composer tools.
  const usable = !modelLoaded || supportsTools;

  const refresh = useCallback(async () => {
    try {
      const rows = await listMcpServers();
      setServers(rows);
    } catch {
      // Keep prior state if the list call fails.
    }
  }, []);

  // Load the server list on mount and whenever the menu opens.
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

  // One dropdown row. Enabled rows get a green underlay and a tick that becomes
  // an X on hover (click removes). The hint tooltip anchor is pointer-events-none
  // so the row stays clickable (a Radix TooltipTrigger would swallow the select).
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
      {opts.enabled ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
      ) : null}
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
              data-pill-label="MCP"
              data-active={active ? "true" : "false"}
              aria-label="MCP servers"
            >
              {/* Icon doubles as an off switch: hover swaps to an X; clicking
                  it turns MCP off without opening the menu. In compact
                  icon-only mode the glyph is the whole button, so clicks fall
                  through to the trigger and open the menu instead. */}
              <span
                role="button"
                aria-label="Turn off MCP"
                tabIndex={-1}
                onPointerDown={(e) => {
                  if (e.currentTarget.closest('[data-pill-compact="true"]')) return;
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  if (e.currentTarget.closest('[data-pill-compact="true"]')) return;
                  e.stopPropagation();
                  setMcpEnabledForChat(false);
                }}
                className="composer-pill-glyph cursor-pointer"
              >
                <HugeiconsIcon
                  icon={McpServerIcon}
                  className="size-[15px]"
                  strokeWidth={2}
                />
                <XIcon className="composer-pill-x" />
              </span>
              <span>MCP</span>
              <ArrowDownStandardIcon className="composer-pill-caret size-[15px]" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side={side}
            align="start"
            sideOffset={0}
            avoidCollisions={true}
            className="unsloth-plus-menu mcp-menu w-[232px]"
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
              Manage MCP servers
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
                className="size-[15px]"
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
      />
    </>
  );
}
