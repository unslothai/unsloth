// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tick02Icon } from "@/lib/tick-icon";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChevronDownIcon,
  XIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
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
import { useShortcut } from "@/features/settings";

import { subscribeToMcpServerMutationSettlements } from "./api/mcp-server-mutation-tracker";
import {
  type McpServerConfig,
  createMcpServer,
  listMcpServers,
  updateMcpServer,
} from "./api/mcp-servers-api";
import { ChatMcpServersDialog } from "./chat-mcp-servers-dialog";
import { useChatActive } from "./runtime-provider";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import { useMcpServersDialogStore } from "./stores/mcp-servers-dialog-store";

// Matches the Thinking pill chevron so the affordance reads the same.
type McpPreset = {
  id: string;
  displayName: string; // stored row name
  url: string;
  label?: string; // dropdown text, if different from displayName
  hint?: string; // shown when the row is highlighted
  disablesWebSearch?: boolean; // turn the built-in Search pill off when enabled
};

// Keyless remote MCP presets (rate-limited free tiers, no API key). Hugging Face runs
// anonymously; add a token via "Manage MCP servers".
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
    id: "huggingface",
    displayName: "Hugging Face",
    url: "https://huggingface.co/mcp",
  },
] as const;

// mcp_servers has no UNIQUE(url); dedupe by normalized URL so a preset toggle reuses its row instead of duplicating.
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
  const dialogOpen = useMcpServersDialogStore((s) => s.open);
  const setDialogOpen = useMcpServersDialogStore((s) => s.setOpen);
  const [menuOpen, setMenuOpen] = useState(false);
  const [serversLoaded, setServersLoaded] = useState(false);
  const [pendingUrls, setPendingUrls] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const pendingUrlsRef = useRef(new Set<string>());
  const [hintKey, setHintKey] = useState<string | null>(null);
  const listRefreshGenerationRef = useRef(0);
  const hasLoadedServerSnapshotRef = useRef(false);

  // Grey out only when a loaded model lacks tool support; with no model yet, MCP can still be
  // pre-selected, like the other composer tools.
  const usable = !modelLoaded || supportsTools;

  const refresh = useCallback(
    async (waitForPendingMutations = true, minimumMutationEpoch = 0) => {
      const generation = listRefreshGenerationRef.current + 1;
      listRefreshGenerationRef.current = generation;
      setServersLoaded(false);
      try {
        const rows = await listMcpServers({
          waitForPendingMutations,
          minimumMutationEpoch,
        });
        if (listRefreshGenerationRef.current !== generation) return;
        setServers(rows);
        hasLoadedServerSnapshotRef.current = true;
        setServersLoaded(true);
      } catch {
        if (
          listRefreshGenerationRef.current === generation &&
          hasLoadedServerSnapshotRef.current
        ) {
          setServersLoaded(true);
        }
      }
    },
    [],
  );

  const applyServer = useCallback((server: McpServerConfig) => {
    setServers((current) => {
      const index = current.findIndex(
        (candidate) => candidate.id === server.id,
      );
      if (index === -1) return [...current, server];
      return current.map((candidate) =>
        candidate.id === server.id ? server : candidate,
      );
    });
  }, []);

  useEffect(() => {
    const unsubscribe = subscribeToMcpServerMutationSettlements((epoch) => {
      void refresh(false, epoch);
    });
    return () => {
      unsubscribe();
      listRefreshGenerationRef.current += 1;
    };
  }, [refresh]);

  // Load the server list on mount, and again when the dialog closes: it can be opened from the
  // chord as well as from this menu.
  useEffect(() => {
    if (dialogOpen) return;
    let cancelled = false;
    queueMicrotask(() => {
      if (!cancelled) void refresh();
    });
    return () => {
      cancelled = true;
    };
  }, [refresh, dialogOpen]);

  const enabledUrls = new Set(
    servers.filter((s) => s.is_enabled).map((s) => normalizeMcpUrl(s.url)),
  );
  // Non-preset servers, shown below the presets so they stay toggleable.
  const customServers = servers.filter(
    (s) => !PRESET_URLS.has(normalizeMcpUrl(s.url)),
  );
  const enabledCount = servers.filter((s) => s.is_enabled).length;
  const active = usable && mcpEnabledForChat && enabledCount > 0;

  async function toggleServer(args: {
    url: string;
    displayName: string;
    checked: boolean;
    existing?: McpServerConfig;
    disablesWebSearch?: boolean;
  }) {
    const norm = normalizeMcpUrl(args.url);
    if (pendingUrlsRef.current.has(norm)) return; // guard rapid double-clicks
    pendingUrlsRef.current.add(norm);
    setPendingUrls(new Set(pendingUrlsRef.current));
    try {
      if (args.checked) {
        // Reuse the already-loaded row, else create one.
        if (args.existing) {
          if (!args.existing.is_enabled) {
            applyServer(
              await updateMcpServer(args.existing.id, { isEnabled: true }),
            );
          }
        } else {
          applyServer(
            await createMcpServer({
              displayName: args.displayName,
              url: args.url,
              isEnabled: true,
            }),
          );
        }
        setMcpEnabledForChat(true);
        // Search servers turn off the built-in Web Search to avoid overlap.
        if (args.disablesWebSearch) setToolsEnabled(false);
      } else if (args.existing) {
        applyServer(
          await updateMcpServer(args.existing.id, { isEnabled: false }),
        );
      }
    } catch (err) {
      toast.error("Failed to update MCP server", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      pendingUrlsRef.current.delete(norm);
      setPendingUrls(new Set(pendingUrlsRef.current));
    }
  }

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
      // Server configuration remains available when the loaded model lacks tools.
      disabled={!serversLoaded || pendingUrls.has(normalizeMcpUrl(opts.url))}
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
              // pointer-events-none so the anchor cannot swallow row clicks.
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
            className={`composer-pill-btn ${usable ? "" : "opacity-40"}`}
            data-pill-label="MCP"
            data-active={active ? "true" : "false"}
            aria-label={
              usable
                ? "MCP servers"
                : "MCP servers, unavailable for the loaded model"
            }
          >
            {/* Outside compact mode, the hover X disables MCP without opening the menu. */}
            <span
              role="button"
              aria-label="Turn off MCP"
              tabIndex={-1}
              onPointerDown={(e) => {
                if (e.currentTarget.closest('[data-pill-compact="true"]'))
                  return;
                e.stopPropagation();
              }}
              onClick={(e) => {
                if (e.currentTarget.closest('[data-pill-compact="true"]'))
                  return;
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
            <ChevronDownIcon strokeWidth={1.5} className="composer-pill-caret size-[15px]" />
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
          {usable ? null : (
            <DropdownMenuLabel className="text-muted-foreground text-xs font-normal">
              The loaded model cannot use MCP tools
            </DropdownMenuLabel>
          )}
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
    </>
  );
}

/** Mount the dialog independently so its shortcut works while MCP is disabled. */
export function McpServersDialogMount() {
  const open = useMcpServersDialogStore((s) => s.open);
  const setOpen = useMcpServersDialogStore((s) => s.setOpen);
  const chatActive = useChatActive();
  useShortcut("openMcpServers", () => setOpen(true), { enabled: chatActive });
  useEffect(() => {
    if (!chatActive && open) setOpen(false);
  }, [chatActive, open, setOpen]);
  // Also clear it when logout or expiry unmounts this subtree directly.
  useEffect(() => {
    return () => useMcpServersDialogStore.getState().setOpen(false);
  }, []);
  return (
    <ChatMcpServersDialog open={chatActive && open} onOpenChange={setOpen} />
  );
}
