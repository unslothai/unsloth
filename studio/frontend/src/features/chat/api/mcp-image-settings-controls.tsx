// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { getChatSettings, saveChatSettingsPatch } from "./chat-settings-api";
import { eligibleImageFields } from "./mcp-image-mapping-options";
import {
  beginMcpImageSettingRefresh,
  beginMcpImageSettingSave,
  canApplyMcpImageSettingRefresh,
  canApplyMcpImageSettingSave,
  finishMcpImageSettingSave,
  MCP_IMAGE_SETTING_SAVED_EVENT,
} from "./mcp-image-setting-order";
import { clearSelectedMcpImage } from "./mcp-image-selection";
import {
  type McpImageInputMapping,
  type McpServerConfig,
  updateMcpServer,
} from "./mcp-servers-api";

export function McpImageSharingSetting() {
  const enabled = useChatRuntimeStore((s) => s.mcpImageAttachmentsEnabled);
  const [busy, setBusy] = useState(true);
  useEffect(() => {
    let active = true;
    const refresh = () => {
      const requestId = beginMcpImageSettingRefresh();
      if (requestId === null) return;
      void getChatSettings()
        .then((settings) => {
          if (active && canApplyMcpImageSettingRefresh(requestId)) {
            const savedEnabled = settings.mcpImageAttachmentsEnabled === true;
            useChatRuntimeStore.setState({
              mcpImageAttachmentsEnabled: savedEnabled,
            });
            if (!savedEnabled) {
              clearSelectedMcpImage();
            }
          }
        })
        .catch(() => {
          if (active && canApplyMcpImageSettingRefresh(requestId)) {
            useChatRuntimeStore.setState({
              mcpImageAttachmentsEnabled: false,
            });
            clearSelectedMcpImage();
          }
        })
        .finally(() => {
          if (active && canApplyMcpImageSettingRefresh(requestId)) {
            setBusy(false);
          }
        });
    };
    refresh();
    window.addEventListener("focus", refresh);
    window.addEventListener(MCP_IMAGE_SETTING_SAVED_EVENT, refresh);
    return () => {
      active = false;
      window.removeEventListener("focus", refresh);
      window.removeEventListener(MCP_IMAGE_SETTING_SAVED_EVENT, refresh);
    };
  }, []);
  return (
    <div className="space-y-2 rounded-md border p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        Allow tool-only image attachments
        <Switch
          aria-label="Allow tool-only image attachments"
          checked={enabled}
          disabled={busy}
          onCheckedChange={async (next) => {
            const requestId = beginMcpImageSettingSave();
            setBusy(true);
            try {
              const saved = await saveChatSettingsPatch({
                mcpImageAttachmentsEnabled: next,
              });
              if (!canApplyMcpImageSettingSave(requestId)) return;
              const savedEnabled = saved.mcpImageAttachmentsEnabled === true;
              useChatRuntimeStore.setState({
                mcpImageAttachmentsEnabled: savedEnabled,
              });
              if (!savedEnabled) {
                clearSelectedMcpImage();
              }
            } catch {
              if (canApplyMcpImageSettingSave(requestId)) {
                toast.error("Could not save image sharing setting. Try again.");
              }
            } finally {
              if (finishMcpImageSettingSave(requestId)) {
                setBusy(false);
                window.dispatchEvent(new Event(MCP_IMAGE_SETTING_SAVED_EVENT));
              }
            }
          }}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Off by default. Choose one tool-only image in the composer, then approve
        each disclosure separately. MCP must also be enabled.
      </p>
    </div>
  );
}

export function McpImageMappingSettings({
  server,
}: {
  server: McpServerConfig;
}) {
  const [options, setOptions] = useState<
    { tool: string; field: string }[] | null
  >(null);
  const [mappings, setMappings] = useState(server.image_input_mappings ?? []);
  const [choice, setChoice] = useState("");
  const [encoding, setEncoding] =
    useState<McpImageInputMapping["encoding"]>("base64");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setMappings(server.image_input_mappings ?? []);
    setOptions(null);
    setChoice("");
  }, [server]);

  async function save(next: McpImageInputMapping[]) {
    setBusy(true);
    try {
      const saved = await updateMcpServer(server.id, {
        imageInputMappings: next,
      });
      setMappings(saved.image_input_mappings ?? []);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Could not save image mapping.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="mt-2 text-xs">
      <summary>Image input mappings</summary>
      <div className="space-y-2 py-2">
        {mappings.map((mapping) => (
          <div key={mapping.tool} className="break-all">
            {mapping.tool}.{mapping.field} ({mapping.encoding})
            {options &&
              !options.some(
                (option) =>
                  option.tool === mapping.tool &&
                  option.field === mapping.field,
              ) && (
                <span className="text-destructive">
                  {" "}
                  Unavailable in discovered schema
                </span>
              )}
            <Button
              size="xs"
              variant="ghost"
              disabled={busy}
              onClick={() =>
                void save(mappings.filter((item) => item.tool !== mapping.tool))
              }
            >
              Remove
            </Button>
          </div>
        ))}
        <Button
          size="xs"
          variant="outline"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              const response = await authFetch(
                `/api/mcp/servers/${encodeURIComponent(server.id)}/tools`,
              );
              if (!response.ok) {
                throw new Error("Could not discover tool schemas.");
              }
              const data = await response.json();
              const tools = Array.isArray(data) ? data : data.tools;
              setOptions(
                (
                  tools as {
                    name: string;
                    inputSchema?: unknown;
                  }[]
                ).flatMap((tool) =>
                  eligibleImageFields(tool.inputSchema).map((field) => ({
                    tool: tool.name,
                    field,
                  })),
                ),
              );
            } catch {
              toast.error(
                "Could not discover image mapping fields. Refresh and try again.",
              );
            } finally {
              setBusy(false);
            }
          }}
        >
          Discover fields
        </Button>
        {options && (
          <>
            <select
              aria-label="Tool and image field"
              value={choice}
              onChange={(event) => setChoice(event.target.value)}
              className="w-full rounded border bg-background p-1"
            >
              <option value="">Select raw tool and string field</option>
              {options.map((option, index) => (
                <option
                  key={`${option.tool}:${option.field}`}
                  value={String(index)}
                >
                  {option.tool} / {option.field}
                </option>
              ))}
            </select>
            <select
              aria-label="Image encoding"
              value={encoding}
              onChange={(event) =>
                setEncoding(
                  event.target.value as McpImageInputMapping["encoding"],
                )
              }
              className="rounded border bg-background p-1"
            >
              <option value="base64">Base64</option>
              <option value="data_url">Data URL</option>
            </select>
            <Button
              size="xs"
              disabled={busy || choice === ""}
              onClick={() => {
                const selected = options[Number(choice)];
                if (selected) {
                  void save([
                    ...mappings.filter(
                      (mapping) => mapping.tool !== selected.tool,
                    ),
                    { ...selected, encoding },
                  ]);
                }
              }}
            >
              Save mapping
            </Button>
            {options.length === 0 && (
              <p>No supported top-level string fields discovered.</p>
            )}
          </>
        )}
      </div>
    </details>
  );
}
