// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";

import { imageFieldCandidates } from "./api/mcp-image";
import {
  type McpImageInputMapping,
  listMcpServerTools,
  refreshMcpServerTools,
} from "./api/mcp-servers-api";

type Option = { tool: string; field: string };

export function McpImageMappings({
  serverId,
  value,
  onChange,
  disabled,
  connectionUnsaved,
}: {
  serverId?: string;
  value: McpImageInputMapping[];
  onChange: (value: McpImageInputMapping[]) => void;
  disabled: boolean;
  connectionUnsaved: boolean;
}) {
  const [enabled, setEnabled] = useState(value.length > 0);
  const [options, setOptions] = useState<Option[] | null>(null);
  const [loading, setLoading] = useState(false);

  async function discover(id: string) {
    setLoading(true);
    try {
      const probe = await refreshMcpServerTools(id);
      if (!probe.ok) {
        throw new Error(probe.error ?? "The server did not respond.");
      }
      const tools = await listMcpServerTools(id);
      setOptions(
        tools.flatMap((tool) =>
          imageFieldCandidates(tool.inputSchema).map((field) => ({
            tool: tool.name,
            field,
          })),
        ),
      );
    } catch (err) {
      toast.error("Could not list this server's tools", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3 rounded-[14px] bg-muted/50 p-3">
      <label
        htmlFor="mcp-image-attachments"
        className="flex cursor-pointer items-start justify-between gap-3"
      >
        <span className="space-y-0.5">
          <span className="block text-sm font-medium">
            Send attached images to tools
          </span>
          <span className="block max-w-[70ch] text-xs text-muted-foreground">
            Images you attach go to the mapped tool field instead of the model,
            and only after you approve each call.
          </span>
        </span>
        <Switch
          id="mcp-image-attachments"
          checked={enabled}
          disabled={disabled}
          onCheckedChange={(next) => {
            setEnabled(next);
            if (!next) onChange([]);
          }}
        />
      </label>
      {enabled && !serverId ? (
        <p className="text-xs text-muted-foreground">
          Save this server first, then edit it to choose the image field.
        </p>
      ) : null}
      {enabled && serverId ? (
        <div className="space-y-2">
          {value.map((mapping) => (
            <div
              key={mapping.tool}
              className="flex items-center justify-between gap-3 text-xs"
            >
              <span className="min-w-0 flex-1 truncate">
                {mapping.tool} / {mapping.field}
              </span>
              <Select
                value={mapping.encoding}
                disabled={disabled}
                onValueChange={(next) =>
                  onChange(
                    value.map((m) =>
                      m.tool === mapping.tool
                        ? {
                            ...m,
                            encoding: next as McpImageInputMapping["encoding"],
                          }
                        : m,
                    ),
                  )
                }
              >
                <SelectTrigger
                  className="h-7 w-32"
                  aria-label={`Image encoding for ${mapping.tool}`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="base64">Raw base64</SelectItem>
                  <SelectItem value="data_url">Data URL</SelectItem>
                </SelectContent>
              </Select>
              <Button
                type="button"
                size="xs"
                variant="ghost"
                disabled={disabled}
                onClick={() =>
                  onChange(value.filter((m) => m.tool !== mapping.tool))
                }
              >
                Remove
              </Button>
            </div>
          ))}
          {connectionUnsaved ? (
            <p className="text-xs text-muted-foreground">
              Save the connection changes first, then choose the image field.
            </p>
          ) : options === null ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={disabled || loading}
              onClick={() => void discover(serverId)}
            >
              {loading ? <Spinner /> : null}
              Choose image field
            </Button>
          ) : options.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No tool on this server takes a top-level string field.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              <Select
                value=""
                disabled={disabled}
                onValueChange={(index) => {
                  const option = options[Number(index)];
                  if (!option) return;
                  onChange([
                    ...value.filter((m) => m.tool !== option.tool),
                    { ...option, encoding: "base64" },
                  ]);
                }}
              >
                <SelectTrigger
                  className="min-w-56 flex-1"
                  aria-label="Image field"
                >
                  <SelectValue placeholder="Add tool / field" />
                </SelectTrigger>
                <SelectContent>
                  {options.map((option, index) => (
                    <SelectItem
                      key={`${option.tool}:${option.field}`}
                      value={String(index)}
                    >
                      {option.tool} / {option.field}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
