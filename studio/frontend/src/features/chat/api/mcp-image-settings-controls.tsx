// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
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
import { authFetch } from "@/features/auth";

import { eligibleImageFields } from "./mcp-image-mapping-options";
import type { McpImageInputMapping } from "./mcp-servers-api";

type MappingOption = { tool: string; field: string };

export function McpImageMappingSettings({
  serverId,
  value,
  onChange,
  disabled = false,
}: {
  serverId?: string;
  value: McpImageInputMapping[];
  onChange: (value: McpImageInputMapping[]) => void;
  disabled?: boolean;
}) {
  const [options, setOptions] = useState<MappingOption[] | null>(null);
  const [choice, setChoice] = useState("");
  const [encoding, setEncoding] =
    useState<McpImageInputMapping["encoding"]>("base64");
  const [discovering, setDiscovering] = useState(false);

  useEffect(() => {
    setOptions(null);
    setChoice("");
  }, [serverId]);

  async function discoverFields() {
    if (!serverId) return;
    setDiscovering(true);
    try {
      const response = await authFetch(
        `/api/mcp/servers/${encodeURIComponent(serverId)}/tools`,
      );
      if (!response.ok) throw new Error("Could not discover tool schemas.");
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
        "Could not discover image fields. Refresh the server and try again.",
      );
    } finally {
      setDiscovering(false);
    }
  }

  const selected = choice === "" ? undefined : options?.[Number(choice)];

  return (
    <div className="space-y-3 border-t pt-4">
      <div className="space-y-1">
        <div className="text-sm font-medium">Image input mappings</div>
        <p className="max-w-[70ch] text-xs text-muted-foreground">
          Choose the exact top-level string field that receives one approved
          image. Studio validates each mapping against the tool schema.
        </p>
      </div>

      {value.length > 0 && (
        <div className="space-y-2">
          {value.map((mapping) => {
            const unavailable =
              options !== null &&
              !options.some(
                (option) =>
                  option.tool === mapping.tool &&
                  option.field === mapping.field,
              );
            return (
              <div
                key={mapping.tool}
                className="flex items-center justify-between gap-3 rounded-[14px] bg-muted/60 px-3 py-2"
              >
                <div className="min-w-0 text-xs">
                  <div className="truncate font-medium">
                    {mapping.tool} / {mapping.field}
                  </div>
                  <div className="text-muted-foreground">
                    {mapping.encoding === "data_url"
                      ? "Image data URL"
                      : "Raw base64"}
                    {unavailable ? (
                      <span className="text-destructive">
                        {" "}
                        · Field unavailable
                      </span>
                    ) : null}
                  </div>
                </div>
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  disabled={disabled}
                  onClick={() =>
                    onChange(value.filter((item) => item.tool !== mapping.tool))
                  }
                >
                  Remove
                </Button>
              </div>
            );
          })}
        </div>
      )}

      {serverId ? (
        <>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={disabled || discovering}
            onClick={discoverFields}
          >
            {discovering ? <Spinner /> : null}
            Discover fields
          </Button>

          {options !== null && (
            <div className="flex flex-wrap items-end gap-2">
              <Select
                value={choice}
                onValueChange={setChoice}
                disabled={disabled}
              >
                <SelectTrigger
                  className="min-w-64 flex-1"
                  aria-label="Tool and image field"
                >
                  <SelectValue placeholder="Select tool and string field" />
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
              <Select
                value={encoding}
                onValueChange={(next) =>
                  setEncoding(next as McpImageInputMapping["encoding"])
                }
                disabled={disabled}
              >
                <SelectTrigger className="min-w-40" aria-label="Image encoding">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="base64">Raw base64</SelectItem>
                  <SelectItem value="data_url">Image data URL</SelectItem>
                </SelectContent>
              </Select>
              <Button
                type="button"
                size="sm"
                disabled={disabled || !selected}
                onClick={() => {
                  if (!selected) return;
                  onChange([
                    ...value.filter(
                      (mapping) => mapping.tool !== selected.tool,
                    ),
                    { ...selected, encoding },
                  ]);
                  setChoice("");
                }}
              >
                Add mapping
              </Button>
            </div>
          )}

          {options?.length === 0 && (
            <p className="text-xs text-muted-foreground">
              No supported top-level string fields were found.
            </p>
          )}
        </>
      ) : (
        <p className="text-xs text-muted-foreground">
          Add this server first, then edit it to discover and map its tool
          fields.
        </p>
      )}
    </div>
  );
}
