// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

/** Colored primitive (string/number/bool/null) for the tree view. */
function Primitive({ value }: { value: unknown }) {
  if (typeof value === "string") {
    return (
      <span className="text-emerald-600 dark:text-emerald-400">"{value}"</span>
    );
  }
  if (typeof value === "number") {
    return <span className="text-amber-600 dark:text-amber-400">{value}</span>;
  }
  if (typeof value === "boolean") {
    return (
      <span className="text-purple-600 dark:text-purple-400">
        {String(value)}
      </span>
    );
  }
  if (value === null) {
    return <span className="text-muted-foreground">null</span>;
  }
  return <span>{String(value)}</span>;
}

/** Recursive, collapsible, read-only JSON node. */
function JsonNode({
  k,
  value,
  depth,
}: {
  k?: string;
  value: unknown;
  depth: number;
}) {
  const [open, setOpen] = useState(true);
  const isContainer = value !== null && typeof value === "object";
  const pad = { paddingLeft: `${depth * 12}px` };
  const keyLabel =
    k !== undefined ? (
      <span className="text-sky-600 dark:text-sky-400">{k}: </span>
    ) : null;

  if (!isContainer) {
    return (
      <div style={pad} className="whitespace-pre-wrap break-words py-0.5">
        {keyLabel}
        <Primitive value={value} />
      </div>
    );
  }

  const isArr = Array.isArray(value);
  const entries: [string, unknown][] = isArr
    ? (value as unknown[]).map((v, i) => [String(i), v])
    : Object.entries(value as Record<string, unknown>);
  const summary = isArr ? `[${entries.length}]` : `{${entries.length}}`;

  return (
    <div>
      <div
        style={pad}
        className="flex cursor-pointer select-none items-center gap-1 py-0.5"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="w-3 shrink-0 text-muted-foreground">
          {open ? "▾" : "▸"}
        </span>
        {keyLabel}
        <span className="text-muted-foreground">{summary}</span>
      </div>
      {open &&
        entries.map(([ck, cv]) => (
          <JsonNode key={ck} k={ck} value={cv} depth={depth + 1} />
        ))}
    </div>
  );
}

/**
 * JSON field with a "Raw" tab (editable textarea) and a "Tree" tab (pretty,
 * collapsible, read-only viewer). Value is the raw text; editing is done in Raw.
 */
export function JsonField({
  id,
  value,
  onChange,
}: {
  id?: string;
  value: string;
  onChange: (v: string) => void;
}) {
  const [tab, setTab] = useState<"raw" | "tree">("raw");

  let parsed: unknown;
  let parseError = false;
  const trimmed = value.trim();
  if (trimmed) {
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      parseError = true;
    }
  }

  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as "raw" | "tree")}>
      <TabsList className="h-8">
        <TabsTrigger value="raw" className="text-xs">
          Raw
        </TabsTrigger>
        <TabsTrigger value="tree" className="text-xs">
          Tree
        </TabsTrigger>
      </TabsList>

      <TabsContent value="raw" className="mt-2">
        <Textarea
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="min-h-24 max-h-64 overflow-auto font-mono text-xs"
          spellCheck={false}
        />
      </TabsContent>

      <TabsContent value="tree" className="mt-2">
        {!trimmed ? (
          <p className="text-xs text-muted-foreground">
            Nothing to show — paste JSON in the Raw tab.
          </p>
        ) : parseError ? (
          <p className="text-xs text-destructive">
            Invalid JSON — fix it in the Raw tab.
          </p>
        ) : (
          <div className="max-h-64 overflow-auto rounded-md bg-muted/40 p-2 font-mono text-xs leading-relaxed">
            <JsonNode value={parsed} depth={0} />
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}
