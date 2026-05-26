// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  previewSchemaComparators,
  type SchemaComparatorField,
} from "../api/eval-api";

/**
 * Live preview of the comparator each field resolves to for the json_document
 * metric's schema — so it's obvious a pasted JSON Schema derives numeric/date/
 * categorical/string comparators (not "just string").
 */
export function SchemaComparatorPreview({
  schemaText,
}: {
  schemaText: unknown;
}) {
  const [fields, setFields] = useState<SchemaComparatorField[]>([]);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    const text =
      typeof schemaText === "string"
        ? schemaText.trim()
        : schemaText
          ? JSON.stringify(schemaText)
          : "";
    if (!text) {
      setFields([]);
      setNote(null);
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      setFields([]);
      setNote("Enter valid JSON to preview comparators.");
      return;
    }
    let active = true;
    const id = setTimeout(() => {
      previewSchemaComparators(parsed)
        .then((f) => {
          if (!active) return;
          setFields(f);
          setNote(f.length ? null : "No scorable fields detected.");
        })
        .catch((e) => {
          if (!active) return;
          setFields([]);
          setNote(e instanceof Error ? e.message : String(e));
        });
    }, 400);
    return () => {
      active = false;
      clearTimeout(id);
    };
  }, [schemaText]);

  if (fields.length === 0 && !note) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">
        Derived comparators
      </span>
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
      {fields.length > 0 && (
        <div className="flex max-h-48 flex-col gap-1 overflow-auto rounded-md bg-muted/40 p-2">
          {fields.map((f) => (
            <div
              key={f.path}
              className="flex items-center justify-between gap-2 text-xs"
            >
              <span className="truncate font-mono text-muted-foreground">
                {f.path}
              </span>
              <Badge variant="secondary" className="shrink-0">
                {f.comparator}
              </Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
