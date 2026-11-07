// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import type { Lorebook, LorebookEntry, LorebookPosition } from "../types";
import { parseKeyList } from "../prompt/lorebook";

const PAGE = 25;

export function emptyLorebook(): Lorebook {
  return { name: "", scan_depth: 4, token_budget: 1024, recursive_scanning: false, extensions: {}, entries: [] };
}

function newEntry(entries: LorebookEntry[]): LorebookEntry {
  const numeric = entries.map((entry) => Number(entry.id)).filter(Number.isFinite);
  return {
    id: numeric.length ? Math.max(...numeric) + 1 : 0,
    keys: [],
    secondary_keys: [],
    content: "",
    comment: "",
    enabled: true,
    constant: false,
    selective: false,
    case_sensitive: false,
    insertion_order: 100,
    position: "before_char",
    depth: 4,
    selective_logic: "and_any",
    extensions: {},
  };
}

export function LorebookEditor({ book, onChange }: { book: Lorebook | null; onChange: (book: Lorebook | null) => void }) {
  const t = useT();
  const value = book ?? emptyLorebook();
  const [open, setOpen] = useState<Set<string | number>>(new Set());
  const [limit, setLimit] = useState(PAGE);

  function update(index: number, patch: Partial<LorebookEntry>) {
    const entries = value.entries.map((entry, i) => (i === index ? { ...entry, ...patch } : entry));
    onChange({ ...value, entries });
  }

  function add() {
    const entry = newEntry(value.entries);
    onChange({ ...value, entries: [entry, ...value.entries] });
    setOpen((current) => new Set(current).add(entry.id));
  }

  function remove(index: number) {
    const entries = value.entries.filter((_, i) => i !== index);
    onChange(entries.length || book?.name ? { ...value, entries } : null);
  }

  const shown = value.entries.slice(0, limit);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <p className="min-w-0 flex-1 text-ui-13 text-muted-foreground">{t("characters.lorebook.hint")}</p>
        <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
          {t("characters.lorebook.scanDepth")}
          <Input
            type="number"
            min={1}
            max={100}
            value={value.scan_depth}
            className="h-8 w-24"
            onChange={(event) => onChange({ ...value, scan_depth: Math.max(1, Number(event.target.value) || 1) })}
          />
        </label>
        <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
          {t("characters.lorebook.budget")}
          <Input
            type="number"
            min={64}
            step={64}
            value={value.token_budget}
            className="h-8 w-28"
            onChange={(event) => onChange({ ...value, token_budget: Math.max(64, Number(event.target.value) || 64) })}
          />
        </label>
        <Button type="button" variant="outline" size="sm" onClick={add}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.lorebook.addEntry")}
        </Button>
      </div>
      {value.entries.length === 0 ? (
        <p className="rounded-xl bg-muted/60 px-4 py-3 text-ui-14 text-muted-foreground dark:bg-card">
          {t("characters.lorebook.empty")}
        </p>
      ) : (
        <ul className="flex flex-col gap-2" data-testid="lorebook-entries">
          {shown.map((entry, index) => {
            const expanded = open.has(entry.id);
            const label = entry.comment || entry.keys.join(", ") || t("characters.lorebook.untitled");
            return (
              <li key={`${entry.id}-${index}`} className="rounded-2xl bg-muted/55 dark:bg-card">
                <div className="flex items-center gap-3 px-3.5 py-2.5">
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
                    aria-expanded={expanded}
                    onClick={() =>
                      setOpen((current) => {
                        const next = new Set(current);
                        if (next.has(entry.id)) next.delete(entry.id);
                        else next.add(entry.id);
                        return next;
                      })
                    }
                  >
                    <HugeiconsIcon
                      icon={ArrowDown01Icon}
                      strokeWidth={1.75}
                      className={cn("size-4 shrink-0 text-muted-foreground transition-transform", !expanded && "-rotate-90")}
                    />
                    <span className={cn("truncate text-ui-14 font-medium", !entry.enabled && "text-muted-foreground line-through")}>
                      {label}
                    </span>
                    {entry.constant ? (
                      <span className="shrink-0 rounded-full bg-primary/12 px-2 py-0.5 text-ui-11 font-semibold text-primary">
                        {t("characters.lorebook.alwaysOn")}
                      </span>
                    ) : null}
                  </button>
                  <Switch
                    checked={entry.enabled}
                    aria-label={t("characters.lorebook.enabled")}
                    onCheckedChange={(checked) => update(index, { enabled: checked })}
                  />
                </div>
                {expanded ? (
                  <div className="flex flex-col gap-3 border-t border-border/60 px-3.5 pb-3.5 pt-3">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                        {t("characters.lorebook.label")}
                        <Input value={entry.comment} onChange={(event) => update(index, { comment: event.target.value })} />
                      </label>
                      <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                        {t("characters.lorebook.keys")}
                        <Input
                          defaultValue={entry.keys.join(", ")}
                          title={t("characters.lorebook.keysHint")}
                          onBlur={(event) => {
                            if (event.target.value === entry.keys.join(", ")) return;
                            update(index, { keys: parseKeyList(event.target.value) });
                          }}
                        />
                      </label>
                    </div>
                    <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                      {t("characters.lorebook.content")}
                      <Textarea
                        value={entry.content}
                        className="max-h-80 min-h-20 font-sans text-ui-14"
                        onChange={(event) => update(index, { content: event.target.value })}
                      />
                    </label>
                    <div className="flex flex-wrap items-end gap-4">
                      <label className="flex items-center gap-2 text-ui-13">
                        <Switch checked={entry.constant} onCheckedChange={(checked) => update(index, { constant: checked })} />
                        {t("characters.lorebook.alwaysOn")}
                      </label>
                      <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                        {t("characters.lorebook.position")}
                        <Select
                          value={entry.position}
                          onValueChange={(position) => update(index, { position: position as LorebookPosition })}
                        >
                          <SelectTrigger size="sm" className="h-8 w-48">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="before_char">{t("characters.lorebook.beforeChar")}</SelectItem>
                            <SelectItem value="after_char">{t("characters.lorebook.afterChar")}</SelectItem>
                            <SelectItem value="at_depth">{t("characters.lorebook.atDepth")}</SelectItem>
                          </SelectContent>
                        </Select>
                      </label>
                      {entry.position === "at_depth" ? (
                        <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                          {t("characters.lorebook.depth")}
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={entry.depth}
                            className="h-8 w-20"
                            onChange={(event) => update(index, { depth: Math.max(0, Number(event.target.value) || 0) })}
                          />
                        </label>
                      ) : null}
                      <label className="flex flex-col gap-1 text-ui-12 text-muted-foreground">
                        {t("characters.lorebook.order")}
                        <Input
                          type="number"
                          value={entry.insertion_order}
                          className="h-8 w-20"
                          onChange={(event) => update(index, { insertion_order: Number(event.target.value) || 0 })}
                        />
                      </label>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="ml-auto text-destructive hover:text-destructive"
                        onClick={() => remove(index)}
                      >
                        <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} className="size-icon" />
                        {t("characters.lorebook.remove")}
                      </Button>
                    </div>
                  </div>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
      {value.entries.length > limit ? (
        <Button type="button" variant="ghost" size="sm" className="self-start" onClick={() => setLimit((n) => n + PAGE * 4)}>
          {t("characters.lorebook.showMore", { count: value.entries.length - limit })}
        </Button>
      ) : null}
    </div>
  );
}
