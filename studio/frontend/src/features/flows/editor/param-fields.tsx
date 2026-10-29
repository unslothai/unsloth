// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthSessionEpoch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";
import { uploadAsset } from "../api";
import type { MediaValue, ParamSpec } from "../types";
import { modelChoiceOf } from "./model-choice";
import { FlowModelPicker } from "./model-picker-field";
import { DiceIcon, Upload01Icon } from "./icon-data";
import { MediaThumb, isMediaValue } from "./media";
import { pickOptions } from "./remote-options";

const FIELD = "text-ui-13";

export interface FieldProps {
  spec: ParamSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}

function JsonField({
  value,
  onChange,
  rows = 6,
  placeholder,
}: {
  value: unknown;
  onChange: (v: unknown) => void;
  rows?: number;
  placeholder?: string;
}) {
  const serialized =
    value === null || value === undefined ? "" : JSON.stringify(value, null, 2);
  const [text, setText] = useState(serialized);
  const [error, setError] = useState<string | null>(null);
  const lastCommitted = useRef(serialized);
  useEffect(() => {
    if (serialized !== lastCommitted.current) {
      setText(serialized);
      lastCommitted.current = serialized;
      setError(null);
    }
  }, [serialized]);
  return (
    <div className="flex flex-col gap-1">
      <Textarea
        fieldSizing="fixed"
        value={text}
        rows={rows}
        spellCheck={false}
        placeholder={placeholder}
        onChange={(e) => {
          setText(e.target.value);
          if (!e.target.value.trim()) {
            setError(null);
            lastCommitted.current = "";
            onChange(null);
            return;
          }
          try {
            const parsed = JSON.parse(e.target.value);
            setError(null);
            lastCommitted.current = JSON.stringify(parsed, null, 2);
            onChange(parsed);
          } catch (err) {
            setError(err instanceof Error ? err.message : "Invalid JSON");
          }
        }}
        className={`${FIELD} font-mono text-ui-12 leading-snug`}
      />
      {error && <p className="text-ui-11 text-destructive">{error}</p>}
    </div>
  );
}

function ModelField({ spec, value, onChange }: FieldProps) {
  const choice = modelChoiceOf(value);
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-2 text-ui-12 text-muted-foreground">
        <Checkbox
          checked={choice === null}
          onCheckedChange={(checked) =>
            onChange(checked === true ? null : { repo: "", variant: null })
          }
        />
        Use the loaded {spec.model_kind ?? ""} model
      </label>
      {choice !== null && (
        <FlowModelPicker
          kind={spec.model_kind ?? "chat"}
          choice={choice}
          onChange={onChange}
        />
      )}
    </div>
  );
}

export const DOCUMENT_ACCEPT = ".pdf,.docx,.txt,.md,.markdown,.html,.htm";

export function documentName(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const v = value as { kind?: unknown; ref?: unknown; name?: unknown };
  if (v.kind !== "document" || typeof v.ref !== "string") return null;
  return typeof v.name === "string" && v.name ? v.name : "Document";
}

function AssetField({ spec, value, onChange }: FieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const docName = documentName(value);
  const media = isMediaValue(value) ? (value as MediaValue) : null;
  const hasValue = media !== null || docName !== null;
  const accept =
    spec.accept === "document"
      ? DOCUMENT_ACCEPT
      : spec.accept
        ? `${spec.accept}/*`
        : undefined;
  return (
    <div className="flex flex-col gap-2">
      {media && (
        <MediaThumb
          media={media}
          className="max-h-40 w-full rounded-md object-contain"
        />
      )}
      {docName !== null && (
        <p className="truncate rounded-md border border-border px-2 py-1.5 text-ui-12 text-foreground">
          {docName}
        </p>
      )}
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          size="xs"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          <HugeiconsIcon icon={Upload01Icon} size={13} />
          {busy ? "Uploading…" : hasValue ? "Replace" : "Upload"}
        </Button>
        {hasValue && (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => onChange(null)}
            className="text-muted-foreground"
          >
            Clear
          </Button>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={async (e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          setBusy(true);
          try {
            onChange(await uploadAsset(file));
          } catch (error) {
            toast.error(
              error instanceof Error ? error.message : "Upload failed",
            );
          } finally {
            setBusy(false);
          }
        }}
      />
    </div>
  );
}

const remoteOptions = new Map<string, Promise<string[] | null>>();

function useRemoteOptions(spec: ParamSpec): string[] | null {
  const [options, setOptions] = useState<string[] | null>(null);
  const source = spec.options_from;
  const path = spec.options_path;
  useEffect(() => {
    if (!source) return;
    const key = `${getAuthSessionEpoch()}|${source}|${path ?? ""}`;
    if (source === "/api/flows") remoteOptions.delete(key);
    let request = remoteOptions.get(key);
    if (!request) {
      request = authFetch(source)
        .then(async (r) => (r.ok ? pickOptions(await r.json(), path) : null))
        .catch(() => null);
      remoteOptions.set(key, request);
      void request.then((found) => {
        if (found === null) remoteOptions.delete(key);
      });
    }
    let alive = true;
    void request.then((found) => {
      if (alive) setOptions(found);
    });
    return () => {
      alive = false;
    };
  }, [source, path]);
  return options;
}

function RemoteOptionsField({ spec, value, onChange }: FieldProps) {
  const options = useRemoteOptions(spec);
  const listId = useId();
  return (
    <>
      <Input
        value={String(value ?? "")}
        list={options ? listId : undefined}
        placeholder={options ? `${options.length} choices` : undefined}
        onChange={(e) => onChange(e.target.value)}
        className={FIELD}
      />
      {options ? (
        <datalist id={listId}>
          {options.map((option) => (
            <option key={option} value={option} />
          ))}
        </datalist>
      ) : null}
    </>
  );
}

function useRouteSuggestions(spec: ParamSpec): string[] {
  return useRemoteOptions(spec) ?? [];
}

function TextField({ spec, value, onChange }: FieldProps) {
  const listId = useId();
  const suggestions = useRouteSuggestions(spec);
  return (
    <>
      <Input
        value={(value as string) ?? ""}
        list={suggestions.length ? listId : undefined}
        onChange={(e) => onChange(e.target.value)}
        className={FIELD}
      />
      {suggestions.length > 0 && (
        <datalist id={listId}>
          {suggestions.map((id) => (
            <option key={id} value={id} />
          ))}
        </datalist>
      )}
    </>
  );
}

interface LoraEntry {
  id: string;
  weight: number;
}

function LoraListField({ spec, value, onChange }: FieldProps) {
  const listId = useId();
  const suggestions = useRouteSuggestions(spec);
  const rows = Array.isArray(value) ? (value as LoraEntry[]) : [];
  const update = (index: number, patch: Partial<LoraEntry>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((row, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id of their own
        <div key={index} className="flex items-center gap-1.5">
          <Input
            value={row.id}
            list={listId}
            placeholder="LoRA id"
            onChange={(e) => update(index, { id: e.target.value })}
            className={FIELD}
          />
          <Input
            type="number"
            min={0}
            max={2}
            step={0.05}
            value={row.weight}
            onChange={(e) =>
              update(index, { weight: Number(e.target.value) || 0 })
            }
            className={`${FIELD} w-20`}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => onChange(rows.filter((_, i) => i !== index))}
            className="text-muted-foreground hover:text-destructive"
            aria-label="Remove LoRA"
          >
            x
          </Button>
        </div>
      ))}
      {rows.length < 8 && (
        <Button
          type="button"
          variant="outline"
          size="xs"
          onClick={() => onChange([...rows, { id: "", weight: 1 }])}
          className="self-start"
        >
          Add LoRA
        </Button>
      )}
      <datalist id={listId}>
        {suggestions.map((id) => (
          <option key={id} value={id} />
        ))}
      </datalist>
    </div>
  );
}

function ParamFieldImpl({ spec, value, onChange }: FieldProps) {
  if (spec.options_from && (spec.kind === "text" || spec.kind === "select"))
    return <RemoteOptionsField spec={spec} value={value} onChange={onChange} />;
  switch (spec.kind) {
    case "text":
      // Fine-tune's base model is a plain repo string; it gets the same picker.
      if (spec.id === "base_model")
        return (
          <FlowModelPicker
            kind="train"
            choice={{ repo: String(value ?? ""), variant: null }}
            onChange={(choice) => onChange(choice.repo)}
          />
        );
      return <TextField spec={spec} value={value} onChange={onChange} />;
    case "lora_list":
      return <LoraListField spec={spec} value={value} onChange={onChange} />;
    case "textarea":
      return (
        <div className="flex flex-col gap-1">
          <Textarea
            fieldSizing="fixed"
            value={(value as string) ?? ""}
            rows={Math.min(
              14,
              Math.max(3, String(value ?? "").split("\n").length + 1),
            )}
            onChange={(e) => onChange(e.target.value)}
            className={`${FIELD} leading-snug`}
          />
          {spec.id.includes("prompt") || spec.id === "template" ? (
            <p className="text-ui-10p5 text-muted-foreground">
              Use <code className="text-foreground">{"{{name}}"}</code> to insert
              a connected input or flow input.
            </p>
          ) : null}
        </div>
      );
    case "number":
    case "int":
      return (
        <Input
          type="number"
          value={value === null || value === undefined ? "" : String(value)}
          min={spec.min}
          max={spec.max}
          step={spec.step ?? (spec.kind === "int" ? 1 : "any")}
          placeholder={spec.default === null ? "default" : undefined}
          onChange={(e) => {
            if (e.target.value === "")
              return onChange(spec.default === null ? null : 0);
            const n =
              spec.kind === "int"
                ? Math.round(Number(e.target.value))
                : Number(e.target.value);
            if (Number.isFinite(n)) onChange(n);
          }}
          className={FIELD}
        />
      );
    case "bool":
      return (
        <Switch checked={!!value} onCheckedChange={() => onChange(!value)} />
      );
    case "select":
      return (
        <Select
          // A param the doc never stored shows its default, as the run uses it, not an empty box.
          value={String(
            value === null || value === undefined || value === ""
              ? (spec.default ?? spec.options?.[0] ?? "")
              : value,
          )}
          onValueChange={(next) => onChange(next)}
        >
          <SelectTrigger className={`${FIELD} w-full`}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(spec.options ?? []).map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    case "seed":
      return (
        <div className="flex gap-2">
          <Input
            type="number"
            value={value === null || value === undefined ? "" : String(value)}
            placeholder="From the flow seed"
            onChange={(e) =>
              onChange(
                e.target.value === ""
                  ? null
                  : Math.max(0, Math.round(Number(e.target.value))),
              )
            }
            className={FIELD}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            title="New random seed"
            onClick={() => onChange(Math.floor(Math.random() * 2 ** 31))}
          >
            <HugeiconsIcon icon={DiceIcon} size={15} />
          </Button>
        </div>
      );
    case "model":
      return <ModelField spec={spec} value={value} onChange={onChange} />;
    case "color":
      return (
        <input
          type="color"
          value={(value as string) || "#000000"}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-16 cursor-pointer rounded border border-border bg-transparent"
        />
      );
    case "string_list":
      return (
        <Textarea
          fieldSizing="fixed"
          value={Array.isArray(value) ? (value as string[]).join("\n") : ""}
          rows={Math.min(
            12,
            Math.max(3, (Array.isArray(value) ? value.length : 0) + 1),
          )}
          placeholder="One per line"
          onChange={(e) => onChange(e.target.value.split("\n"))}
          className={`${FIELD} leading-snug`}
        />
      );
    case "expr":
      return (
        <div className="flex flex-col gap-1">
          <JsonField value={value} onChange={onChange} rows={4} />
          <p className="text-ui-10p5 leading-snug text-muted-foreground">
            {'{"op": ">", "args": [{"var": "courage"}, 2]}'} · ops: == != &lt;
            &gt; and or not + - * / min max len contains dice
          </p>
        </div>
      );
    case "asset":
      return <AssetField spec={spec} value={value} onChange={onChange} />;
    default:
      return (
        <JsonField
          value={value}
          onChange={onChange}
          rows={spec.kind === "json" ? 5 : 10}
        />
      );
  }
}

export const ParamField = memo(ParamFieldImpl);
