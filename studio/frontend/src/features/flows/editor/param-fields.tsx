// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";
import { uploadAsset } from "../api";
import type { MediaValue, ModelChoice, ParamSpec } from "../types";
import { DiceIcon, Upload01Icon } from "./icon-data";
import { MediaThumb, isMediaValue } from "./media";

const FIELD =
  "w-full rounded-md border border-white/10 bg-zinc-900 px-2 py-1.5 text-[13px] text-zinc-100 outline-none focus:border-sky-500/60";

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
      <textarea
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
        className={`${FIELD} font-mono text-[12px] leading-snug`}
      />
      {error && <p className="text-[11px] text-red-300">{error}</p>}
    </div>
  );
}

let modelSuggestions: Promise<string[]> | null = null;
function loadModelSuggestions(): Promise<string[]> {
  modelSuggestions ??= authFetch("/api/models/list")
    .then(async (response) => {
      if (!response.ok) return [];
      const body = (await response.json()) as {
        models?: { id?: string; name?: string }[];
        default_models?: string[];
      };
      const ids = new Set<string>();
      for (const model of body.models ?? []) {
        const id = model.id ?? model.name;
        if (id && !id.startsWith("/")) ids.add(id);
      }
      for (const id of body.default_models ?? []) ids.add(id);
      return [...ids].slice(0, 400);
    })
    .catch(() => []);
  return modelSuggestions;
}

function ModelField({ spec, value, onChange }: FieldProps) {
  const choice = (value ?? null) as ModelChoice | null;
  const listId = useId();
  const [suggestions, setSuggestions] = useState<string[]>([]);
  useEffect(() => {
    let alive = true;
    void loadModelSuggestions().then((ids) => alive && setSuggestions(ids));
    return () => {
      alive = false;
    };
  }, []);
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-2 text-[12px] text-zinc-400">
        <input
          type="checkbox"
          checked={choice === null}
          onChange={(e) =>
            onChange(e.target.checked ? null : { repo: "", variant: null })
          }
          className="accent-sky-500"
        />
        Use the loaded {spec.model_kind ?? ""} model
      </label>
      {choice !== null && (
        <>
          <input
            value={choice.repo}
            list={listId}
            placeholder="org/model (Hugging Face repo)"
            onChange={(e) => onChange({ ...choice, repo: e.target.value })}
            className={FIELD}
          />
          <datalist id={listId}>
            {suggestions.map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
          <input
            value={choice.variant ?? ""}
            placeholder="Variant / quant (optional, e.g. Q4_K_M)"
            onChange={(e) =>
              onChange({ ...choice, variant: e.target.value || null })
            }
            className={FIELD}
          />
        </>
      )}
    </div>
  );
}

function AssetField({ spec, value, onChange }: FieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const media = isMediaValue(value) ? (value as MediaValue) : null;
  const accept = spec.accept ? `${spec.accept}/*` : undefined;
  return (
    <div className="flex flex-col gap-2">
      {media && (
        <MediaThumb
          media={media}
          className="max-h-40 w-full rounded-md object-contain"
        />
      )}
      <div className="flex gap-2">
        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-1.5 rounded-md border border-white/10 px-2 py-1 text-[12px] text-zinc-300 hover:bg-white/5 disabled:opacity-50"
        >
          <HugeiconsIcon icon={Upload01Icon} size={13} />
          {busy ? "Uploading…" : media ? "Replace" : "Upload"}
        </button>
        {media && (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="text-[12px] text-zinc-500 hover:text-zinc-300"
          >
            Clear
          </button>
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

function ParamFieldImpl({ spec, value, onChange }: FieldProps) {
  switch (spec.kind) {
    case "text":
      return (
        <input
          value={(value as string) ?? ""}
          onChange={(e) => onChange(e.target.value)}
          className={FIELD}
        />
      );
    case "textarea":
      return (
        <div className="flex flex-col gap-1">
          <textarea
            value={(value as string) ?? ""}
            rows={Math.min(
              14,
              Math.max(3, String(value ?? "").split("\n").length + 1),
            )}
            onChange={(e) => onChange(e.target.value)}
            className={`${FIELD} leading-snug`}
          />
          {spec.id.includes("prompt") || spec.id === "template" ? (
            <p className="text-[10.5px] text-zinc-500">
              Use <code className="text-zinc-400">{"{{name}}"}</code> to insert
              a connected input or flow input.
            </p>
          ) : null}
        </div>
      );
    case "number":
    case "int":
      return (
        <input
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
        <button
          type="button"
          role="switch"
          aria-checked={!!value}
          onClick={() => onChange(!value)}
          className={`relative h-5 w-9 rounded-full transition-colors ${value ? "bg-sky-500" : "bg-zinc-700"}`}
        >
          <span
            className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-[left] ${value ? "left-4.5" : "left-0.5"}`}
          />
        </button>
      );
    case "select":
      return (
        <select
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className={FIELD}
        >
          {(spec.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    case "seed":
      return (
        <div className="flex gap-2">
          <input
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
          <button
            type="button"
            title="New random seed"
            onClick={() => onChange(Math.floor(Math.random() * 2 ** 31))}
            className="rounded-md border border-white/10 px-2 text-zinc-300 hover:bg-white/5"
          >
            <HugeiconsIcon icon={DiceIcon} size={15} />
          </button>
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
          className="h-8 w-16 cursor-pointer rounded border border-white/10 bg-transparent"
        />
      );
    case "string_list":
      return (
        <textarea
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
          <p className="text-[10.5px] leading-snug text-zinc-500">
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
