// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One model's saved sampling defaults: the server copy fills API requests that leave a value
// out, and Chat's per-model memory gets the same values so a load starts from them.

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { type InferenceParams, useChatRuntimeStore } from "@/features/chat";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import {
  SAMPLING_DEFAULT_FIELDS,
  type SamplingDefaultField,
  type SamplingDefaults,
  fetchModelSamplingDefaults,
  putModelSamplingDefaults,
} from "../api/model-sampling-defaults";

type ChatSamplingKey =
  | "temperature"
  | "topP"
  | "topK"
  | "minP"
  | "repetitionPenalty"
  | "presencePenalty";

type ChatSamplingPatch = Partial<
  Pick<InferenceParams, "minPMode" | ChatSamplingKey>
>;

interface FieldSpec {
  label: string;
  min: number;
  max: number;
  step: number;
  chatKey: ChatSamplingKey;
}

// Bounds match the backend's _SAMPLING_FIELDS, so nothing typed here is dropped on save.
const FIELDS: Record<SamplingDefaultField, FieldSpec> = {
  temperature: {
    label: "Temperature",
    min: 0,
    max: 2,
    step: 0.05,
    chatKey: "temperature",
  },
  // biome-ignore lint/style/useNamingConvention: API schema
  top_p: { label: "Top P", min: 0, max: 1, step: 0.01, chatKey: "topP" },
  // biome-ignore lint/style/useNamingConvention: API schema
  top_k: { label: "Top K", min: -1, max: 100, step: 1, chatKey: "topK" },
  // biome-ignore lint/style/useNamingConvention: API schema
  min_p: { label: "Min P", min: 0, max: 1, step: 0.01, chatKey: "minP" },
  // biome-ignore lint/style/useNamingConvention: API schema
  repetition_penalty: {
    label: "Repetition Penalty",
    min: 1,
    max: 2,
    step: 0.05,
    chatKey: "repetitionPenalty",
  },
  // biome-ignore lint/style/useNamingConvention: API schema
  presence_penalty: {
    label: "Presence Penalty",
    min: 0,
    max: 2,
    step: 0.1,
    chatKey: "presencePenalty",
  },
};

type Drafts = Partial<Record<SamplingDefaultField, string>>;

function parseDraft(
  field: SamplingDefaultField,
  raw: string | undefined,
): number | null {
  if (raw == null || raw.trim() === "") {
    return null;
  }
  const value = Number(raw);
  const { min, max } = FIELDS[field];
  if (!Number.isFinite(value) || value < min || value > max) {
    return null;
  }
  return field === "top_k" ? Math.round(value) : value;
}

function draftsFrom(saved: SamplingDefaults): Drafts {
  const drafts: Drafts = {};
  for (const field of SAMPLING_DEFAULT_FIELDS) {
    const value = saved[field];
    if (typeof value === "number") {
      drafts[field] = String(value);
    }
  }
  return drafts;
}

/** Chat lays its memory of a model over the model's own defaults on every load, so a save has to
 *  land there too or the chat keeps showing the last values used. A cleared field has no value to
 *  write and keeps what chat remembers. */
function chatPatchFrom(saved: SamplingDefaults): ChatSamplingPatch {
  const patch: ChatSamplingPatch = {};
  for (const field of SAMPLING_DEFAULT_FIELDS) {
    const value = saved[field];
    if (typeof value === "number") {
      patch[FIELDS[field].chatKey] = value;
    }
  }
  if (typeof saved.min_p === "number") {
    patch.minPMode = "custom";
  }
  return patch;
}

function SamplingFieldRow({
  field,
  value,
  onChange,
}: {
  field: SamplingDefaultField;
  value: string;
  onChange: (value: string) => void;
}) {
  const spec = FIELDS[field];
  const id = `sampling-default-${field}`;
  const invalid = value.trim() !== "" && parseDraft(field, value) == null;
  return (
    <div className="flex items-center justify-between gap-4">
      <Label htmlFor={id}>
        {spec.label}
        <span className="ml-1 font-normal text-muted-foreground">
          ({spec.min} to {spec.max})
        </span>
      </Label>
      <Input
        id={id}
        type="number"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        placeholder="Model default"
        aria-invalid={invalid || undefined}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-32"
      />
    </div>
  );
}

function SamplingDefaultsDialog({
  open,
  onOpenChange,
  stored,
  displayName,
  rememberParamsPerModel,
  onWrite,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  stored: SamplingDefaults;
  displayName: string;
  rememberParamsPerModel: boolean;
  onWrite: (next: SamplingDefaults) => Promise<boolean>;
}) {
  // Seeded once per open: the parent remounts this with a new key each time it opens.
  const [drafts, setDrafts] = useState<Drafts>(() => draftsFrom(stored));
  const [saving, setSaving] = useState(false);
  const invalid = SAMPLING_DEFAULT_FIELDS.some(
    (field) =>
      (drafts[field] ?? "").trim() !== "" &&
      parseDraft(field, drafts[field]) == null,
  );

  const write = async (next: SamplingDefaults) => {
    setSaving(true);
    const ok = await onWrite(next);
    setSaving(false);
    if (ok) {
      onOpenChange(false);
    }
  };

  const save = () => {
    const next: SamplingDefaults = {};
    for (const field of SAMPLING_DEFAULT_FIELDS) {
      const value = parseDraft(field, drafts[field]);
      if (value != null) {
        next[field] = value;
      }
    }
    write(next).catch(() => undefined);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{displayName} sampling defaults</DialogTitle>
          <DialogDescription>
            Used when Chat loads this model, and by API requests that leave a
            value out, including one that switches to this model. Leave a field
            empty to follow the model&apos;s recommended value.
          </DialogDescription>
        </DialogHeader>

        {rememberParamsPerModel ? null : (
          <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs leading-relaxed text-muted-foreground">
            &quot;Remember settings per model&quot; is off in Settings, so Chat
            will not restore these. API requests still use them.
          </p>
        )}

        <div className="flex flex-col gap-3">
          {SAMPLING_DEFAULT_FIELDS.map((field) => (
            <SamplingFieldRow
              key={field}
              field={field}
              value={drafts[field] ?? ""}
              onChange={(value) =>
                setDrafts((current) => ({ ...current, [field]: value }))
              }
            />
          ))}
        </div>

        <DialogFooter>
          {Object.keys(stored).length > 0 ? (
            <Button
              variant="outline"
              className="sm:mr-auto"
              disabled={saving}
              onClick={() => {
                write({}).catch(() => undefined);
              }}
            >
              Clear
            </Button>
          ) : null}
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button disabled={saving || invalid} onClick={save}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** The run settings row that opens the editor and says whether this model has saved values. */
export function SamplingDefaultsRow({
  modelId,
  chatModelId,
  displayName,
  rowClassName,
  labelClassName,
}: {
  /** What the server row is keyed by: the id the run settings are stored under. */
  modelId: string;
  /** The id Chat remembers this model's settings under. */
  chatModelId: string;
  displayName: string;
  rowClassName: string;
  labelClassName: string;
}) {
  const setRememberedParamsForModel = useChatRuntimeStore(
    (state) => state.setRememberedParamsForModel,
  );
  const rememberParamsPerModel = useChatRuntimeStore(
    (state) => state.rememberParamsPerModel,
  );
  const [stored, setStored] = useState<{
    modelId: string;
    saved: SamplingDefaults;
  } | null>(null);
  const [openCount, setOpenCount] = useState(0);
  const [open, setOpen] = useState(false);
  const [opening, setOpening] = useState(false);
  const saved = stored?.modelId === modelId ? stored.saved : null;

  useEffect(() => {
    let cancelled = false;
    fetchModelSamplingDefaults()
      .then((all) => {
        if (!cancelled) {
          setStored({ modelId, saved: all[modelId] ?? {} });
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const openEditor = async () => {
    setOpening(true);
    try {
      // Fresh, so another tab's save is what the fields start from.
      const all = await fetchModelSamplingDefaults();
      setStored({ modelId, saved: all[modelId] ?? {} });
      setOpenCount((count) => count + 1);
      setOpen(true);
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setOpening(false);
    }
  };

  const write = async (next: SamplingDefaults): Promise<boolean> => {
    try {
      const all = await putModelSamplingDefaults(modelId, next);
      const written = all[modelId] ?? {};
      setRememberedParamsForModel(chatModelId, chatPatchFrom(written));
      setStored({ modelId, saved: written });
      toast.success(
        Object.keys(written).length > 0
          ? "Sampling defaults saved."
          : "Sampling defaults cleared.",
      );
      return true;
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : String(error));
      return false;
    }
  };

  const savedCount = saved ? Object.keys(saved).length : 0;
  return (
    <div className={rowClassName}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={labelClassName}>Sampling Defaults</span>
        <InfoHint>
          Temperature, Top P and the other sampling values this model starts
          with in Chat and uses for API requests that leave them out.
        </InfoHint>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="shrink-0"
        aria-label="Edit sampling defaults"
        disabled={opening}
        onClick={() => {
          openEditor().catch(() => undefined);
        }}
      >
        {savedCount > 0 ? `Custom (${savedCount})` : "Model default"}
      </Button>
      {open ? (
        <SamplingDefaultsDialog
          key={openCount}
          open={open}
          onOpenChange={setOpen}
          stored={saved ?? {}}
          displayName={displayName}
          rememberParamsPerModel={rememberParamsPerModel}
          onWrite={write}
        />
      ) : null}
    </div>
  );
}
