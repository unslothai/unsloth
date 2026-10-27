// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Textarea } from "@/components/ui/textarea";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  cancelRun,
  followRunEvents,
  getFlow,
  startRun,
  uploadAsset,
} from "../api";
import { MediaThumb } from "../editor/media";
import { OutputValue } from "../editor/output-value";
import { DOCUMENT_ACCEPT, documentName } from "../editor/param-fields";
import type { FlowDocument, MediaValue, RunEvent } from "../types";
import {
  type AppField,
  type AppRunState,
  appFields,
  appOutputs,
  applyAppValues,
  foldAppEvents,
} from "./app-form";

const IDLE: AppRunState = {
  status: "idle",
  done: 0,
  total: 0,
  results: {},
  errors: [],
};

function FieldInput({
  field,
  value,
  onChange,
  onUploading,
}: {
  field: AppField;
  value: unknown;
  onChange: (value: unknown) => void;
  onUploading?: (busy: boolean) => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  switch (field.kind) {
    case "number":
      return (
        <Input
          type="number"
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) =>
            onChange(e.target.value === "" ? null : Number(e.target.value))
          }
        />
      );
    case "bool":
      return (
        <Checkbox
          checked={value === true || value === "true"}
          onCheckedChange={(checked) => onChange(checked === true)}
        />
      );
    case "select":
      return (
        <Select
          value={String(value ?? "")}
          onValueChange={(next) => onChange(next)}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(field.options ?? []).filter(Boolean).map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    case "list":
      return (
        <Textarea
          fieldSizing="fixed"
          rows={4}
          placeholder="One item per line"
          value={Array.isArray(value) ? value.join("\n") : String(value ?? "")}
          onChange={(e) =>
            onChange(e.target.value.split("\n").filter((l) => l.trim()))
          }
        />
      );
    case "image":
    case "audio":
    case "video":
    case "document": {
      const media =
        value && typeof value === "object" ? (value as MediaValue) : null;
      const docName = documentName(value);
      return (
        <div className="flex flex-col gap-2">
          {media && !docName && (
            <MediaThumb media={media} className="max-h-48 w-auto rounded-md" />
          )}
          {docName && <span className="text-sm">{docName}</span>}
          {uploadError && (
            <span className="text-destructive text-xs">{uploadError}</span>
          )}
          <input
            type="file"
            accept={
              field.kind === "document" ? DOCUMENT_ACCEPT : `${field.kind}/*`
            }
            disabled={uploading}
            className="text-sm"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (!file) return;
              setUploading(true);
              onUploading?.(true);
              void uploadAsset(file)
                .then(onChange)
                .then(() => setUploadError(null))
                .catch((error) =>
                  setUploadError(
                    error instanceof Error ? error.message : "Upload failed",
                  ),
                )
                .finally(() => {
                  setUploading(false);
                  onUploading?.(false);
                });
            }}
          />
        </div>
      );
    }
    case "json":
    case "text":
    case "textarea":
      return (
        <Textarea
          fieldSizing="fixed"
          rows={field.kind === "json" ? 5 : 3}
          value={
            typeof value === "string"
              ? value
              : value == null
                ? ""
                : JSON.stringify(value)
          }
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}

export function FlowAppPage({ flowId }: { flowId: string }): ReactElement {
  const navigate = useNavigate();
  const [doc, setDoc] = useState<FlowDocument | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [run, setRun] = useState<AppRunState>(IDLE);
  const [runId, setRunId] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Record<string, boolean>>({});
  const follower = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const loaded = generation.current;

  useEffect(() => {
    let alive = true;
    generation.current += 1;
    setDoc(null);
    setFailed(null);
    setValues({});
    setRun(IDLE);
    setRunId(null);
    setUploads({});
    void getFlow(flowId)
      .then((record) => {
        if (!alive) return;
        setDoc(record.doc);
        setValues(
          Object.fromEntries(
            appFields(record.doc).map((f) => [f.key, f.initial]),
          ),
        );
      })
      .catch(
        (error) =>
          alive &&
          setFailed(error instanceof Error ? error.message : "Not found"),
      );
    return () => {
      alive = false;
      follower.current?.abort();
    };
  }, [flowId]);

  const fields = useMemo(() => (doc ? appFields(doc) : []), [doc]);
  const outputs = useMemo(() => (doc ? appOutputs(doc) : []), [doc]);
  const running = run.status === "running" || run.status === "queued";
  const uploadingAny = Object.values(uploads).some(Boolean);

  const start = useCallback(async () => {
    if (!doc) return;
    follower.current?.abort();
    const controller = new AbortController();
    follower.current = controller;
    const { doc: filled, params } = applyAppValues(doc, fields, values);
    const started = generation.current;
    setRun({ ...IDLE, status: "queued" });
    try {
      const { run_id } = await startRun(flowId, { doc: filled, params });
      if (started !== generation.current) return;
      setRunId(run_id);
      const ids = new Set(outputs.map((n) => n.id));
      await followRunEvents(
        run_id,
        0,
        (event: RunEvent) =>
          setRun((state) => foldAppEvents(state, [event], ids)),
        controller.signal,
      );
    } catch (error) {
      if (started !== generation.current) return;
      setRun((state) => ({
        ...state,
        status: "failed",
        errors: [
          ...state.errors,
          error instanceof Error ? error.message : "Could not start the run",
        ],
      }));
    }
  }, [doc, fields, values, flowId, outputs]);

  if (failed)
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-muted-foreground">
        {failed}
      </div>
    );
  if (!doc)
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-muted-foreground">
        Loading…
      </div>
    );

  return (
    <div className="h-full min-h-0 w-full overflow-y-auto">
      <div
        className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-6"
        data-testid="flows-app"
      >
        <header className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <h1 className="truncate font-heading text-xl font-semibold">
              {doc.name}
            </h1>
            {doc.description && (
              <p className="mt-1 text-muted-foreground text-sm">
                {doc.description}
              </p>
            )}
          </div>
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              void navigate({ to: "/flows/$flowId", params: { flowId } })
            }
          >
            Edit flow
          </Button>
        </header>

        {fields.length > 0 && (
          <section className="flex flex-col gap-4 rounded-xl border bg-card p-4">
            {fields.map((field) => (
              // biome-ignore lint/a11y/noLabelWithoutControl: the control is the field rendered inside
              <label key={field.key} className="flex flex-col gap-1.5">
                <span className="text-sm font-medium">{field.label}</span>
                <FieldInput
                  field={field}
                  value={values[field.key]}
                  onChange={(value) =>
                    loaded === generation.current &&
                    setValues((prev) => ({ ...prev, [field.key]: value }))
                  }
                  onUploading={(busy) =>
                    loaded === generation.current &&
                    setUploads((prev) => ({ ...prev, [field.key]: busy }))
                  }
                />
              </label>
            ))}
          </section>
        )}

        <div className="flex items-center gap-3">
          {running ? (
            <Button
              type="button"
              variant="destructive"
              onClick={() => runId && void cancelRun(runId)}
              data-testid="flows-app-stop"
            >
              Stop
            </Button>
          ) : (
            <Button
              type="button"
              variant="dark"
              onClick={() => void start()}
              disabled={uploadingAny}
              data-testid="flows-app-run"
            >
              {uploadingAny ? "Uploading…" : "Run"}
            </Button>
          )}
          {run.status !== "idle" && (
            <span className="text-muted-foreground text-sm">
              {running
                ? `Running ${Math.min(run.done, run.total)} of ${run.total || "?"} steps`
                : `Run ${run.status}`}
            </span>
          )}
        </div>

        {run.errors.length > 0 && (
          <div className="rounded-md bg-destructive/10 px-3 py-2 text-destructive text-sm">
            {run.errors.map((e, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: messages can repeat
              <p key={i}>{e}</p>
            ))}
          </div>
        )}

        {outputs.length > 0 && (
          <section
            className="flex flex-col gap-4"
            data-testid="flows-app-results"
          >
            {outputs.map((node) => (
              <div
                key={node.id}
                className="flex flex-col gap-2 rounded-xl border bg-card p-4 dark"
              >
                <span className="text-muted-foreground text-xs uppercase tracking-wide">
                  {node.label ||
                    (node.type === "output.result"
                      ? String(node.params.name ?? "Result")
                      : "Result")}
                </span>
                {node.id in run.results ? (
                  <OutputValue value={run.results[node.id]} />
                ) : (
                  <span className="text-muted-foreground text-sm">
                    {running ? "Waiting…" : "Run the flow to see this."}
                  </span>
                )}
              </div>
            ))}
          </section>
        )}
      </div>
    </div>
  );
}
