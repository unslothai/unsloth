


// What a remote load will apply, otherwise unanswerable from outside the process. Read
// only: the model's settings page is the only place that owns this and the local store.

import { Skeleton } from "@/components/ui/skeleton";
import {
  type ApiModelOverride,
  type ApiModelOverrides,
  fetchModelOverrides,
} from "@/features/model-picker/api/model-overrides";
import { type ReactElement, useCallback, useEffect, useState } from "react";

/** Summary of the fields the loader will apply, in load order. */
function describeOverride(override: ApiModelOverride): string[] {
  const parts: string[] = [];
  if (override.custom_context_length) {
    parts.push(`${override.custom_context_length.toLocaleString()} context`);
  }
  if (override.max_seq_length) {
    parts.push(`${override.max_seq_length.toLocaleString()} max seq`);
  }
  if (override.kv_cache_dtype) {
    parts.push(`KV ${override.kv_cache_dtype}`);
  }
  if (override.speculative_type) {
    parts.push(
      override.spec_draft_n_max
        ? `spec ${override.speculative_type} ×${override.spec_draft_n_max}`
        : `spec ${override.speculative_type}`,
    );
  }
  if (override.n_parallel) {
    parts.push(`${override.n_parallel} parallel slots`);
  }
  if (override.tensor_parallel) {
    parts.push("tensor parallel");
  }
  if (override.gpu_memory_mode === "manual") {
    parts.push("manual GPU memory");
  }
  if (override.gpu_layers != null) {
    parts.push(`${override.gpu_layers} GPU layers`);
  }
  if (override.n_cpu_moe) {
    parts.push(`${override.n_cpu_moe} MoE layers on CPU`);
  }
  if (override.gpu_ids?.length) {
    parts.push(`GPU ${override.gpu_ids.join(", ")}`);
  }
  if (override.chat_template_override) {
    parts.push("custom chat template");
  }
  if (override.llama_extra_args?.length) {
    parts.push(override.llama_extra_args.join(" "));
  }
  return parts;
}

export function SavedModelSettingsPanel(): ReactElement {
  const [overrides, setOverrides] = useState<ApiModelOverrides | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setOverrides(await fetchModelOverrides());
      setError(null);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load saved model settings",
      );
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const entries = Object.entries(overrides ?? {});

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <h2 className="text-ui-16 font-semibold tracking-[-0.01em] text-foreground">
          Settings applied on API load
        </h2>
        <p className="text-sm text-muted-foreground">
          When a request names one of these models, Unsloth loads it with these
          settings, the same ones you saved in the model&apos;s settings page.
          Models without an entry load with app defaults. Edit or forget an
          entry from that model&apos;s settings, which keeps this list and the
          picker in step.
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      ) : overrides == null ? (
        <div className="flex flex-col gap-2">
          {[0, 1].map((i) => (
            <Skeleton key={i} className="h-14 w-full rounded-xl" />
          ))}
        </div>
      ) : entries.length === 0 ? (
        <p className="rounded-xl border border-border/60 bg-card px-4 py-6 text-center text-sm text-muted-foreground">
          No saved model settings yet. Open a model&apos;s settings, turn on
          &quot;Remember for this model&quot;, and it will be applied to API
          loads too.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {entries.map(([modelId, override]) => {
            const summary = describeOverride(override);
            return (
              <li
                key={modelId}
                className="flex min-w-0 items-start gap-3 rounded-xl border border-border/60 bg-card px-4 py-3"
              >
                <div className="flex min-w-0 flex-1 flex-col gap-1">
                  <span className="min-w-0 break-all font-mono text-ui-12 font-medium text-foreground">
                    {modelId}
                  </span>
                  <span className="min-w-0 break-words text-ui-11 text-muted-foreground">
                    {summary.length > 0 ? summary.join(" · ") : "App defaults"}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
