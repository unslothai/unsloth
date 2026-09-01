// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a remote load will apply, otherwise unanswerable from outside the process.

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import {
  type ApiModelOverride,
  type ApiModelOverrides,
  fetchModelOverrides,
  putModelOverride,
} from "@/features/model-picker/api/model-overrides";
import { deletePerModelConfigAliases } from "@/features/model-picker/model-config/per-model-config";
import { Trash2 } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { forgetModelOverride } from "../forget-model-override";

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

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
    parts.push(plural(override.n_parallel, "parallel slot"));
  }
  if (override.n_batch) {
    parts.push(`batch ${override.n_batch}`);
  }
  if (override.n_ubatch) {
    parts.push(`ubatch ${override.n_ubatch}`);
  }
  if (override.load_mode) {
    parts.push(`load ${override.load_mode}`);
  }
  if (override.spec_draft_cache_type) {
    parts.push(`draft KV ${override.spec_draft_cache_type}`);
  }
  // Both compared against undefined rather than tested for truth: 0 is a value the
  // user can pick for either (no checkpoints, no host cache) and would otherwise
  // be listed as unset.
  if (override.ctx_checkpoints !== undefined) {
    parts.push(plural(override.ctx_checkpoints, "checkpoint"));
  }
  if (override.cache_ram !== undefined) {
    parts.push(`cache RAM ${override.cache_ram} MiB`);
  }
  if (override.tensor_parallel) {
    parts.push("tensor parallel");
  }
  if (override.disable_vision) {
    parts.push("vision off");
  }
  if (override.gpu_memory_mode === "manual") {
    parts.push("manual GPU memory");
  }
  if (override.gpu_layers != null) {
    parts.push(plural(override.gpu_layers, "GPU layer"));
  }
  if (override.n_cpu_moe) {
    parts.push(`${plural(override.n_cpu_moe, "MoE layer")} on CPU`);
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
  const [forgetting, setForgetting] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // Each forget refetches, and a row disables only its own button, so two of them
  // overlap. The reads answer in whatever order the network gives, and the older one
  // saw the row the newer forget removed: last issued has to win, or a row the server
  // no longer holds is painted back and stays until the panel remounts.
  const loadSeq = useRef(0);

  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    try {
      const next = await fetchModelOverrides();
      if (seq !== loadSeq.current) {
        return;
      }
      setOverrides(next);
      setError(null);
    } catch (err: unknown) {
      if (seq !== loadSeq.current) {
        return;
      }
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

  const forget = useCallback(
    async (overrideKey: string) => {
      setForgetting((prev) => new Set(prev).add(overrideKey));
      try {
        await forgetModelOverride(overrideKey, {
          removeRemote: (modelId, ggufVariant) =>
            putModelOverride(modelId, ggufVariant, null),
          removeLocal: (modelId, ggufVariant) =>
            deletePerModelConfigAliases(modelId, ggufVariant),
          reload: load,
          onError: (message) => {
            toast.error(message);
          },
        });
      } finally {
        setForgetting((prev) => {
          const next = new Set(prev);
          next.delete(overrideKey);
          return next;
        });
      }
    },
    [load],
  );

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
          Models without an entry load with app defaults. Edit an entry from
          that model&apos;s settings, or forget it here, which clears it from
          the picker too.
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
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="shrink-0 text-muted-foreground hover:text-destructive"
                  aria-label={`Forget settings for ${modelId}`}
                  title="Forget these settings"
                  disabled={forgetting.has(modelId)}
                  onClick={() => void forget(modelId)}
                >
                  {forgetting.has(modelId) ? (
                    <Spinner label="Forgetting" />
                  ) : (
                    <Trash2 />
                  )}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
