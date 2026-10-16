// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import React, { useState, useEffect } from "react";
import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { HugeiconsIcon } from "@hugeicons/react";
import { Exchange01Icon } from "@hugeicons/core-free-icons";
import { listLocalModels, type LocalModelInfo } from "@/features/hub";
import { Spinner } from "@/components/ui/spinner";
import { Progress } from "@/components/ui/progress";
import { toast } from "@/lib/toast";
import { useT, type TranslationKey } from "@/i18n";
import { authFetch } from "@/features/auth";

type ConvertJob = { state: "running" | "done" | "error"; stage: string };

const STAGE_KEYS: Record<string, TranslationKey> = {
  starting: "convert.stageStarting",
  loading: "convert.stageLoading",
  saving: "convert.stageSaving",
};

export function ConvertPage() {
  const t = useT();
  const [format, setFormat] = useState("ov_int4");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [job, setJob] = useState<(ConvertJob & { modelId: string }) | null>(
    null,
  );

  // Poll while a conversion runs, the same way the download manager tracks a download.
  useEffect(() => {
    if (job?.state !== "running") return;
    const modelId = job?.modelId;
    if (!modelId) return;
    const timer = setInterval(async () => {
      try {
        const res = await authFetch(
          `/api/convert/status?model_id=${encodeURIComponent(modelId)}`,
        );
        if (!res.ok) return;
        const next = (await res.json()) as ConvertJob;
        setJob({ ...next, modelId });
        if (next.state === "done") {
          toast.success(t("convert.done", { model: modelId }));
          listLocalModels()
            .then((r) => setLocalModels(r.models))
            .catch(() => {});
        } else if (next.state === "error") {
          toast.error(t("convert.failed", { model: modelId }), {
            description: next.stage,
          });
        }
      } catch {
        // transient; the next tick retries
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [job?.state, job?.modelId, t]);

  useEffect(() => {
    let cancelled = false;
    listLocalModels()
      .then(({ models }) => {
        if (!cancelled) {
          setLocalModels(models);
          setIsLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto max-w-7xl px-5 py-8 sm:px-9">
        <div className="mb-8 flex flex-col gap-0.5">
          <h1 className="text-ui-30 font-semibold text-foreground">
            {t("convert.title")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("convert.subtitle")}
          </p>
        </div>

        <SectionCard
          icon={<HugeiconsIcon icon={Exchange01Icon} className="size-5" />}
          title={t("convert.cardTitle")}
          description={t("convert.cardDescription")}
        >
          <div className="flex flex-col gap-6 py-4">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                {t("convert.selectModel")}
              </label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue
                    placeholder={
                      isLoading
                        ? t("convert.loadingModels")
                        : t("convert.selectModelPlaceholder")
                    }
                  />
                </SelectTrigger>
                <SelectContent>
                  {isLoading ? (
                    <div className="p-4 flex justify-center">
                      <Spinner className="size-4" />
                    </div>
                  ) : localModels.length === 0 ? (
                    <div className="p-4 text-center text-sm text-muted-foreground">
                      {t("convert.noModels")}
                    </div>
                  ) : (
                    localModels
                      .filter((m) => {
                        const id = m.id.toLowerCase();
                        return !(
                          id.includes("int4") ||
                          id.includes("int8") ||
                          id.includes("-ov") ||
                          id.includes("openvino")
                        );
                      })
                      .map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.id}
                        </SelectItem>
                      ))
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                {t("convert.targetFormat")}
              </label>
              <Select value={format} onValueChange={setFormat}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ov_int4">
                    {t("convert.formatInt4")}
                  </SelectItem>
                  <SelectItem value="ov_int8">
                    {t("convert.formatInt8")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button
              className="w-fit"
              onClick={async () => {
                try {
                  const res = await authFetch("/api/convert", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      model_id: selectedModel,
                      format: format,
                    }),
                  });
                  if (!res.ok) throw new Error(await res.text());
                  setJob({
                    state: "running",
                    stage: "starting",
                    modelId: selectedModel,
                  });
                } catch (e) {
                  toast.error(t("convert.startFailed"), {
                    description: String(e),
                  });
                }
              }}
              disabled={!selectedModel || job?.state === "running"}
            >
              {t("convert.start")}
            </Button>
            {job?.state === "running" ? (
              <div className="flex flex-col gap-1.5" aria-live="polite">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="truncate">{job.modelId}</span>
                  <span>
                    {STAGE_KEYS[job.stage]
                      ? t(STAGE_KEYS[job.stage])
                      : job.stage}
                  </span>
                </div>
                <Progress indeterminate />
              </div>
            ) : null}
          </div>
        </SectionCard>
      </main>
    </div>
  );
}
