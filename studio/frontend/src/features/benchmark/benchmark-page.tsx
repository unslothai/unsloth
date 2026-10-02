// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Spinner } from "@/components/ui/spinner";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import { useHfTokenStore } from "@/features/hub";
import {
  DOWNLOAD_KIND,
  downloadManager,
  subscribeJobListeners,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { AlertCircleIcon, Rocket01Icon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { BENCH_CARD, RUN_BUTTON } from "@/features/benchmarks/components/bench-ui";
import { CountInput, Field } from "@/features/benchmarks/components/setup-panel";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDebouncedValue } from "@/hooks";
import type { BenchmarkTaskConfig, BenchmarkTaskInfo } from "./api/benchmark-api";
import { fetchBenchmarkTaskConfig, fetchBenchmarkTasks } from "./api/benchmark-api";
import type { EvalsModel } from "./use-evals-model";
import { BenchmarkRunPanel } from "./components/benchmark-run-panel";
import { BenchmarkHistoryPanel } from "./components/benchmark-history-panel";
import { EvalScoreboard } from "./components/eval-scoreboard";
import {
  isBenchmarkPanelActive,
  useBenchmarkRuntimeStore,
} from "./stores/benchmark-runtime-store";
import { useChatRuntimeStore } from "@/features/chat";

/** The Evals tab of the Benchmarks page: lm-eval tasks on a picked model. The model
 * itself is picked in the header pill next to the tabs (EvalsModelPicker), so the
 * selection state arrives via the `evals` bundle BenchmarksPage owns. */
export function BenchmarkPage({ evals }: { evals: EvalsModel }) {
  const t = useT();
  const hfToken = useHfTokenStore((s) => s.token);

  const [batchSize, setBatchSize] = useState(0); // 0 = auto
  const [numFewshot, setNumFewshot] = useState(0);
  const [maxTokens, setMaxTokens] = useState(32768);

  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);

  // A pick made in the header pill closes a stale run panel: the results below it
  // belong to the model that was picked before.
  const pickedModelRef = useRef<string | null>(evals.selectedModel);
  useEffect(() => {
    if (pickedModelRef.current !== evals.selectedModel) {
      pickedModelRef.current = evals.selectedModel;
      setPanelOpen(false);
    }
  }, [evals.selectedModel]);

  const [tasks, setTasks] = useState<BenchmarkTaskInfo[]>([]);
  const [loadingTasks, setLoadingTasks] = useState(true);
  const [selectedTask, setSelectedTask] = useState("mmlu");
  const [taskConfig, setTaskConfig] = useState<BenchmarkTaskConfig | null>(null);
  const fewshotSupported = taskConfig?.num_fewshot !== 0;

  // Fetch task config when selected task changes
  useEffect(() => {
    let cancelled = false;
    setTaskConfig(null);
    fetchBenchmarkTaskConfig(selectedTask)
      .then((cfg) => {
        if (!cancelled) {
          setTaskConfig(cfg);
          if (cfg.num_fewshot != null && cfg.num_fewshot > 0) {
            setNumFewshot(cfg.num_fewshot);
          }
        }
      })
      .catch(() => {
        if (!cancelled) setTaskConfig(null);
      });
    return () => { cancelled = true; };
  }, [selectedTask]);
  const [taskSearch, setTaskSearch] = useState("");
  const taskAnchorRef = useRef<HTMLDivElement>(null);
  const selectingTaskRef = useRef(false);

  const [panelOpen, setPanelOpen] = useState(false);

  const runBenchmark = useBenchmarkRuntimeStore((s) => s.run);
  const resetBenchmarkRun = useBenchmarkRuntimeStore((s) => s.reset);
  const isRunning = useBenchmarkRuntimeStore((s) => s.isRunning);
  const panelActive = useBenchmarkRuntimeStore(isBenchmarkPanelActive);

  useEffect(() => {
    let cancelled = false;
    setLoadingTasks(true);
    fetchBenchmarkTasks()
      .then((data) => {
        if (!cancelled) {
          setTasks(data.tasks);
        }
      })
      .catch(() => {
        // silently ignore — the dropdown will just be empty
      })
      .finally(() => {
        if (!cancelled) setLoadingTasks(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const debouncedTaskSearch = useDebouncedValue(taskSearch, 150);

  // Pre-compute lowercased lookup and curated/non-curated split (stable across searches)
  const taskIndex = useMemo(() => {
    const curated = new Set([
      "mmlu", "mmlu_pro", "gpqa_main_cot_zeroshot", "hellaswag",
      "arc_challenge", "winogrande", "gsm8k", "hendrycks_math",
      "ifeval", "truthfulqa_mc1", "longbench",
      "bbh", "bbh_fewshot", "bbh_zeroshot",
    ]);
    const curatedIds: string[] = [];
    const rest: Array<{ id: string; lowerId: string; lowerName: string }> = [];
    for (const t of tasks) {
      const entry = { id: t.id, lowerId: t.id.toLowerCase(), lowerName: t.name.toLowerCase() };
      if (curated.has(t.id)) {
        curatedIds.push(t.id);
      } else {
        rest.push(entry);
      }
    }
    return { curatedIds, rest };
  }, [tasks]);

  const taskItems = useMemo(() => {
    const q = debouncedTaskSearch.toLowerCase().trim();
    if (!q) {
      return [...taskIndex.curatedIds, ...taskIndex.rest.map((e) => e.id)].slice(0, 50);
    }

    const words = q.split(/\s+/).filter(Boolean);
    const starts: string[] = [];
    const contains: string[] = [];
    for (const { id, lowerId, lowerName } of taskIndex.rest) {
      const matchesAll = words.every((w) => lowerId.includes(w) || lowerName.includes(w));
      if (!matchesAll) continue;
      if (lowerId.startsWith(q) || lowerName.startsWith(q)) {
        starts.push(id);
      } else {
        contains.push(id);
      }
    }
    // Also search curated tasks
    for (const id of taskIndex.curatedIds) {
      const lowerId = id.toLowerCase();
      const matchesAll = words.every((w) => lowerId.includes(w));
      if (!matchesAll) continue;
      if (lowerId.startsWith(q)) {
        starts.unshift(id);
      } else {
        starts.push(id);
      }
    }
    return [...starts, ...contains].slice(0, 50);
  }, [taskIndex, debouncedTaskSearch]);

  const handleStartAndOpenPanel = useCallback(() => {
    const model = evals.selectedModel;
    if (!model) return;

    const resolvedSource = evals.selectedModelSource === "lora" ? "checkpoint" : "local";

    if (!isRunning) {
      resetBenchmarkRun();
    }
    setPanelOpen(false);

    const extraParams = {
      batch_size: batchSize > 0 ? String(batchSize) : "auto",
      num_fewshot: numFewshot > 0 ? numFewshot : null,
      max_tokens: maxTokens,
    };

    const needsDownload =
      evals.selectedModelSource === "hub" &&
      evals.selectedModelIsDownloaded === false &&
      !evals.downloadingForBenchmark;
    if (needsDownload) {
      evals.setDownloadingForBenchmark(true);
      evals.setDownloadWatch(subscribeJobListeners(DOWNLOAD_KIND.MODEL, model, {
        onComplete: () => {
          evals.setDownloadWatch(null);
          evals.setDownloadingForBenchmark(false);
          evals.setSelectedModelIsDownloaded(true);
          setPanelOpen(true);
          prepareHfTokenForUse(hfToken, { allowAnonymous: true }).then((preparedToken) => {
            if (preparedToken.proceed) {
              void runBenchmark(model, resolvedSource, selectedTask, extraParams);
            }
          });
        },
        onError: () => {
          evals.setDownloadWatch(null);
          evals.setDownloadingForBenchmark(false);
          setPanelOpen(false);
        },
        onCancelled: () => {
          evals.setDownloadWatch(null);
          evals.setDownloadingForBenchmark(false);
          setPanelOpen(false);
        },
      }));
      void downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: model,
        variant: evals.selectedGgufVariant,
        expectedBytes: 0,
      });
      return;
    }

    setPanelOpen(true);
    prepareHfTokenForUse(hfToken, { allowAnonymous: true }).then((preparedToken) => {
      if (preparedToken.proceed) {
        void runBenchmark(model, resolvedSource, selectedTask, extraParams);
      }
    });
  }, [evals, selectedTask, hfToken, runBenchmark, isRunning, resetBenchmarkRun, batchSize, numFewshot, maxTokens]);

  const handleClosePanel = useCallback(() => {
    resetBenchmarkRun();
    setPanelOpen(false);
  }, [resetBenchmarkRun]);

  const showPanel = panelOpen || panelActive;

  const panelEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showPanel) return;
    const id = window.setTimeout(() => {
      panelEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, 60);
    return () => window.clearTimeout(id);
  }, [showPanel]);

  return (
    <div className="@container/evals">
      <div className="grid grid-cols-1 items-start gap-6 @3xl/evals:grid-cols-[calc(264px*var(--ui-space-scale,1))_minmax(0,1fr)]">
        <div
          className={cn(
            BENCH_CARD,
            "flex min-w-0 flex-col gap-6 px-5 pb-5 pt-4 @3xl/evals:sticky @3xl/evals:top-6",
          )}
        >
          <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
            Setup
          </span>

          {evals.modelsError && (
            <div className="flex items-center gap-2 text-ui-12 text-destructive">
              <HugeiconsIcon icon={AlertCircleIcon} className="size-4" />
              {evals.modelsError}
            </div>
          )}

          {!evals.modelsError && (
            <>
              <Field label={t("benchmark.taskLabel")}>
                {loadingTasks ? (
                  <span className="flex h-9 items-center gap-2 text-ui-12 text-muted-foreground">
                    <Spinner className="size-3.5" />
                    {t("benchmark.loadingTasks")}
                  </span>
                ) : (
                  <div ref={taskAnchorRef}>
                    <Combobox
                      items={taskItems}
                      filteredItems={taskItems}
                      filter={null}
                      value={selectedTask}
                      onValueChange={(next) => {
                        if (next) setSelectedTask(next);
                      }}
                      onInputValueChange={(next) => {
                        if (selectingTaskRef.current) {
                          selectingTaskRef.current = false;
                          return;
                        }
                        setTaskSearch(next);
                      }}
                      itemToStringValue={(item) => {
                        const info = tasks.find((x) => x.id === item);
                        return info ? info.name : item;
                      }}
                      autoHighlight={true}
                    >
                      <ComboboxInput
                        placeholder={t("benchmark.searchTaskPlaceholder")}
                        className="w-full"
                        showClear={true}
                      />
                      <ComboboxContent anchor={taskAnchorRef}>
                        <ComboboxEmpty>{t("benchmark.noTasksFound")}</ComboboxEmpty>
                        <ComboboxList>
                          {taskItems.map((id) => {
                            const info = tasks.find((x) => x.id === id);
                            const tl = info?.task_type;
                            const typeLabel = !tl
                              ? null
                              : tl.includes("log")
                                ? "log likelihood"
                                : tl === "greedy_until"
                                  ? "generation"
                                  : tl;
                            return (
                              <ComboboxItem
                                key={id}
                                value={id}
                                onPointerDown={() => {
                                  selectingTaskRef.current = true;
                                }}
                              >
                                <span className="truncate font-medium">
                                  {info?.name ?? id}
                                </span>
                                {typeLabel && (
                                  <span className="ml-2 shrink-0 text-xs text-muted-foreground">
                                    {typeLabel}
                                  </span>
                                )}
                              </ComboboxItem>
                            );
                          })}
                        </ComboboxList>
                      </ComboboxContent>
                    </Combobox>
                  </div>
                )}
              </Field>

              <div className="grid grid-cols-2 gap-2">
                <Field label={t("benchmark.numFewshotLabel")}>
                  {fewshotSupported ? (
                    <CountInput
                      value={numFewshot}
                      min={0}
                      max={20}
                      step={1}
                      onCommit={setNumFewshot}
                    />
                  ) : (
                    <span className="flex h-9 items-center text-ui-11 text-muted-foreground/60">
                      {t("benchmark.fewshotNotSupported")}
                    </span>
                  )}
                </Field>
                <Field label={t("benchmark.batchSizeLabel")}>
                  <Input
                    type="number"
                    value={batchSize > 0 ? String(batchSize) : ""}
                    onChange={(e) => {
                      const v = e.target.value;
                      setBatchSize(v === "" ? 0 : Math.max(0, Number(v) || 0));
                    }}
                    placeholder="auto"
                    min={0}
                    max={512}
                    step={1}
                    className="text-center font-mono tabular-nums"
                  />
                </Field>
              </div>

              <Field label={t("benchmark.maxTokensLabel")}>
                <CountInput
                  value={maxTokens}
                  min={1024}
                  max={262144}
                  step={1024}
                  onCommit={setMaxTokens}
                />
              </Field>

              {!showPanel && (
                <Button
                  size="lg"
                  className={RUN_BUTTON}
                  disabled={
                    !evals.selectedModel ||
                    modelLoading ||
                    !!evals.loadingModel ||
                    evals.downloadingForBenchmark
                  }
                  onClick={handleStartAndOpenPanel}
                >
                  <HugeiconsIcon
                    icon={Rocket01Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                  {evals.loadingModel
                    ? t("benchmark.loadingModel")
                    : evals.downloadingForBenchmark
                      ? t("benchmark.downloadingModel")
                      : t("benchmark.runButton")}
                </Button>
              )}
            </>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          {showPanel && (
            <div>
              <BenchmarkRunPanel task={selectedTask} onClose={handleClosePanel} />
              <div ref={panelEndRef} aria-hidden="true" className="h-px w-full" />
            </div>
          )}
          <EvalScoreboard />
          <section className={cn(BENCH_CARD, "p-4 sm:p-5")}>
            <BenchmarkHistoryPanel />
          </section>
        </div>
      </div>
    </div>
  );
}
