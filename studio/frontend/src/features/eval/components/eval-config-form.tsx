// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SectionCard } from "@/components/section-card";
import {
  ChipIcon,
  Database02Icon,
  ChartAverageIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { listMetrics, type EvalStartRequest, type MetricInfo } from "../api/eval-api";
import { MetricConfigFields } from "./metric-config-fields";
import { SchemaComparatorPreview } from "./schema-comparator-preview";
import { EvalModelFields } from "./eval-model-fields";
import { EvalDatasetFields } from "./eval-dataset-fields";
import { useEvalConfigStore } from "../stores/eval-config-store";

export function EvalConfigForm({
  disabled,
  onStart,
}: {
  disabled: boolean;
  onStart: (payload: EvalStartRequest) => void;
}) {
  // Persisted config from store
  const modelIdentifier = useEvalConfigStore((s) => s.modelIdentifier);
  const setModelIdentifier = useEvalConfigStore((s) => s.setModelIdentifier);
  const hfToken = useEvalConfigStore((s) => s.hfToken);
  const setHfToken = useEvalConfigStore((s) => s.setHfToken);
  const dataset = useEvalConfigStore((s) => s.dataset);
  const setDataset = useEvalConfigStore((s) => s.setDataset);
  const metricName = useEvalConfigStore((s) => s.metricName);
  const metricConfig = useEvalConfigStore((s) => s.metricConfig);
  const setMetricConfig = useEvalConfigStore((s) => s.setMetricConfig);
  const selectMetric = useEvalConfigStore((s) => s.selectMetric);
  const runAll = useEvalConfigStore((s) => s.runAll);
  const setRunAll = useEvalConfigStore((s) => s.setRunAll);
  const limit = useEvalConfigStore((s) => s.limit);
  const setLimit = useEvalConfigStore((s) => s.setLimit);
  const maxNewTokens = useEvalConfigStore((s) => s.maxNewTokens);
  const setMaxNewTokens = useEvalConfigStore((s) => s.setMaxNewTokens);
  const temperature = useEvalConfigStore((s) => s.temperature);
  const setTemperature = useEvalConfigStore((s) => s.setTemperature);

  // Runtime/derived state — NOT persisted
  const [metrics, setMetrics] = useState<MetricInfo[]>([]);
  const [formError, setFormError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  // On mount: load metrics. Only seed metricName/metricConfig if no metric is already persisted.
  useEffect(() => {
    listMetrics()
      .then((loaded) => {
        setMetrics(loaded);
        if (loaded.length > 0 && !metricName) {
          // First-ever use: pick the first metric and seed its defaults.
          const first = loaded[0];
          const defaults: Record<string, unknown> = {};
          for (const f of first.config_fields) {
            defaults[f.name] = f.default;
          }
          selectMetric(first.name, defaults);
        }
      })
      .catch((err) => {
        setMetricsError(err instanceof Error ? err.message : "Failed to load metrics.");
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedMetric = metrics.find((m) => m.name === metricName) ?? null;

  function buildPayload(): EvalStartRequest {
    const dsRef = dataset.isLocal ? dataset.path : dataset.name;
    if (!modelIdentifier.trim()) throw new Error("Model identifier is required.");
    if (!dsRef.trim()) throw new Error("Dataset name or path is required.");
    if (!dataset.inputColumn.trim()) throw new Error("Input column is required.");
    if (!dataset.referenceColumn.trim()) throw new Error("Reference column is required.");
    if (!metricName) throw new Error("Metric is required.");

    // Parse json-typed fields
    const parsedMetricConfig: Record<string, unknown> = { ...metricConfig };
    if (selectedMetric) {
      for (const field of selectedMetric.config_fields) {
        if (field.type === "json") {
          const val = parsedMetricConfig[field.name];
          if (typeof val === "string") {
            if (val.trim() === "") {
              delete parsedMetricConfig[field.name];
            } else {
              try {
                parsedMetricConfig[field.name] = JSON.parse(val);
              } catch (e) {
                throw new Error(
                  `${field.label} is not valid JSON: ${(e as Error).message}`,
                );
              }
            }
          }
        }
      }
    }

    return {
      model_identifier: modelIdentifier.trim(),
      dataset: {
        is_local: dataset.isLocal,
        name: dataset.isLocal ? null : dataset.name.trim(),
        path: dataset.isLocal ? dataset.path.trim() : null,
        split: dataset.split.trim() || "train",
        subset: dataset.subset.trim() || null,
      },
      input_column: dataset.inputColumn.trim(),
      reference_column: dataset.referenceColumn.trim(),
      metric_name: metricName,
      metric_config: parsedMetricConfig,
      system_prompt: "",
      template: null,
      limit: runAll ? null : Math.max(1, Math.floor(limit)),
      max_new_tokens: Math.max(1, Math.floor(maxNewTokens)),
      temperature: Number.isFinite(temperature) ? temperature : 0,
    };
  }

  function handleSubmit() {
    setFormError(null);
    try {
      const payload = buildPayload();
      onStart(payload);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="flex min-w-0 flex-col gap-4 md:gap-6">
      {/* Model — full width, featured */}
      <SectionCard
        icon={<HugeiconsIcon icon={ChipIcon} className="size-5" />}
        title="Model"
        description="Select a model to evaluate"
        accent="emerald"
        featured
      >
        <EvalModelFields
          modelIdentifier={modelIdentifier}
          onModelChange={setModelIdentifier}
          hfToken={hfToken}
          onHfTokenChange={setHfToken}
        />
      </SectionCard>

      <div className="grid min-w-0 grid-cols-1 items-start gap-4 md:grid-cols-2 md:gap-6">
        {/* Left column: Dataset */}
        <SectionCard
          icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
          title="Dataset"
          description="Pick a dataset and map input / reference columns"
          accent="indigo"
        >
          <EvalDatasetFields hfToken={hfToken} value={dataset} onChange={setDataset} />
        </SectionCard>

        {/* Right column: Metric + Generation */}
        <div className="flex min-w-0 flex-col gap-4 md:gap-6">
        {/* Metric */}
        <SectionCard
          icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
          title="Metric"
          description="Choose how outputs are scored"
          accent="orange"
        >
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="ecf-metric">Metric</Label>
              <Select
                value={metricName}
                onValueChange={(name) => {
                  const m = metrics.find((x) => x.name === name);
                  if (m) {
                    const defaults: Record<string, unknown> = {};
                    for (const f of m.config_fields) {
                      defaults[f.name] = f.default;
                    }
                    selectMetric(name, defaults);
                  }
                }}
              >
                <SelectTrigger id="ecf-metric" className="w-full">
                  <SelectValue placeholder="Select a metric…" />
                </SelectTrigger>
                <SelectContent>
                  {metrics.map((m) => (
                    <SelectItem key={m.name} value={m.name}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {selectedMetric && (
              <MetricConfigFields
                fields={selectedMetric.config_fields}
                values={metricConfig}
                onChange={setMetricConfig}
              />
            )}
            {metricName === "json_document" && (
              <SchemaComparatorPreview schemaText={metricConfig.schema} />
            )}
            {metricsError && (
              <p className="text-sm text-red-500">{metricsError}</p>
            )}
          </div>
        </SectionCard>

        {/* Generation */}
        <SectionCard
          icon={<HugeiconsIcon icon={ZapIcon} className="size-5" />}
          title="Generation"
          description="Run size & sampling parameters"
          accent="orange"
        >
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <Switch
                id="ecf-run-all"
                checked={runAll}
                onCheckedChange={setRunAll}
              />
              <Label htmlFor="ecf-run-all">Evaluate all rows</Label>
            </div>
            {!runAll && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="ecf-limit">Max rows</Label>
                <Input
                  id="ecf-limit"
                  type="number"
                  value={limit}
                  onChange={(e) => {
                    const v = e.target.value === "" ? NaN : Number(e.target.value);
                    setLimit(Number.isNaN(v) ? limit : v);
                  }}
                />
              </div>
            )}
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="ecf-max-tokens">Max new tokens</Label>
              <Input
                id="ecf-max-tokens"
                type="number"
                value={maxNewTokens}
                onChange={(e) => {
                  const v = e.target.value === "" ? NaN : Number(e.target.value);
                  setMaxNewTokens(Number.isNaN(v) ? maxNewTokens : v);
                }}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="ecf-temperature">Temperature</Label>
              <Input
                id="ecf-temperature"
                type="number"
                step="0.1"
                value={temperature}
                onChange={(e) => {
                  const v = e.target.value === "" ? NaN : Number(e.target.value);
                  setTemperature(Number.isNaN(v) ? temperature : v);
                }}
              />
              <p className="text-xs text-muted-foreground">
                0 = greedy / deterministic
              </p>
            </div>
          </div>
        </SectionCard>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center gap-3">
        <Button onClick={handleSubmit} disabled={disabled}>
          Run eval
        </Button>
        {disabled && (
          <span className="text-sm text-muted-foreground">
            An eval is already running.
          </span>
        )}
        {formError && <span className="text-sm text-red-500">{formError}</span>}
      </div>
    </div>
  );
}
