// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { checkDatasetFormat } from "@/features/training/api/datasets-api";
import { listMetrics, type EvalStartRequest, type MetricInfo } from "../api/eval-api";
import { MetricConfigFields } from "./metric-config-fields";
import { EvalModelFields } from "./eval-model-fields";

export function EvalConfigForm({
  disabled,
  onStart,
}: {
  disabled: boolean;
  onStart: (payload: EvalStartRequest) => void;
}) {
  const [modelIdentifier, setModelIdentifier] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [datasetIsLocal, setDatasetIsLocal] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [split, setSplit] = useState("train");
  const [subset, setSubset] = useState("");
  const [inputColumn, setInputColumn] = useState("");
  const [referenceColumn, setReferenceColumn] = useState("");
  const [detectedColumns, setDetectedColumns] = useState<string[]>([]);
  const [previewSample, setPreviewSample] = useState<Record<string, unknown> | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [template, setTemplate] = useState("");
  const [metrics, setMetrics] = useState<MetricInfo[]>([]);
  const [metricName, setMetricName] = useState("");
  const [metricConfig, setMetricConfig] = useState<Record<string, unknown>>({});
  const [runAll, setRunAll] = useState(false);
  const [limit, setLimit] = useState(100);
  const [maxNewTokens, setMaxNewTokens] = useState(256);
  const [temperature, setTemperature] = useState(0);
  const [formError, setFormError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  // On mount: load metrics
  useEffect(() => {
    listMetrics()
      .then((loaded) => {
        setMetrics(loaded);
        if (loaded.length > 0) {
          // The [metricName, metrics] effect reseeds metricConfig once
          // both metrics and metricName are set, so no need to seed here.
          setMetricName(loaded[0].name);
        }
      })
      .catch((err) => {
        setMetricsError(err instanceof Error ? err.message : "Failed to load metrics.");
      });
  }, []);

  // When metricName changes, reseed config defaults
  useEffect(() => {
    if (metrics.length === 0 || !metricName) return;
    const m = metrics.find((x) => x.name === metricName);
    if (!m) return;
    const defaults: Record<string, unknown> = {};
    for (const f of m.config_fields) {
      defaults[f.name] = f.default;
    }
    setMetricConfig(defaults);
  }, [metricName, metrics]);

  const selectedMetric = metrics.find((m) => m.name === metricName) ?? null;

  async function detectColumns() {
    if (!datasetName.trim()) {
      setDetectError("Enter a dataset name or path first.");
      return;
    }
    setDetecting(true);
    setDetectError(null);
    try {
      const res = await checkDatasetFormat({
        datasetName: datasetName.trim(),
        hfToken: hfToken || null,
        subset: subset || null,
        split: split || "train",
      });
      setDetectedColumns(res.columns ?? []);
      setPreviewSample(res.preview_samples?.[0] ?? null);
      if (res.columns?.length) {
        if (!inputColumn) setInputColumn(res.columns[0]);
        if (!referenceColumn) setReferenceColumn(res.columns[1] ?? res.columns[0]);
      }
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetecting(false);
    }
  }

  function buildPayload(): EvalStartRequest {
    if (!modelIdentifier.trim()) throw new Error("Model identifier is required.");
    if (!datasetName.trim()) throw new Error("Dataset name or path is required.");
    if (!inputColumn.trim()) throw new Error("Input column is required.");
    if (!referenceColumn.trim()) throw new Error("Reference column is required.");
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
        is_local: datasetIsLocal,
        name: datasetIsLocal ? null : datasetName.trim(),
        path: datasetIsLocal ? datasetName.trim() : null,
        split: split.trim() || "train",
        subset: subset.trim() || null,
      },
      input_column: inputColumn.trim(),
      reference_column: referenceColumn.trim(),
      metric_name: metricName,
      metric_config: parsedMetricConfig,
      system_prompt: systemPrompt,
      template: template.trim() ? template : null,
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
    <div className="flex flex-col gap-5">
      {/* 1. Model */}
      <Card>
        <CardHeader>
          <CardTitle>Model</CardTitle>
        </CardHeader>
        <CardContent>
          <EvalModelFields
            modelIdentifier={modelIdentifier}
            onModelChange={setModelIdentifier}
            hfToken={hfToken}
            onHfTokenChange={setHfToken}
          />
        </CardContent>
      </Card>

      {/* 2. Dataset */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <Switch
              id="ecf-dataset-local"
              checked={datasetIsLocal}
              onCheckedChange={setDatasetIsLocal}
            />
            <Label htmlFor="ecf-dataset-local">
              {datasetIsLocal ? "Local file" : "Hugging Face"}
            </Label>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-dataset-name">Dataset</Label>
            <Input
              id="ecf-dataset-name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder={datasetIsLocal ? "/path/to/data.jsonl" : "org/dataset-name"}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-split">Split</Label>
            <Input
              id="ecf-split"
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              placeholder="train"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-subset">Subset (optional)</Label>
            <Input
              id="ecf-subset"
              value={subset}
              onChange={(e) => setSubset(e.target.value)}
              placeholder="default"
            />
          </div>
          <Button
            variant="secondary"
            onClick={detectColumns}
            disabled={detecting}
            type="button"
          >
            {detecting ? "Detecting…" : "Detect columns"}
          </Button>
          {detectError && (
            <p className="text-sm text-red-500">{detectError}</p>
          )}
          {detectedColumns.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {detectedColumns.map((col) => (
                <Badge
                  key={col}
                  variant="secondary"
                  className="cursor-pointer"
                  onClick={() => {
                    if (!inputColumn) {
                      setInputColumn(col);
                    } else {
                      setReferenceColumn(col);
                    }
                  }}
                >
                  {col}
                </Badge>
              ))}
            </div>
          )}
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-input-col">Input column</Label>
            <Input
              id="ecf-input-col"
              value={inputColumn}
              onChange={(e) => setInputColumn(e.target.value)}
              placeholder="input"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-ref-col">Reference column</Label>
            <Input
              id="ecf-ref-col"
              value={referenceColumn}
              onChange={(e) => setReferenceColumn(e.target.value)}
              placeholder="output"
            />
          </div>
          {previewSample && (
            <pre className="max-h-40 overflow-auto rounded-md bg-muted p-2 font-mono text-xs">
              {JSON.stringify(previewSample, null, 2)}
            </pre>
          )}
        </CardContent>
      </Card>

      {/* 3. Prompt */}
      <Card>
        <CardHeader>
          <CardTitle>Prompt</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-system-prompt">System prompt (optional)</Label>
            <Textarea
              id="ecf-system-prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              className="min-h-24 font-mono text-xs"
              spellCheck={false}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-template">Template</Label>
            <Textarea
              id="ecf-template"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              className="min-h-24 font-mono text-xs"
              spellCheck={false}
            />
            <p className="text-xs text-muted-foreground">
              Use {"{input}"} as the column placeholder. Leave blank to send the raw input column.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* 4. Metric */}
      <Card>
        <CardHeader>
          <CardTitle>Metric</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ecf-metric">Metric</Label>
            <Select value={metricName} onValueChange={setMetricName}>
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
            {metricsError && (
              <p className="text-sm text-red-500">{metricsError}</p>
            )}
          </div>
          {selectedMetric && (
            <MetricConfigFields
              fields={selectedMetric.config_fields}
              values={metricConfig}
              onChange={setMetricConfig}
            />
          )}
        </CardContent>
      </Card>

      {/* 5. Run size & generation */}
      <Card>
        <CardHeader>
          <CardTitle>Run size &amp; generation</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
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
        </CardContent>
      </Card>

      {/* 6. Footer */}
      <div className="flex items-center gap-3">
        <Button onClick={handleSubmit} disabled={disabled} type="button">
          Run eval
        </Button>
        {disabled && (
          <span className="text-sm text-muted-foreground">
            An eval is already running.
          </span>
        )}
        {formError && (
          <span className="text-sm text-red-500">{formError}</span>
        )}
      </div>
    </div>
  );
}
