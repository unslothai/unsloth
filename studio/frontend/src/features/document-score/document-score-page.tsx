// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  scoreDocument,
  type ScoreNode,
  type ScoreResult,
} from "./api/document-score-api";

const EXAMPLE_GROUND_TRUTH = JSON.stringify(
  {
    total: 100,
    currency: "USD",
    issue_date: "2024-01-15",
    vendor: { name: "Acme Inc", vat_id: "X1" },
    line_items: [
      { desc: "Apple", qty: 1, price: 40 },
      { desc: "Banana", qty: 2, price: 60 },
    ],
  },
  null,
  2,
);

const EXAMPLE_PREDICTION = JSON.stringify(
  {
    total: 90,
    currency: "EUR",
    issue_date: "Jan 15, 2024",
    vendor: { name: "Acme Inc", vat_id: "X1" },
    line_items: [
      { desc: "Apple", qty: 1, price: 40 },
      { desc: "Banana", qty: 2, price: 60 },
    ],
  },
  null,
  2,
);

const EXAMPLE_SCHEMA = JSON.stringify(
  {
    total: { type: "money" },
    currency: "categorical",
    issue_date: { type: "date" },
    vendor: { name: "string", vat_id: "categorical" },
    line_items: [{ desc: "string", qty: "numeric", price: "money" }],
  },
  null,
  2,
);

function scoreColor(score: number): string {
  if (score >= 0.999) return "text-emerald-500";
  if (score >= 0.5) return "text-amber-500";
  return "text-red-500";
}

function parseField(label: string, raw: string, allowEmpty: boolean): unknown {
  const trimmed = raw.trim();
  if (!trimmed) {
    if (allowEmpty) return undefined;
    throw new Error(`${label} is empty.`);
  }
  try {
    return JSON.parse(trimmed);
  } catch (e) {
    throw new Error(`${label} is not valid JSON: ${(e as Error).message}`);
  }
}

/** Recursive per-field breakdown row. */
function BreakdownTree({
  label,
  node,
  depth,
}: {
  label: string;
  node: ScoreNode;
  depth: number;
}) {
  const children = node.children;
  const entries: [string, ScoreNode][] = Array.isArray(children)
    ? children.map((c, i) => [`[${i}]`, c])
    : children
      ? Object.entries(children)
      : [];

  return (
    <div className="text-sm">
      <div
        className="flex items-center justify-between gap-3 py-1"
        style={{ paddingLeft: `${depth * 16}px` }}
      >
        <span className="truncate font-medium text-muted-foreground">
          {label}
          {node.note ? (
            <span className="ml-2 text-xs text-red-500/80">({node.note})</span>
          ) : null}
        </span>
        <span className={cn("tabular-nums font-semibold", scoreColor(node.score))}>
          {node.score.toFixed(3)}
        </span>
      </div>
      {entries.map(([k, child]) => (
        <BreakdownTree key={k} label={k} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}

export function DocumentScorePage() {
  const [groundTruth, setGroundTruth] = useState(EXAMPLE_GROUND_TRUTH);
  const [prediction, setPrediction] = useState(EXAMPLE_PREDICTION);
  const [schema, setSchema] = useState(EXAMPLE_SCHEMA);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleScore() {
    setError(null);
    setResult(null);
    let parsed: { gt: unknown; pred: unknown; schema: unknown };
    try {
      parsed = {
        gt: parseField("Ground truth", groundTruth, false),
        pred: parseField("Prediction", prediction, false),
        schema: parseField("Schema", schema, true),
      };
    } catch (e) {
      setError((e as Error).message);
      return;
    }
    setLoading(true);
    try {
      const res = await scoreDocument({
        ground_truth: parsed.gt,
        prediction: parsed.pred,
        schema: parsed.schema,
      });
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 p-6 overflow-y-auto">
      <div className="flex flex-col gap-1">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Document Score
        </h1>
        <p className="text-sm text-muted-foreground">
          Type-aware JSON document scoring — each field is compared with the rule
          its schema assigns (money, categorical, string/ANLS, date), then
          aggregated with ANLS*-style nesting and list matching.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="flex flex-col gap-2">
          <Label htmlFor="ds-gt">Ground truth (JSON)</Label>
          <Textarea
            id="ds-gt"
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            className="min-h-64 font-mono text-xs"
            spellCheck={false}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="ds-pred">Prediction (JSON)</Label>
          <Textarea
            id="ds-pred"
            value={prediction}
            onChange={(e) => setPrediction(e.target.value)}
            className="min-h-64 font-mono text-xs"
            spellCheck={false}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="ds-schema">Schema (JSON, optional)</Label>
          <Textarea
            id="ds-schema"
            value={schema}
            onChange={(e) => setSchema(e.target.value)}
            className="min-h-64 font-mono text-xs"
            spellCheck={false}
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Button onClick={handleScore} disabled={loading}>
          {loading ? "Scoring…" : "Score document"}
        </Button>
        {error ? <span className="text-sm text-red-500">{error}</span> : null}
      </div>

      {result ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-baseline justify-between">
              <span>Document score</span>
              <span
                className={cn(
                  "text-3xl tabular-nums font-bold",
                  scoreColor(result.score),
                )}
              >
                {result.score.toFixed(4)}
              </span>
            </CardTitle>
            <CardDescription>
              Per-field breakdown — green is a match, red dragged the score down.
            </CardDescription>
          </CardHeader>
          {result.breakdown ? (
            <CardContent>
              <BreakdownTree label="document" node={result.breakdown} depth={0} />
            </CardContent>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
