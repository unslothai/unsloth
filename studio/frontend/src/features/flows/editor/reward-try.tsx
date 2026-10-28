// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { authFetch } from "@/features/auth";
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { useEffect, useState } from "react";
import { useDocStore } from "../stores/doc-store";
import type { FlowNode } from "../types";
import { rewardMax } from "./train-status";

const PROMPT_NAMES = ["prompt", "question", "instruction", "problem", "query", "input"];
const ANSWER_NAMES = ["answer", "solution", "final_answer", "target", "label"];

interface Picked {
  name: string;
  weight: number;
}
interface Score {
  name: string;
  score: number;
  weighted: number;
}

function pick(row: Record<string, unknown>, named: unknown, names: string[]) {
  const key =
    String(named ?? "").trim() ||
    Object.keys(row).find((k) => names.includes(k.toLowerCase()));
  return key && row[key] !== undefined ? String(row[key]) : "";
}

/** The column mapped to `role` in the node's own column mapping, if any. */
function mapped(params: Record<string, unknown>, role: string): string | undefined {
  const mapping = params.custom_format_mapping;
  if (!mapping || typeof mapping !== "object") return undefined;
  return Object.entries(mapping as Record<string, unknown>).find(([, r]) => r === role)?.[0];
}

function sampleCompletion(reference: string): string {
  const final = reference.split("####").pop()?.trim() ?? "";
  return `<reasoning>\n...\n</reasoning>\n<answer>\n${final}\n</answer>`;
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.detail ?? `HTTP ${res.status}`);
  return body as T;
}

/** "Try the rewards" on row 1 of the Hub dataset wired into Fine-tune, scored by the server. */
export function RewardTryPanel({ node }: { node: FlowNode }) {
  const source = useDocStore((s) => {
    const edge = Object.values(s.graph.edges).find(
      (e) => e.target === node.id && e.target_port === "dataset",
    );
    const from = edge ? s.graph.nodes[edge.source] : undefined;
    return from?.type === "data.hf_dataset" ? from : undefined;
  });
  const repo = String(source?.params.repo ?? "").trim();
  const subset = String(source?.params.subset ?? "");
  const split = String(source?.params.split ?? "train");
  const [row, setRow] = useState<Record<string, unknown> | null>(null);
  const [prompt, setPrompt] = useState("");
  const [reference, setReference] = useState("");
  const [completion, setCompletion] = useState("");
  const [scores, setScores] = useState<{ rows: Score[]; total: number; max: number | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const saved = node.params.rewards ?? [
    { name: "exact-answer", weight: 1 },
    { name: "soft-xml-format", weight: 1 },
  ];
  const rewards = (Array.isArray(saved) ? (saved as Picked[]) : []).filter(
    (r) => r.name?.trim(),
  );

  useEffect(() => {
    if (!repo) return;
    let alive = true;
    void authFetch("/api/hub/datasets/check-format", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name: repo, subset: subset || undefined, train_split: split }),
    })
      .then((res) => json<{ preview_samples?: Record<string, unknown>[] }>(res))
      .then((found) => {
        const first = found.preview_samples?.[0];
        if (!alive || !first) return;
        const params = node.params;
        const answer = pick(first, params.answer_column || mapped(params, "answer"), ANSWER_NAMES);
        setRow(first);
        setPrompt(pick(first, params.prompt_column || mapped(params, "prompt"), PROMPT_NAMES));
        setReference(answer);
        setCompletion((c) => c || sampleCompletion(answer));
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [repo, subset, split, node.params]);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const library = await authFetch("/api/rewards").then((res) =>
        json<{ name: string; rule: Record<string, unknown> | null }[]>(res),
      );
      const rules = Object.fromEntries(library.map((r) => [r.name, r.rule]));
      // Each reward is scored against the column it compares to, as in training; "answer" is the role.
      const rows = await Promise.all(
        rewards.map((reward) => {
          const column = rules[reward.name]?.compare_to;
          const own =
            typeof column === "string" && column !== "answer" && row?.[column] !== undefined
              ? String(row[column])
              : reference;
          return authFetch("/api/rewards/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rewards: [reward], completion, reference: own || null }),
          })
            .then((res) => json<{ scores: Score[] }>(res))
            .then((res) => res.scores[0]);
        }),
      );
      const total = rows.reduce((sum, r) => sum + r.weighted, 0);
      setScores({ rows, total, max: rewardMax(rewards, rules) });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="flex flex-col gap-2 border-border/60 border-t px-4 py-3" data-reward-try="">
      <h3 className="text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
        Try the rewards
      </h3>
      <p className="text-ui-11 leading-snug text-muted-foreground">
        {repo
          ? `Row 1 of ${repo}. Edit the reply to see what each reward gives it.`
          : "Wire a Hugging Face dataset in to start from its row 1, or type an answer."}
      </p>
      {prompt && (
        <p className="line-clamp-3 rounded-md bg-muted/40 px-2 py-1 text-ui-11 text-foreground">
          {prompt}
        </p>
      )}
      <FieldLabel label="Dataset answer" />
      <Textarea
        fieldSizing="fixed"
        rows={2}
        value={reference}
        onChange={(e) => setReference(e.target.value)}
        className="text-[13px] leading-snug"
      />
      <FieldLabel label="A model reply" />
      <Textarea
        fieldSizing="fixed"
        rows={5}
        value={completion}
        onChange={(e) => setCompletion(e.target.value)}
        className="text-[13px] leading-snug"
      />
      <Button
        type="button"
        size="xs"
        variant="outline"
        className="self-start"
        disabled={busy || !rewards.length || !completion.trim()}
        onClick={() => void run()}
      >
        Score it
      </Button>
      {error && <p className="text-ui-11 text-destructive">{error}</p>}
      {scores && (
        <div className="flex flex-col gap-0.5 text-ui-11">
          {scores.rows.map((row) => (
            <div key={row.name} className="flex justify-between gap-2">
              <span className="text-muted-foreground">{row.name}</span>
              <span className="tabular-nums text-foreground">{row.weighted.toFixed(2)}</span>
            </div>
          ))}
          <div className="flex justify-between gap-2 border-border/60 border-t pt-0.5 font-medium">
            <span>Total</span>
            <span className="tabular-nums">
              {scores.total.toFixed(2)}
              {scores.max !== null && ` / ${Number(scores.max.toFixed(2))}`}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
