// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { ActivationMetadata, ActivationRecord } from "@/features/training/api/train-api";
import { type ReactElement, useMemo, useState } from "react";
import katex from "katex";
import { computeLayerStats, computeOverlaySets, DEAD_THRESHOLD } from "./activation-stats";

// ── KaTeX helpers ─────────────────────────────────────────────────────────────

function KatexInline({ latex }: { latex: string }): ReactElement {
  return (
    <span
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(latex, { throwOnError: false, displayMode: false }),
      }}
    />
  );
}

function KatexDisplay({ latex }: { latex: string }): ReactElement {
  return (
    <span
      className="block my-1 overflow-x-auto"
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(latex, { throwOnError: false, displayMode: true }),
      }}
    />
  );
}

// ── Info dialog (exported so heatmap can reuse it) ────────────────────────────

interface InfoDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function InterpretabilityInfoDialog({ open, onOpenChange }: InfoDialogProps): ReactElement {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[min(92vw,900px)] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="font-bold">How to read Interpretability</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-6 text-sm mt-2">

          {/* Heatmap */}
          <section className="flex flex-col gap-2">
            <h3 className="font-semibold text-foreground">Neuron Activations heatmap</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Each cell is one neuron channel in one layer. The color encodes the mean absolute
              activation for that channel at the current training step. Bright green = strongly
              firing. Dark/dim = barely firing. Red = outlier above the 99th percentile.
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              The color scale ceiling is set to <strong>p99</strong> of all values so a small
              number of very large activations don&apos;t wash out everything else.
            </p>
          </section>

          {/* View modes */}
          <section className="flex flex-col gap-3">
            <h3 className="font-semibold text-foreground">View modes</h3>
            <p className="text-xs text-muted-foreground">
              Swap what the heatmap encodes. Only one view is active at a time.
            </p>
            {[
              {
                label: "Activations (default)",
                latex: "\\bar{a}_c \\equiv \\frac{1}{N}\\sum_{n=1}^{N}|x_{n,c}| \\;\\equiv\\; \\texttt{mean\\_abs}[c]",
                desc: "N = batch × seq tokens, x_{n,c} = activation at token n, channel c. This is the mean_abs value stored in the activation log — the average absolute magnitude per channel.",
              },
              {
                label: "Gradients",
                latex: "g \\equiv \\|\\nabla_{W}L\\|_2 = \\sqrt{\\sum_i \\left(\\frac{\\partial L}{\\partial w_i}\\right)^2}",
                desc: "Per-layer gradient norm. Near zero in early layers → vanishing gradients. Very large → exploding gradients.",
              },
              {
                label: "LoRA norms",
                latex: "\\|\\Delta W\\|_F \\equiv \\sqrt{\\sum_{i,j}(BA)_{ij}^2}",
                desc: "B, A = LoRA matrices, ΔW = BA. Shows how much each layer's adapter has moved. Low → adapter barely adapting.",
              },
              {
                label: "Delta",
                latex: "\\Delta \\equiv |\\bar{a}_t - \\bar{a}_{t-1}|",
                desc: "āₜ = mean_abs at current step. Absolute change from the previous captured step. High → actively changing right now.",
              },
              {
                label: "Trend",
                latex: "\\beta \\equiv \\frac{\\sum(t_i-\\bar{t})(x_i-\\bar{x})}{\\sum(t_i-\\bar{t})^2}",
                desc: "Linear regression slope per channel across all captured steps. |β| shown — large → rapidly changing over training.",
              },
            ].map(({ label, latex, desc }) => (
              <div key={label} className="flex flex-col gap-0.5 border-l-2 border-border/60 pl-3">
                <p className="text-xs font-medium text-foreground">{label}</p>
                <KatexDisplay latex={latex} />
                <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            ))}
          </section>

          {/* Overlays */}
          <section className="flex flex-col gap-3">
            <h3 className="font-semibold text-foreground">Overlays</h3>
            <p className="text-xs text-muted-foreground">
              Drawn on top of whichever view is active. Multiple overlays can be enabled at once.
            </p>
            {[
              {
                label: "Dead",
                color: "rgb(96,165,250)",
                latex: "\\max_t(\\bar{a}_t) < \\varepsilon, \\quad \\varepsilon = 0.01",
                desc: "Channel never exceeded ε across all captured steps. Dead neurons contribute nothing to model output. May indicate too-high learning rate or poor initialisation.",
              },
              {
                label: "Constant",
                color: "rgb(251,146,60)",
                latex: "CV \\equiv \\frac{\\sigma}{\\mu} < 0.05, \\quad \\sigma = \\sqrt{\\frac{\\sum(\\bar{a}_t - \\mu)^2}{N}}",
                desc: "Neuron fires but its output barely changes over training (CV < 5%). Limits the model's expressive capacity.",
              },
              {
                label: "Onset Dead",
                color: "rgb(147,51,234)",
                latex: "\\bar{a}_0 \\geq \\varepsilon \\;\\wedge\\; \\max_{t>0}(\\bar{a}_t) < \\varepsilon",
                desc: "Neuron was alive at step 0 but died mid-training. This is the critical red flag — dying neurons suggest an unstable learning rate or a corrupted batch.",
              },
            ].map(({ label, color, latex, desc }) => (
              <div key={label} className="flex flex-col gap-0.5 border-l-2 pl-3" style={{ borderColor: color }}>
                <p className="text-xs font-medium" style={{ color }}>{label}</p>
                <KatexDisplay latex={latex} />
                <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            ))}
          </section>

          {/* Trend chart */}
          <section className="flex flex-col gap-2">
            <h3 className="font-semibold text-foreground">Neuron Health Trend chart</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Shows how the average dead% and constant% evolve over training steps.
              The dashed vertical line tracks the current replay position.
              Click anywhere on the chart to scrub to that step.
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              A flat dead% line means neurons died at initialisation (possibly expected).
              A rising dead% line mid-training is the concerning case — something is killing neurons.
            </p>
          </section>

          {/* Diagnostics */}
          <section className="flex flex-col gap-2">
            <h3 className="font-semibold text-foreground">Diagnostics panel</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Automatically scans activation records and surfaces notable findings.
              Red = likely problem. Orange = warning worth investigating. Grey = informational.
              Click a finding to jump to the step where it was first detected.
            </p>
          </section>

        </div>
      </DialogContent>
    </Dialog>
  );
}

// ── Finding computation ───────────────────────────────────────────────────────

type Severity = "red" | "orange" | "grey";

interface Finding {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  stepIndex?: number;
  layers?: string[];
}

function computeFindings(
  records: ActivationRecord[],
  stepIndex: number,
  metadata: ActivationMetadata | null,
): Finding[] {
  const findings: Finding[] = [];
  if (records.length === 0) return findings;

  const slice = records.slice(0, stepIndex + 1);
  const stats = computeLayerStats(slice);
  const overlays = computeOverlaySets(records, stepIndex);

  // ── Onset-dead neurons ────────────────────────────────────────────────────
  if (overlays.onsetDead.size > 0) {
    // Count per layer
    const layerCounts = new Map<string, number>();
    for (const id of overlays.onsetDead) {
      const [key] = id.split(":");
      layerCounts.set(key, (layerCounts.get(key) ?? 0) + 1);
    }
    const sorted = [...layerCounts.entries()].sort((a, b) => b[1] - a[1]);
    const topLayers = sorted.slice(0, 3).map(([k]) => `L${k}`);
    const total = [...layerCounts.values()].reduce((s, v) => s + v, 0);

    // Find first step where onset-dead appeared
    let firstOnsetStep: number | undefined;
    for (let si = 1; si < slice.length; si++) {
      const prev = computeOverlaySets(records, si - 1);
      const curr = computeOverlaySets(records, si);
      if (curr.onsetDead.size > prev.onsetDead.size) {
        firstOnsetStep = si;
        break;
      }
    }

    findings.push({
      id: "onset_dead",
      severity: "red",
      title: "Onset-dead neurons detected",
      detail: `${total} channel${total !== 1 ? "s" : ""} were alive at step 0 but died mid-training. Check learning rate and batch quality.`,
      stepIndex: firstOnsetStep,
      layers: topLayers,
    });
  }

  // ── Dead % spike ──────────────────────────────────────────────────────────
  if (slice.length >= 2) {
    let spikeStep: number | undefined;
    let spikeLayer: string | undefined;
    let spikeDelta = 0;

    for (let si = 1; si < slice.length; si++) {
      const prev = computeLayerStats(records.slice(0, si));
      const curr = computeLayerStats(records.slice(0, si + 1));
      for (let li = 0; li < curr.length; li++) {
        const delta = curr[li].deadPct - (prev[li]?.deadPct ?? 0);
        if (delta > 10 && delta > spikeDelta) {
          spikeDelta = delta;
          spikeStep = si;
          spikeLayer = curr[li].layer;
        }
      }
    }

    if (spikeStep !== undefined) {
      findings.push({
        id: "dead_spike",
        severity: "red",
        title: "Dead neuron spike",
        detail: `${spikeDelta.toFixed(1)}% of channels in ${spikeLayer} died in a single step. Check for batch anomalies or a learning rate spike.`,
        stepIndex: spikeStep,
        layers: spikeLayer ? [spikeLayer] : undefined,
      });
    }
  }

  // ── High dead rate ────────────────────────────────────────────────────────
  const avgDead = stats.length > 0 ? stats.reduce((s, l) => s + l.deadPct, 0) / stats.length : 0;
  if (avgDead > 20) {
    const badLayers = stats.filter((l) => l.deadPct > 30).map((l) => l.layer);
    findings.push({
      id: "high_dead",
      severity: avgDead > 40 ? "red" : "orange",
      title: `High dead neuron rate (avg ${avgDead.toFixed(1)}%)`,
      detail: "A large fraction of channels are permanently inactive. Consider lowering the learning rate or checking weight initialisation.",
      layers: badLayers.slice(0, 4),
    });
  }

  // ── High constant rate ────────────────────────────────────────────────────
  const avgConst = stats.length > 0 ? stats.reduce((s, l) => s + l.constantPct, 0) / stats.length : 0;
  if (avgConst > 35) {
    const badLayers = stats.filter((l) => l.constantPct > 50).map((l) => l.layer);
    findings.push({
      id: "high_constant",
      severity: "orange",
      title: `High constant neuron rate (avg ${avgConst.toFixed(1)}%)`,
      detail: "Many channels fire but never change over training. These neurons carry no dynamic information, limiting model expressivity.",
      layers: badLayers.slice(0, 4),
    });
  }

  // ── Born-dead neurons (informational) ─────────────────────────────────────
  const bornDead = [...overlays.dead].filter((id) => !overlays.onsetDead.has(id));
  if (bornDead.length > 0 && !findings.some((f) => f.id === "onset_dead")) {
    const layerCounts = new Map<string, number>();
    for (const id of bornDead) {
      const [key] = id.split(":");
      layerCounts.set(key, (layerCounts.get(key) ?? 0) + 1);
    }
    findings.push({
      id: "born_dead",
      severity: "grey",
      title: `Born-dead neurons (${bornDead.length} channel${bornDead.length !== 1 ? "s" : ""})`,
      detail: `${layerCounts.size} layer${layerCounts.size !== 1 ? "s" : ""} had channels inactive from step 0. This can be expected for some architectures — watch for the count increasing over training.`,
    });
  }

  // ── Capture settings suggestions ─────────────────────────────────────────
  if (metadata && !metadata.capture_gradients) {
    findings.push({
      id: "no_gradients",
      severity: "grey",
      title: "Gradient capture disabled",
      detail: "Enable capture_gradients to unlock the Gradients view mode and richer diagnostics.",
    });
  }
  if (metadata && !metadata.capture_lora_norms) {
    findings.push({
      id: "no_lora",
      severity: "grey",
      title: "LoRA norm capture disabled",
      detail: "Enable capture_lora_norms to unlock the LoRA Norms view and adapter health tracking.",
    });
  }

  // Sort: red → orange → grey, cap at 8
  const order: Record<Severity, number> = { red: 0, orange: 1, grey: 2 };
  return findings.sort((a, b) => order[a.severity] - order[b.severity]).slice(0, 8);
}

// ── Severity styles ───────────────────────────────────────────────────────────

const SEVERITY_STYLES: Record<Severity, { bar: string; dot: string; title: string }> = {
  red:    { bar: "bg-red-500",    dot: "bg-red-500",    title: "text-red-500" },
  orange: { bar: "bg-amber-500",  dot: "bg-amber-500",  title: "text-amber-500" },
  grey:   { bar: "bg-border",     dot: "bg-muted-foreground/40", title: "text-muted-foreground" },
};

// ── DiagnosticsPanel ──────────────────────────────────────────────────────────

export interface DiagnosticsPanelProps {
  records: ActivationRecord[];
  stepIndex: number;
  metadata: ActivationMetadata | null;
  onStepChange: (idx: number) => void;
}

export function DiagnosticsPanel({
  records,
  stepIndex,
  metadata,
  onStepChange,
}: DiagnosticsPanelProps): ReactElement {
  const [infoOpen, setInfoOpen] = useState(false);

  const findings = useMemo(
    () => computeFindings(records, stepIndex, metadata),
    [records, stepIndex, metadata],
  );

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center justify-between">
            Diagnostics
            <button
              type="button"
              onClick={() => setInfoOpen(true)}
              className="rounded p-1 text-muted-foreground opacity-40 transition-opacity hover:opacity-100 hover:bg-muted/60 hover:text-foreground focus:opacity-100 text-xs font-normal leading-none"
              title="How to read interpretability"
              aria-label="Open interpretability guide"
            >
              <KatexInline latex="?" />
            </button>
          </CardTitle>
        </CardHeader>

        <CardContent className="pt-0">
          {findings.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
              <svg viewBox="0 0 16 16" fill="none" className="h-3.5 w-3.5 shrink-0 text-emerald-500" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13.5 4.5L6 12 2.5 8.5" />
              </svg>
              No issues detected
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {findings.map((f) => {
                const styles = SEVERITY_STYLES[f.severity];
                const clickable = f.stepIndex !== undefined;
                return (
                  <button
                    key={f.id}
                    type="button"
                    disabled={!clickable}
                    onClick={() => clickable && onStepChange(f.stepIndex!)}
                    className={cn(
                      "flex items-start gap-2.5 rounded-md border border-border/40 bg-muted/20 px-3 py-2 text-left transition-colors",
                      clickable
                        ? "cursor-pointer hover:bg-muted/40 hover:border-border/60"
                        : "cursor-default",
                    )}
                  >
                    {/* Severity bar */}
                    <div className={cn("mt-0.5 h-full w-0.5 rounded-full shrink-0 self-stretch min-h-[1rem]", styles.bar)} />

                    <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className={cn("text-xs font-medium leading-tight", styles.title)}>
                          {f.title}
                        </span>
                        {f.stepIndex !== undefined && (
                          <span className="text-[10px] text-muted-foreground/60 tabular-nums">
                            step {records[f.stepIndex]?.step ?? f.stepIndex}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-muted-foreground leading-relaxed">
                        {f.detail}
                      </p>
                      {f.layers && f.layers.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-0.5">
                          {f.layers.map((l) => (
                            <span
                              key={l}
                              className="rounded border border-border/50 px-1.5 py-px text-[10px] font-mono text-muted-foreground"
                            >
                              {l}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {clickable && (
                      <svg viewBox="0 0 16 16" fill="none" className="h-3 w-3 shrink-0 mt-0.5 text-muted-foreground/40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M6 3l5 5-5 5" />
                      </svg>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      <InterpretabilityInfoDialog open={infoOpen} onOpenChange={setInfoOpen} />
    </>
  );
}

