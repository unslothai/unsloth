// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shared building blocks for decision models (boolean / choice / score).
 *
 * Used by the Settings "Try a decision" dialog and the Chat decision composer,
 * so a criteria editor or an answer distribution looks and behaves identically
 * in both places.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
// eslint-disable-next-line no-restricted-imports -- Leaf module: request types and pure builders, no React.
import {
  type DecisionAnswer,
  type DecisionDraft,
  type WireQuestion,
  choiceCriterion,
  criterionText,
  noulCriterion,
  scoreLevels,
} from "@/features/settings/lib/decision-request";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Cancel01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useLayoutEffect, useRef } from "react";

export type DecisionDraftUpdate = (patch: Partial<DecisionDraft>) => void;

export function SectionLabel({
  children,
}: {
  children: ReactNode;
}): ReactElement {
  return (
    <h3 className="text-ui-11 font-medium uppercase tracking-wider text-muted-foreground">
      {children}
    </h3>
  );
}

export function Field({
  label,
  help,
  htmlFor,
  children,
}: {
  label: string;
  help: string;
  htmlFor?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-0.5">
        <label
          htmlFor={htmlFor}
          className="text-sm font-semibold text-foreground"
        >
          {label}
        </label>
        <span className="text-ui-12 text-muted-foreground">{help}</span>
      </div>
      {children}
    </div>
  );
}

export function IconButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
}): ReactElement {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      disabled={disabled}
      // Keep Tab inside the row's inputs; the row is still removable by mouse.
      tabIndex={-1}
      onClick={onClick}
      className="shrink-0 text-muted-foreground"
    >
      <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
    </Button>
  );
}

export function AddButton({
  onClick,
  children,
}: {
  onClick: () => void;
  children: ReactNode;
}): ReactElement {
  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      tabIndex={-1}
      onClick={onClick}
      className="w-fit gap-1.5 px-2.5 text-ui-12 text-muted-foreground"
    >
      <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-3.5" />
      {children}
    </Button>
  );
}

/** Yes / no criteria. Both boxes share the taller one's height. */
export function NoulCriteria({
  draft,
  update,
  idPrefix = "decision",
  minHeight = 80,
}: {
  draft: DecisionDraft;
  update: DecisionDraftUpdate;
  idPrefix?: string;
  minHeight?: number;
}): ReactElement {
  const t = useT();
  const yesRef = useRef<HTMLTextAreaElement | null>(null);
  const noRef = useRef<HTMLTextAreaElement | null>(null);

  // One measurement for the pair: both boxes always end at the same height,
  // even when only one of them wraps to several lines.
  useLayoutEffect(() => {
    const yes = yesRef.current;
    const no = noRef.current;
    if (!yes || !no) return;
    const fit = () => {
      for (const el of [yes, no]) el.style.height = "auto";
      const height = Math.max(yes.scrollHeight, no.scrollHeight, minHeight);
      yes.style.height = `${height}px`;
      no.style.height = `${height}px`;
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(yes);
    observer.observe(no);
    return () => observer.disconnect();
  }, [draft.yesWhen, draft.noWhen, minHeight]);

  const sides = [
    { key: "yesWhen" as const, label: t("decisions.yesWhen"), ref: yesRef },
    { key: "noWhen" as const, label: t("decisions.noWhen"), ref: noRef },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {sides.map((side) => (
        <div key={side.key} className="flex min-w-0 flex-col gap-1.5">
          <label
            htmlFor={`${idPrefix}-${side.key}`}
            className="text-ui-12 text-muted-foreground"
          >
            {side.label}
          </label>
          <textarea
            ref={side.ref}
            id={`${idPrefix}-${side.key}`}
            value={draft[side.key]}
            onChange={(event) => update({ [side.key]: event.target.value })}
            placeholder={t("decisions.criteriaPlaceholder")}
            rows={1}
            className="w-full resize-none overflow-y-auto rounded-lg border-0 bg-muted/50 px-3 py-2 text-ui-13 leading-relaxed text-foreground outline-none placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>
      ))}
    </div>
  );
}

/** Choice criteria: numbered rows in the order the model should see them. */
export function ChoiceCriteria({
  draft,
  update,
}: {
  draft: DecisionDraft;
  update: DecisionDraftUpdate;
}): ReactElement {
  const t = useT();
  const setOption = (index: number, name: string) =>
    update({
      options: draft.options.map((option, i) =>
        i === index ? { ...option, name } : option,
      ),
    });
  return (
    <div className="flex flex-col gap-2">
      {draft.options.map((option, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: rows have no stable id
          key={i}
          className="flex min-w-0 items-center gap-2"
        >
          <span className="flex size-6 shrink-0 items-center justify-center font-mono text-ui-11 tabular-nums text-muted-foreground">
            {i + 1}
          </span>
          <Input
            value={option.name}
            onChange={(e) => setOption(i, e.target.value)}
            placeholder={t("decisions.optionPlaceholder")}
            aria-label={t("decisions.optionLabel", { n: i + 1 })}
            className="h-9 min-w-0 flex-1 border-0 bg-muted/50"
          />
          <IconButton
            label={t("decisions.removeOption", { n: i + 1 })}
            disabled={draft.options.length <= 2}
            onClick={() =>
              update({ options: draft.options.filter((_, j) => j !== i) })
            }
          />
        </div>
      ))}
      <AddButton
        onClick={() =>
          update({
            options: [...draft.options, { name: "", description: "" }],
          })
        }
      >
        {t("decisions.addOption")}
      </AddButton>
    </div>
  );
}

/** Score criteria: numbered levels, lowest first. */
export function ScoreCriteria({
  draft,
  update,
}: {
  draft: DecisionDraft;
  update: DecisionDraftUpdate;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex flex-col gap-2">
      {draft.levels.map((level, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: rows have no stable id
          key={i}
          className="flex min-w-0 items-center gap-2"
        >
          <span className="flex size-6 shrink-0 items-center justify-center font-mono text-ui-11 tabular-nums text-muted-foreground">
            {i + 1}
          </span>
          <Input
            value={level}
            onChange={(e) =>
              update({
                levels: draft.levels.map((l, j) =>
                  j === i ? e.target.value : l,
                ),
              })
            }
            placeholder={t("decisions.levelPlaceholder")}
            aria-label={t("decisions.levelLabel", { n: i + 1 })}
            className="h-9 min-w-0 flex-1 border-0 bg-muted/50"
          />
          <IconButton
            label={t("decisions.removeLevel", { n: i + 1 })}
            disabled={draft.levels.length <= 1}
            onClick={() =>
              update({ levels: draft.levels.filter((_, j) => j !== i) })
            }
          />
        </div>
      ))}
      <AddButton onClick={() => update({ levels: [...draft.levels, ""] })}>
        {t("decisions.addLevel")}
      </AddButton>
    </div>
  );
}

export function ProbabilityRow({
  label,
  hint,
  index,
  value,
  top,
  mono,
  percent,
}: {
  label: string;
  hint?: string;
  index?: string;
  value: number;
  top: boolean;
  mono?: boolean;
  percent: (value: number) => string;
}): ReactElement {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span
          className={cn(
            "min-w-0 truncate text-sm",
            mono && "font-mono text-ui-13",
            top ? "font-semibold text-foreground" : "text-muted-foreground",
          )}
          title={hint ? `${label}: ${hint}` : label}
        >
          {index !== undefined ? (
            <span className="mr-1.5 font-mono text-ui-11 font-normal tabular-nums text-muted-foreground">
              {index}
            </span>
          ) : null}
          {label}
          {hint ? (
            <span className="ml-2 font-sans text-ui-12 font-normal text-muted-foreground">
              {hint}
            </span>
          ) : null}
        </span>
        <span
          className={cn(
            "shrink-0 font-mono text-ui-12 tabular-nums",
            top ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {percent(value)}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-500",
            top ? "bg-primary" : "bg-muted-foreground/35",
          )}
          style={{ width: `${Math.max(value * 100, 0.5)}%` }}
        />
      </div>
    </div>
  );
}

export function AnswerHeading({
  answer,
  detail,
  showLabel = true,
}: {
  answer: string;
  detail?: string;
  showLabel?: boolean;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex flex-col gap-1">
      {showLabel ? (
        <span className="text-ui-12 text-muted-foreground">
          {t("decisions.answer")}
        </span>
      ) : null}
      <span className="break-words text-ui-34 font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
        {answer}
      </span>
      {detail ? (
        <span className="break-words text-ui-15 leading-snug text-muted-foreground">
          {detail}
        </span>
      ) : null}
    </div>
  );
}

/**
 * The distribution alone: no heading, for surfaces that already name the answer
 * (the chat card puts it next to Copy JSON).
 */
export function AnswerDistribution({
  answer,
  question,
  percent,
  labels,
}: {
  answer: DecisionAnswer;
  question: WireQuestion | undefined;
  percent: (value: number) => string;
  /** Choice options by their wire key ("1", "2", ...), so a row reads "2 Mango". */
  labels?: Record<string, string>;
}): ReactElement {
  const t = useT();
  if (answer.type === "noul") {
    const yes = answer.noul;
    return (
      <div className="flex flex-col gap-2">
        {/* Two segments with a real gap: the gap is the divider, not a rule over the bar. */}
        <div className="flex h-2.5 items-center gap-1">
          <div
            role="presentation"
            className="h-full min-w-0.5 rounded-full bg-emerald-500 transition-transform duration-200 ease-out hover:scale-y-125 motion-reduce:transition-none"
            style={{ flexGrow: Math.max(yes, 0.001), flexBasis: 0 }}
          />
          <div
            role="presentation"
            className="h-full min-w-0.5 rounded-full bg-red-500 transition-transform duration-200 ease-out hover:scale-y-125 motion-reduce:transition-none"
            style={{ flexGrow: Math.max(1 - yes, 0.001), flexBasis: 0 }}
          />
        </div>
        <div className="flex justify-between font-mono text-ui-12 tabular-nums text-muted-foreground">
          <span>
            {t("decisions.yes")} {percent(yes)}
          </span>
          <span>
            {t("decisions.no")} {percent(1 - yes)}
          </span>
        </div>
      </div>
    );
  }

  if (answer.type === "choice") {
    // Input order, not probability order: the rows are the user's own list.
    return (
      <div className="flex flex-col gap-3.5">
        {Object.entries(answer.probabilities).map(([key, value]) => {
          const name = labels?.[key] ?? choiceCriterion(question, key) ?? "";
          return (
            <ProbabilityRow
              key={key}
              label={name ? `${key} ${name}` : key}
              value={value}
              top={key === answer.choice}
              mono={true}
              percent={percent}
            />
          );
        })}
      </div>
    );
  }

  const { keys, top } = scoreLevels(answer);
  return (
    <div className="flex flex-col gap-3.5">
      {keys.map((key) => (
        <ProbabilityRow
          key={key}
          index={`${Number(key) + 1}`}
          label={criterionText(answer.legend?.[key]) ?? key}
          value={answer.probabilities[key] ?? 0}
          top={key === top}
          percent={percent}
        />
      ))}
    </div>
  );
}

export function AnswerPreview({
  answer,
  question,
  percent,
  showLabel = true,
}: {
  answer: DecisionAnswer;
  question: WireQuestion | undefined;
  percent: (value: number) => string;
  showLabel?: boolean;
}): ReactElement {
  const t = useT();
  if (answer.type === "noul") {
    const isYes = answer.noul >= 0.5;
    return (
      <div className="flex flex-col gap-6">
        <AnswerHeading
          answer={isYes ? t("decisions.yes") : t("decisions.no")}
          detail={noulCriterion(question, isYes)}
          showLabel={showLabel}
        />
        <AnswerDistribution answer={answer} question={question} percent={percent} />
      </div>
    );
  }
  if (answer.type === "choice") {
    const label = choiceCriterion(question, answer.choice);
    return (
      <div className="flex flex-col gap-6">
        <AnswerHeading
          answer={label ? `${answer.choice} ${label}` : answer.choice}
          showLabel={showLabel}
        />
        <AnswerDistribution answer={answer} question={question} percent={percent} />
      </div>
    );
  }
  const { top, max } = scoreLevels(answer);
  return (
    <div className="flex flex-col gap-6">
      <AnswerHeading
        answer={criterionText(answer.legend?.[top]) ?? top}
        detail={t("decisions.scoreDetail", {
          score: answer.score.toFixed(2),
          max,
        })}
        showLabel={showLabel}
      />
      <AnswerDistribution answer={answer} question={question} percent={percent} />
    </div>
  );
}
