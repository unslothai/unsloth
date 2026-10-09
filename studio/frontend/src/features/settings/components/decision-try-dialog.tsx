// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  AlertCircleIcon,
  ArrowDown01Icon,
  BookOpen01Icon,
  Cancel01Icon,
  CodeIcon,
  PlayIcon,
  SlidersHorizontalIcon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  DecisionError,
  type SystemOneConnection,
  type SystemOneSettings,
  runDecision,
} from "../api/systemone";
import {
  DECISION_TYPES,
  DEFAULT_MODEL,
  type DecisionAnswer,
  type DecisionDraft,
  type DecisionResponse,
  type DecisionType,
  PRESETS,
  type PresetId,
  type WireQuestion,
  buildRequest,
  choiceCriterion,
  criterionText,
  draftFromRequest,
  initialDrafts,
  noulCriterion,
  requestText,
  scoreLevels,
} from "../lib/decision-request";
import {
  DECISION_MODEL_LABELS,
  isClefDecisionModel,
} from "../lib/decision-model-labels";
import { isMacPlatform } from "../lib/keyboard-shortcuts";

const SHORTCUT = isMacPlatform() ? "⌘↵" : "Ctrl ↵";

const TYPE_LABELS: Record<DecisionType, TranslationKey> = {
  noul: "decisions.typeNoul",
  choice: "decisions.typeChoice",
  score: "decisions.typeScore",
};
const TYPE_ABOUT: Record<DecisionType, TranslationKey> = {
  noul: "decisions.aboutNoul",
  choice: "decisions.aboutChoice",
  score: "decisions.aboutScore",
};
const CRITERIA_HELP: Record<DecisionType, TranslationKey> = {
  noul: "decisions.criteriaNoulHelp",
  choice: "decisions.criteriaChoiceHelp",
  score: "decisions.criteriaScoreHelp",
};
const PRESET_LABELS: Record<PresetId, TranslationKey> = {
  toolApproval: "decisions.presetToolApproval",
  promptInjection: "decisions.presetPromptInjection",
  ticketRouting: "decisions.presetTicketRouting",
  moderation: "decisions.presetModeration",
  intent: "decisions.presetIntent",
  grading: "decisions.presetGrading",
};

type InputView = "form" | "json";
type OutputView = "preview" | "json";

type Outcome =
  | {
      kind: "ok";
      response: DecisionResponse;
      latencyMs: number;
      sent: unknown;
    }
  | { kind: "error"; message: string; status: number | null };

type Update = (patch: Partial<DecisionDraft>) => void;

function parseJson(text: string): { value: unknown } | { error: string } {
  try {
    return { value: JSON.parse(text) as unknown };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

function sentQuestions(sent: unknown): Record<string, WireQuestion> {
  const questions = (sent as { questions?: unknown } | null)?.questions;
  return questions && typeof questions === "object"
    ? (questions as Record<string, WireQuestion>)
    : {};
}

function SectionLabel({ children }: { children: ReactNode }): ReactElement {
  return (
    <h3 className="text-ui-11 font-medium uppercase tracking-wider text-muted-foreground">
      {children}
    </h3>
  );
}

function Field({
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

function ViewToggle<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { id: T; label: string; icon: IconSvgElement; disabled?: boolean }[];
}): ReactElement {
  return (
    <Tabs value={value} onValueChange={(next) => onChange(next as T)}>
      <TabsList className="group-data-horizontal/tabs:h-8">
        {options.map((option) => (
          <TabsTrigger
            key={option.id}
            value={option.id}
            disabled={option.disabled}
            className="px-2.5 text-ui-12"
          >
            <span className="inline-flex items-center gap-1.5">
              <HugeiconsIcon
                icon={option.icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              {option.label}
            </span>
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}

function CodeBlock({
  title,
  value,
  onChange,
}: {
  title: string;
  value: string;
  onChange?: (value: string) => void;
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-border/70 bg-muted/40">
      <div className="truncate border-b border-border/60 px-3.5 py-2 font-mono text-ui-11 text-muted-foreground">
        {title}
      </div>
      {onChange ? (
        <textarea
          spellCheck={false}
          value={value}
          aria-label={title}
          onChange={(e) => onChange(e.target.value)}
          className="field-sizing-content min-h-72 w-full resize-none bg-transparent px-3.5 py-3 font-mono text-ui-12 leading-relaxed text-foreground outline-none"
        />
      ) : (
        <pre className="max-h-128 overflow-auto px-3.5 py-3 font-mono text-ui-12 leading-relaxed text-foreground">
          {value}
        </pre>
      )}
    </div>
  );
}

function IconButton({
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
      onClick={onClick}
      className="shrink-0 text-muted-foreground"
    >
      <HugeiconsIcon
        icon={Cancel01Icon}
        strokeWidth={1.75}
        className="size-4"
      />
    </Button>
  );
}

function AddButton({
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
      onClick={onClick}
      className="w-fit gap-1.5 px-2.5 text-ui-12 text-muted-foreground"
    >
      <HugeiconsIcon icon={Add01Icon} strokeWidth={1.75} className="size-3.5" />
      {children}
    </Button>
  );
}

function NoulCriteria({
  draft,
  update,
}: {
  draft: DecisionDraft;
  update: Update;
}): ReactElement {
  const t = useT();
  const sides = [
    {
      key: "yesWhen" as const,
      label: t("decisions.yesWhen"),
      dot: "bg-emerald-500",
      border: "border-emerald-500/35 dark:border-emerald-500/30",
    },
    {
      key: "noWhen" as const,
      label: t("decisions.noWhen"),
      dot: "bg-red-500",
      border: "border-red-500/35 dark:border-red-500/30",
    },
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {sides.map((side) => (
        <div key={side.key} className="flex min-w-0 flex-col gap-1.5">
          <label
            htmlFor={`decision-${side.key}`}
            className="inline-flex items-center gap-1.5 text-ui-12 text-muted-foreground"
          >
            <span className={cn("size-1.5 rounded-full", side.dot)} />
            {side.label}
          </label>
          <Textarea
            id={`decision-${side.key}`}
            value={draft[side.key]}
            onChange={(e) => update({ [side.key]: e.target.value })}
            className={cn("min-h-20 dark:border", side.border)}
          />
        </div>
      ))}
    </div>
  );
}

function ChoiceCriteria({
  draft,
  update,
}: {
  draft: DecisionDraft;
  update: Update;
}): ReactElement {
  const t = useT();
  const setOption = (
    index: number,
    patch: Partial<DecisionDraft["options"][number]>,
  ) =>
    update({
      options: draft.options.map((option, i) =>
        i === index ? { ...option, ...patch } : option,
      ),
    });
  return (
    <div className="flex flex-col gap-2">
      <div className="hidden gap-2 px-1 text-ui-11 text-muted-foreground sm:flex">
        <span className="w-36 shrink-0">{t("decisions.optionName")}</span>
        <span>{t("decisions.optionDescription")}</span>
      </div>
      {draft.options.map((option, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: rows have no stable id
          key={i}
          className="flex items-start gap-2 max-sm:flex-wrap"
        >
          <Input
            value={option.name}
            onChange={(e) => setOption(i, { name: e.target.value })}
            placeholder={t("decisions.optionNamePlaceholder")}
            aria-label={t("decisions.optionNameLabel", { n: i + 1 })}
            className="font-mono text-ui-12 max-sm:min-w-0 max-sm:flex-1 sm:w-36 sm:shrink-0"
          />
          <Input
            value={option.description}
            onChange={(e) => setOption(i, { description: e.target.value })}
            placeholder={t("decisions.optionDescriptionPlaceholder")}
            aria-label={t("decisions.optionDescriptionLabel", { n: i + 1 })}
            className="flex-1 max-sm:order-last max-sm:basis-full"
          />
          <IconButton
            label={t("decisions.removeOption", { n: i + 1 })}
            disabled={draft.options.length <= 1}
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

function ScoreCriteria({
  draft,
  update,
}: {
  draft: DecisionDraft;
  update: Update;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex flex-col gap-2">
      {draft.levels.map((level, i) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: rows have no stable id
          key={i}
          className="flex items-center gap-2"
        >
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted font-mono text-ui-11 text-muted-foreground">
            {i}
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
            aria-label={t("decisions.levelLabel", { n: i })}
            className="flex-1"
          />
          <IconButton
            label={t("decisions.removeLevel", { n: i })}
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

function ProbabilityRow({
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
            <span className="mr-2 font-mono text-ui-11 font-normal text-muted-foreground">
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

function AnswerHeading({
  answer,
  detail,
}: {
  answer: string;
  detail?: string;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex flex-col gap-1">
      <span className="text-ui-12 text-muted-foreground">
        {t("decisions.answer")}
      </span>
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

function AnswerPreview({
  answer,
  question,
  percent,
}: {
  answer: DecisionAnswer;
  question: WireQuestion | undefined;
  percent: (value: number) => string;
}): ReactElement {
  const t = useT();
  if (answer.type === "noul") {
    const yes = answer.noul;
    const isYes = yes >= 0.5;
    return (
      <div className="flex flex-col gap-6">
        <AnswerHeading
          answer={isYes ? t("decisions.yes") : t("decisions.no")}
          detail={noulCriterion(question, isYes)}
        />
        <div className="flex flex-col gap-2">
          <div className="relative flex h-2.5 overflow-hidden rounded-full">
            <div
              className="h-full bg-primary transition-[width] duration-500"
              style={{ width: `${yes * 100}%` }}
            />
            <div className="h-full flex-1 bg-red-400/90 dark:bg-red-500/80" />
            <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-card" />
          </div>
          <div className="flex justify-between font-mono text-ui-12 tabular-nums">
            <span
              className={isYes ? "text-foreground" : "text-muted-foreground"}
            >
              {t("decisions.yes")} {percent(yes)}
            </span>
            <span
              className={isYes ? "text-muted-foreground" : "text-foreground"}
            >
              {t("decisions.no")} {percent(1 - yes)}
            </span>
          </div>
        </div>
      </div>
    );
  }
  if (answer.type === "choice") {
    const rows = Object.entries(answer.probabilities).sort(
      (a, b) => b[1] - a[1],
    );
    return (
      <div className="flex flex-col gap-6">
        <AnswerHeading
          answer={answer.choice}
          detail={choiceCriterion(question, answer.choice)}
        />
        <div className="flex flex-col gap-3.5">
          {rows.map(([name, value]) => (
            <ProbabilityRow
              key={name}
              label={name}
              hint={choiceCriterion(question, name)}
              value={value}
              top={name === answer.choice}
              mono={true}
              percent={percent}
            />
          ))}
        </div>
      </div>
    );
  }
  const { keys, top, max } = scoreLevels(answer);
  return (
    <div className="flex flex-col gap-6">
      <AnswerHeading
        answer={criterionText(answer.legend[top]) ?? top}
        detail={t("decisions.scoreDetail", {
          score: answer.score.toFixed(2),
          max,
        })}
      />
      <div className="flex flex-col gap-3.5">
        {keys.map((key) => (
          <ProbabilityRow
            key={key}
            index={key}
            label={criterionText(answer.legend[key]) ?? key}
            value={answer.probabilities[key] ?? 0}
            top={key === top}
            percent={percent}
          />
        ))}
      </div>
    </div>
  );
}

function Notice({
  icon,
  title,
  children,
}: {
  icon: IconSvgElement;
  title: string;
  children?: ReactNode;
}): ReactElement {
  return (
    <div className="flex min-h-64 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border/70 px-6 py-10 text-center">
      <span className="flex size-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
      </span>
      <div className="flex max-w-xs flex-col items-center gap-3">
        <span className="text-sm font-medium text-foreground">{title}</span>
        {children}
      </div>
    </div>
  );
}

function ErrorResult({
  message,
  status,
}: {
  message: string;
  status: number | null;
}): ReactElement {
  const t = useT();
  return (
    <div
      role="alert"
      className="flex gap-3 rounded-xl border border-red-500/25 bg-red-500/5 px-4 py-3.5"
    >
      <HugeiconsIcon
        icon={AlertCircleIcon}
        strokeWidth={1.75}
        className="mt-0.5 size-4 shrink-0 text-red-600 dark:text-red-400"
      />
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm font-medium text-red-700 dark:text-red-300">
          {status === null
            ? t("decisions.cantRun")
            : t("decisions.failed", { status })}
        </span>
        <span className="break-words text-ui-13 text-red-700/80 dark:text-red-300/80">
          {message}
        </span>
      </div>
    </div>
  );
}

function ModelPicker({
  value,
  onChange,
  settings,
  connections,
}: {
  value: string;
  onChange: (value: string) => void;
  settings: SystemOneSettings;
  connections: SystemOneConnection[];
}): ReactElement {
  const t = useT();
  const label = (name: string) => {
    const connection = connections.find((c) => c.name === name);
    if (connection) return `${connection.provider} · ${connection.model}`;
    const option = settings.models.find((m) => m.name === name);
    if (option?.kind === "fine_tune" && option.label) return option.label;
    const key = DECISION_MODEL_LABELS[name];
    if (!key) return name;
    return isClefDecisionModel(name)
      ? t(key)
      : t("decisions.layaModel", { model: t(key) });
  };
  const display = (name: string) =>
    name === DEFAULT_MODEL
      ? t("decisions.defaultModel", { model: label(settings.model) })
      : label(name);

  const resolved = value === DEFAULT_MODEL ? settings.model : value;
  const tone =
    resolved === settings.model && resolved.startsWith("connection:")
      ? "ready"
      : settings.loadingModel && settings.loadingModel === resolved
        ? "pending"
        : settings.loadedModel && settings.loadedModel === resolved
          ? "ready"
          : null;
  const toneLabel =
    tone === "ready"
      ? t("decisions.modelReady")
      : tone === "pending"
        ? t("decisions.modelLoading")
        : t("decisions.modelNotLoaded");
  const providers = [...new Set(connections.map((c) => c.providerId))];

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        className="h-9 w-72 min-w-0 max-sm:w-full"
        aria-label={t("decisions.model")}
        title={`${display(value)} · ${toneLabel}`}
      >
        <span className="flex min-w-0 items-center gap-2">
          <span
            aria-hidden={true}
            className={cn(
              "size-1.5 shrink-0 rounded-full",
              tone === "ready"
                ? "bg-emerald-500"
                : tone === "pending"
                  ? "animate-pulse bg-blue-500"
                  : "bg-muted-foreground/40",
            )}
          />
          <span className="truncate">{display(value)}</span>
          <span className="sr-only">{toneLabel}</span>
        </span>
      </SelectTrigger>
      <SelectContent align="end">
        <SelectItem value={DEFAULT_MODEL}>{display(DEFAULT_MODEL)}</SelectItem>
        {settings.models.length ? (
          <SelectGroup>
            <SelectLabel>{t("decisions.thisMachine")}</SelectLabel>
            {settings.models.map((option) => (
              <SelectItem
                key={option.name}
                value={option.name}
                disabled={!option.available || option.name !== settings.model}
                title={
                  option.unavailableReason ??
                  (option.name === settings.model
                    ? undefined
                    : t("decisions.connectionNotDefault"))
                }
              >
                {label(option.name)}
              </SelectItem>
            ))}
          </SelectGroup>
        ) : null}
        {providers.map((providerId) => {
          const group = connections.filter((c) => c.providerId === providerId);
          return (
            <SelectGroup key={providerId}>
              <SelectLabel>{group[0].provider}</SelectLabel>
              {group.map((option) => (
                <SelectItem
                  key={option.name}
                  value={option.name}
                  disabled={option.name !== settings.model}
                  title={
                    option.name === settings.model
                      ? undefined
                      : t("decisions.connectionNotDefault")
                  }
                >
                  {option.model}
                </SelectItem>
              ))}
            </SelectGroup>
          );
        })}
      </SelectContent>
    </Select>
  );
}


export function DecisionTryDialog({
  open,
  onOpenChange,
  initialModel,
  settings,
  connections,
  onRun,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialModel?: string;
  settings: SystemOneSettings;
  connections: SystemOneConnection[];
  onRun: () => void;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const [type, setType] = useState<DecisionType>("noul");
  const [drafts, setDrafts] =
    useState<Record<DecisionType, DecisionDraft>>(initialDrafts);
  const [model, setModel] = useState(initialModel ?? DEFAULT_MODEL);
  const [inputView, setInputView] = useState<InputView>("form");
  const [outputView, setOutputView] = useState<OutputView>("preview");
  const [jsonText, setJsonText] = useState("");
  const [outcomes, setOutcomes] = useState<
    Partial<Record<DecisionType, Outcome>>
  >({});
  const [running, setRunning] = useState(false);
  const [waiting, setWaiting] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const draft = drafts[type];
  const outcome = outcomes[type];

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
    setWaiting(null);
  }, []);

  useEffect(() => {
    if (open) return stop;
  }, [open, stop]);

  useEffect(() => {
    if (open && initialModel) setModel(initialModel);
  }, [open, initialModel]);

  const request = useMemo(
    () => buildRequest(type, draft, model),
    [type, draft, model],
  );

  const percent = useMemo(() => {
    const format = new Intl.NumberFormat(locale, {
      style: "percent",
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
    return (value: number) => format.format(value);
  }, [locale]);
  const integer = useMemo(() => new Intl.NumberFormat(locale), [locale]);

  const update = useCallback<Update>(
    (patch) =>
      setDrafts((all) => ({ ...all, [type]: { ...all[type], ...patch } })),
    [type],
  );

  const jsonFitsForm = useMemo(() => {
    if (inputView !== "json") return true;
    const parsed = parseJson(jsonText);
    return "value" in parsed && draftFromRequest(parsed.value) !== null;
  }, [inputView, jsonText]);

  const syncFromJson = () => {
    const parsed = parseJson(jsonText);
    const restored = "value" in parsed ? draftFromRequest(parsed.value) : null;
    if (restored) {
      setDrafts((all) => ({ ...all, [restored.type]: restored.draft }));
      if (restored.model) setModel(restored.model);
    }
    return restored;
  };

  const changeInputView = (next: InputView) => {
    if (next === inputView) return;
    if (next === "json") {
      setJsonText(requestText(request));
    } else {
      const restored = syncFromJson();
      if (!restored) return;
      setType(restored.type);
    }
    setInputView(next);
  };

  const changeType = (next: DecisionType) => {
    if (inputView === "json") {
      const restored = syncFromJson();
      if (restored) {
        const nextDraft =
          restored.type === next ? restored.draft : drafts[next];
        setJsonText(
          requestText(buildRequest(next, nextDraft, restored.model ?? model)),
        );
      }
    }
    setType(next);
  };

  const changeModel = (next: string) => {
    setModel(next);
    if (inputView !== "json") return;
    const parsed = parseJson(jsonText);
    const value = "value" in parsed ? parsed.value : null;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      setJsonText(requestText({ ...value, model: next }));
    }
  };

  const applyPreset = (id: PresetId) => {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setDrafts((all) => ({ ...all, [preset.type]: preset.draft }));
    setType(preset.type);
    setOutcomes((all) => ({ ...all, [preset.type]: undefined }));
    if (inputView === "json") {
      setJsonText(requestText(buildRequest(preset.type, preset.draft, model)));
    }
  };

  const run = async () => {
    if (running) return;
    let body: unknown = request;
    if (inputView === "json") {
      const parsed = parseJson(jsonText);
      if (!("value" in parsed)) {
        setOutcomes((all) => ({
          ...all,
          [type]: {
            kind: "error",
            message: `${t("decisions.invalidJson")} ${parsed.error}`,
            status: null,
          },
        }));
        return;
      }
      body = parsed.value;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    const runType =
      inputView === "json" ? (draftFromRequest(body)?.type ?? type) : type;
    setType(runType);
    setRunning(true);
    setWaiting(null);
    try {
      const { response, latencyMs } = await runDecision(
        body,
        controller.signal,
        setWaiting,
      );
      setOutcomes((all) => ({
        ...all,
        [runType]: { kind: "ok", response, latencyMs, sent: body },
      }));
    } catch (err) {
      if (controller.signal.aborted) return;
      setOutcomes((all) => ({
        ...all,
        [runType]: {
          kind: "error",
          message: err instanceof Error ? err.message : String(err),
          status: err instanceof DecisionError ? err.status : null,
        },
      }));
    } finally {
      if (!controller.signal.aborted) {
        setRunning(false);
        setWaiting(null);
        onRun();
      }
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (
      event.key !== "Enter" ||
      !(event.metaKey || event.ctrlKey) ||
      event.nativeEvent.isComposing ||
      event.repeat ||
      (event.target instanceof Element &&
        event.target.closest('[role="menu"], [role="listbox"]'))
    ) {
      return;
    }
    event.preventDefault();
    void run();
  };

  const answers =
    outcome?.kind === "ok"
      ? Object.entries(outcome.response.answers ?? {})
      : [];
  const questions = outcome?.kind === "ok" ? sentQuestions(outcome.sent) : {};

  let result: ReactNode;
  if (running) {
    result = (
      <div className="flex min-h-64 items-center justify-center gap-2 text-ui-13 text-muted-foreground">
        <Spinner className="size-4" />
        {waiting ?? t("decisions.running")}
      </div>
    );
  } else if (!outcome) {
    result = (
      <Notice icon={PlayIcon} title={t("decisions.emptyTitle")}>
        <span className="text-ui-12 text-muted-foreground">
          {t("decisions.emptyBody", { shortcut: SHORTCUT })}
        </span>
      </Notice>
    );
  } else if (outcome.kind === "error") {
    result = <ErrorResult message={outcome.message} status={outcome.status} />;
  } else if (outputView === "json") {
    result = (
      <CodeBlock
        title={t("decisions.responseTitle")}
        value={requestText(outcome.response)}
      />
    );
  } else {
    result = (
      <div className="flex flex-col gap-6">
        {answers.map(([name, answer]) => (
          <div key={name} className="flex flex-col gap-2">
            {answers.length > 1 ? (
              <span className="font-mono text-ui-12 text-muted-foreground">
                {name}
              </span>
            ) : null}
            <AnswerPreview
              answer={answer}
              question={questions[name]}
              percent={percent}
            />
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-border/70 pt-4 text-ui-13 tabular-nums text-muted-foreground">
          <span>
            {t("decisions.latency", { ms: integer.format(outcome.latencyMs) })}
          </span>
          <span>
            {t("decisions.inputTokens", {
              count: integer.format(outcome.response.usage?.input_tokens ?? 0),
            })}
          </span>
          <span className="font-mono text-ui-12">{outcome.response.model}</span>
        </div>
      </div>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        onKeyDown={onKeyDown}
        className="flex h-[min(90dvh,52rem)] flex-col gap-0 overflow-hidden bg-card p-0 font-heading sm:max-w-5xl"
      >
        <DialogHeader className="gap-1 px-6 pb-4 pt-6 pr-14 text-left">
          <DialogTitle className="font-heading">
            {t("decisions.title")}
          </DialogTitle>
          <DialogDescription>{t("decisions.description")}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center justify-between gap-3 border-y border-border/70 px-4 py-3 sm:px-6">
          <Tabs
            value={type}
            onValueChange={(next) => changeType(next as DecisionType)}
            className="max-sm:w-full"
          >
            <TabsList className="max-sm:w-full">
              {DECISION_TYPES.map((id) => (
                <TabsTrigger key={id} value={id} className="px-3 max-sm:px-2">
                  <span className="inline-flex items-baseline gap-1.5">
                    {t(TYPE_LABELS[id])}
                    <code className="font-mono text-ui-11 font-normal text-muted-foreground max-sm:hidden">
                      {id}
                    </code>
                  </span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          <ModelPicker
            value={model}
            onChange={changeModel}
            settings={settings}
            connections={connections}
          />
        </div>

        <div className="grid min-h-0 flex-1 overflow-y-auto lg:grid-cols-2 lg:grid-rows-[minmax(0,1fr)] lg:overflow-hidden">
          <div className="flex min-h-0 min-w-0 flex-col border-border/70 max-lg:border-b lg:border-r">
            <div className="flex flex-col gap-5 px-4 pb-6 pt-5 sm:px-6 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
              <div className="flex items-center justify-between gap-3">
                <SectionLabel>{t("decisions.question")}</SectionLabel>
                <ViewToggle
                  value={inputView}
                  onChange={changeInputView}
                  options={[
                    {
                      id: "form",
                      label: t("decisions.form"),
                      icon: SlidersHorizontalIcon,
                      disabled: !jsonFitsForm,
                    },
                    { id: "json", label: t("decisions.json"), icon: CodeIcon },
                  ]}
                />
              </div>

              {inputView === "form" ? (
                <>
                  <p className="text-ui-15 leading-relaxed text-foreground">
                    {t(TYPE_ABOUT[type])}
                  </p>
                  <Field
                    label={t("decisions.state")}
                    help={t("decisions.stateHelp")}
                    htmlFor="decision-state"
                  >
                    <Textarea
                      id="decision-state"
                      value={draft.state}
                      onChange={(e) => update({ state: e.target.value })}
                      spellCheck={false}
                      className="min-h-28 font-mono text-ui-13 leading-relaxed"
                    />
                  </Field>
                  <Field
                    label={t("decisions.question")}
                    help={t("decisions.questionHelp")}
                    htmlFor="decision-question"
                  >
                    <Textarea
                      id="decision-question"
                      value={draft.instructions}
                      onChange={(e) => update({ instructions: e.target.value })}
                      className="min-h-20"
                    />
                  </Field>
                  <Field
                    label={t("decisions.criteria")}
                    help={t(CRITERIA_HELP[type])}
                  >
                    {type === "noul" ? (
                      <NoulCriteria draft={draft} update={update} />
                    ) : type === "choice" ? (
                      <ChoiceCriteria draft={draft} update={update} />
                    ) : (
                      <ScoreCriteria draft={draft} update={update} />
                    )}
                  </Field>
                </>
              ) : (
                <>
                  <p className="text-ui-13 leading-relaxed text-muted-foreground">
                    {t("decisions.jsonHelp")}
                  </p>
                  <CodeBlock
                    title="POST /v1/systemone"
                    value={jsonText}
                    onChange={setJsonText}
                  />
                </>
              )}
            </div>

            <div className="z-10 flex shrink-0 items-center justify-between gap-3 border-t border-border/70 bg-card px-4 py-3 max-lg:sticky max-lg:bottom-0 sm:px-6">
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="gap-1.5 text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={BookOpen01Icon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                    {t("decisions.examples")}
                    <HugeiconsIcon
                      icon={ArrowDown01Icon}
                      strokeWidth={1.75}
                      className="size-3.5"
                    />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent side="top" align="start" className="w-64">
                  {PRESETS.map((preset) => (
                    <DropdownMenuItem
                      key={preset.id}
                      onSelect={() => applyPreset(preset.id)}
                      className="justify-between gap-3"
                    >
                      <span className="truncate">
                        {t(PRESET_LABELS[preset.id])}
                      </span>
                      <span className="shrink-0 text-ui-11 text-muted-foreground">
                        {t(TYPE_LABELS[preset.type])}
                      </span>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              <Button
                type="button"
                onClick={() => void run()}
                aria-busy={running}
                className="gap-2 pl-3.5 pr-2"
              >
                {running ? (
                  <Spinner className="size-4 text-primary-foreground" />
                ) : (
                  <HugeiconsIcon
                    icon={PlayIcon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                )}
                {t("decisions.run")}
                <kbd className="rounded-full bg-primary-foreground/20 px-1.5 py-0.5 font-mono text-ui-10 text-primary-foreground/90">
                  {SHORTCUT}
                </kbd>
              </Button>
            </div>
          </div>

          <div
            aria-live="polite"
            className="flex min-w-0 flex-col gap-5 px-4 pb-6 pt-5 sm:px-6 lg:min-h-0 lg:overflow-y-auto"
          >
            <div className="flex items-center justify-between gap-3">
              <SectionLabel>{t("decisions.result")}</SectionLabel>
              <ViewToggle
                value={outputView}
                onChange={setOutputView}
                options={[
                  {
                    id: "preview",
                    label: t("decisions.preview"),
                    icon: ViewIcon,
                  },
                  { id: "json", label: t("decisions.json"), icon: CodeIcon },
                ]}
              />
            </div>
            {result}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
