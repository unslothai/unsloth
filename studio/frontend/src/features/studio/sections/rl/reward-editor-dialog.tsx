// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionField } from "@/components/section-box";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  RULE_TYPES,
  type RewardRecord,
  type RuleType,
  formatScore,
  importReward,
  useRewardPreview,
} from "@/features/training";
// eslint-disable-next-line no-restricted-imports -- through the barrel, js-yaml would load at startup.
import {
  type DraftRule,
  NORMALIZERS,
  type RewardDraft,
  copyRewardName,
  defaultRewardRule,
  keyListText,
  normalizeRewardName,
  parseKeyList,
  rewardDraftMarkdown,
} from "@/features/training/lib/reward-editor";
import { type TranslationKey, useT } from "@/i18n";
import { type ReactElement, type ReactNode, useState } from "react";
import { RewardExplanationHint } from "./reward-explanation";

type Extraction = { between: [string, string] } | { regex: string } | null;
type Tab = "configuration" | "extraction" | "test";

function NumberInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: unknown;
  onChange: (value: string) => void;
}): ReactElement {
  return (
    <SectionField label={label}>
      <Input
        type="number"
        step="any"
        aria-label={label}
        className="tabular-nums"
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value)}
      />
    </SectionField>
  );
}

function KeyListInput({
  label,
  keys,
  onChange,
}: {
  label: string;
  keys: string[];
  onChange: (keys: string[]) => void;
}): ReactElement {
  const [text, setText] = useState(() => keys.join(", "));
  const shown = keyListText(text, keys);
  return (
    <Input
      aria-label={label}
      className="font-mono text-xs"
      placeholder="answer, reasoning"
      value={shown}
      onChange={(e) => {
        setText(e.target.value);
        onChange(parseKeyList(e.target.value));
      }}
    />
  );
}

function OptionSelect<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (value: T) => void;
}): ReactElement {
  return (
    <SectionField label={label}>
      <Select value={value} onValueChange={(v) => onChange(v as T)}>
        <SelectTrigger aria-label={label} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </SectionField>
  );
}

function ExtractionField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: Extraction;
  onChange: (value: Extraction) => void;
}): ReactElement {
  const t = useT();
  const mode = !value ? "whole" : "between" in value ? "between" : "regex";
  return (
    <div className="flex flex-col gap-3">
      <OptionSelect
        label={label}
        value={mode}
        options={[
          { value: "whole", label: t("rl.rewardEditor.wholeText") },
          { value: "between", label: t("rl.rewardEditor.between") },
          { value: "regex", label: t("rl.rewardEditor.regexExtract") },
        ]}
        onChange={(next) =>
          onChange(
            next === "whole"
              ? null
              : next === "between"
                ? { between: ["", ""] }
                : { regex: "" },
          )
        }
      />
      {value && "between" in value && (
        <div className="grid gap-3 sm:grid-cols-2">
          {(["afterText", "beforeText"] as const).map((key, i) => (
            <SectionField key={key} label={t(`rl.rewardEditor.${key}`)}>
              <Input
                aria-label={t(`rl.rewardEditor.${key}`)}
                className="font-mono text-xs"
                value={value.between[i]}
                onChange={(e) => {
                  const between: [string, string] = [...value.between];
                  between[i] = e.target.value;
                  onChange({ between });
                }}
              />
            </SectionField>
          ))}
        </div>
      )}
      {value && "regex" in value && (
        <SectionField
          label={t("rl.rewardEditor.pattern")}
          hint={
            <span className="font-normal text-muted-foreground/80">
              {t("rl.rewardEditor.captureHint")}
            </span>
          }
        >
          <Input
            dir="ltr"
            aria-label={t("rl.rewardEditor.pattern")}
            className="font-mono text-xs"
            value={value.regex}
            onChange={(e) => onChange({ regex: e.target.value })}
          />
        </SectionField>
      )}
    </div>
  );
}

/** Create, edit or copy a rule reward; the draft is scored live and only a save writes it. */
export function RewardEditorDialog({
  record,
  library,
  completion: sampleCompletion,
  referenceFor,
  onClose,
  onSaved,
}: {
  record?: RewardRecord;
  library: RewardRecord[];
  completion: string;
  referenceFor: (column: string) => string;
  onClose: () => void;
  onSaved: (record: RewardRecord) => void;
}): ReactElement {
  const t = useT();
  const editing = record?.source === "user";
  const [initial] = useState<RewardDraft>(() => ({
    name: editing
      ? record.name
      : record
        ? copyRewardName(record.name, library)
        : "",
    description: record?.description ?? "",
    rule: record?.rule
      ? (structuredClone(record.rule) as DraftRule)
      : defaultRewardRule("exact_match", library),
  }));
  const [draft, setDraft] = useState(initial);
  const [tab, setTab] = useState<Tab>("configuration");
  const [completion, setCompletion] = useState(sampleCompletion);
  const [referenceEdit, setReferenceEdit] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const rule = draft.rule;
  const column =
    (rule.type === "exact_match" || rule.type === "numeric") &&
    typeof rule.compare_to === "string"
      ? rule.compare_to
      : null;
  const reference = column ? (referenceEdit ?? referenceFor(column)) : null;
  const preview = useRewardPreview(
    [
      {
        rewards: [
          {
            markdown: rewardDraftMarkdown({ ...draft, name: "draft-preview" }),
            weight: 1,
          },
        ],
        reference,
      },
    ],
    completion,
  );
  const result = preview.result?.scores[0];
  const dirty = JSON.stringify(draft) !== JSON.stringify(initial);

  const update = (patch: Partial<RewardDraft>) => {
    setDraft((d) => ({ ...d, ...patch }));
    setSaveError(null);
  };
  const setRule = (patch: Record<string, unknown>) =>
    update({ rule: { ...rule, ...patch } });
  const score = rule.score as Record<string, unknown> | undefined;
  const setScore = (key: string, value: string) =>
    setRule({ score: { ...score, [key]: value } });
  const bands = (rule.bands ?? []) as Record<string, unknown>[];
  const setBand = (i: number, patch: Record<string, unknown>) =>
    setRule({ bands: bands.map((b, j) => (i === j ? { ...b, ...patch } : b)) });
  const schema = (rule.schema ?? { type: "object", required: [] }) as {
    type: string;
    required: string[];
  };
  const normalize = (rule.normalize ?? []) as string[];
  const [hitKey, missKey] =
    rule.type === "length" ? ["under", "over"] : ["match", "miss"];

  const save = async () => {
    const name = normalizeRewardName(draft.name);
    const taken = library.some(
      (r) => r.name === name && !(editing && r.name === record.name),
    );
    if (!name || taken) {
      setSaveError(
        t(name ? "rl.rewardEditor.duplicate" : "rl.rewardEditor.invalidName"),
      );
      setTab("configuration");
      return;
    }
    setSaving(true);
    try {
      onSaved(await importReward(rewardDraftMarkdown(draft), editing));
      onClose();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  // A dirty draft only closes through Cancel, not a stray click or Escape.
  const guard = (event: Event) => {
    if (dirty) {
      event.preventDefault();
    }
  };

  return (
    <Dialog open={true} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="flex max-h-[calc(100dvh-2rem)] flex-col gap-4 sm:max-w-2xl"
        onInteractOutside={guard}
        onEscapeKeyDown={guard}
      >
        <DialogHeader>
          <DialogTitle>
            {t(
              editing
                ? "rl.rewardEditor.editTitle"
                : record
                  ? "rl.rewardEditor.copyTitle"
                  : "rl.rewardEditor.newTitle",
            )}
          </DialogTitle>
          <DialogDescription>
            {t("rl.rewardEditor.description")}
          </DialogDescription>
        </DialogHeader>
        <Tabs
          value={tab}
          onValueChange={(v) => setTab(v as Tab)}
          className="min-h-0 flex-1 gap-4"
        >
          <SegmentedTabsList
            value={tab}
            ariaLabel={t("rl.rewardEditor.sections")}
            size="compact"
            options={[
              {
                value: "configuration",
                label: t("rl.rewardEditor.configurationTab"),
              },
              {
                value: "extraction",
                label: t("rl.rewardEditor.extractionTab"),
              },
              { value: "test", label: t("rl.rewardEditor.testTab") },
            ]}
          />
          <TabsContent
            value="configuration"
            className="flex min-h-0 flex-col gap-4 overflow-y-auto"
          >
            <div className="grid gap-3 sm:grid-cols-2">
              <SectionField
                label={t("rl.rewardEditor.name")}
                hint={
                  <span className="font-normal text-muted-foreground/80">
                    {t("rl.rewardEditor.nameHint")}
                  </span>
                }
              >
                <Input
                  dir="ltr"
                  autoComplete="off"
                  aria-label={t("rl.rewardEditor.name")}
                  readOnly={editing}
                  value={draft.name}
                  onChange={(e) => update({ name: e.target.value })}
                />
              </SectionField>
              <OptionSelect
                label={t("rl.rewardEditor.type")}
                value={rule.type}
                options={RULE_TYPES.map((type) => ({
                  value: type,
                  label: t(`rl.rewardEditor.types.${type}`),
                }))}
                onChange={(type: RuleType) =>
                  update({ rule: defaultRewardRule(type, library) })
                }
              />
            </div>
            <SectionField label={t("rl.rewardEditor.descriptionLabel")}>
              <Input
                dir="auto"
                aria-label={t("rl.rewardEditor.descriptionLabel")}
                value={draft.description}
                onChange={(e) => update({ description: e.target.value })}
              />
            </SectionField>
            {column !== null && (
              <SectionField
                label={t("rl.rewardEditor.referenceColumn")}
                hint={
                  <span className="font-normal text-muted-foreground/80">
                    {t("rl.rewardEditor.referenceHint")}
                  </span>
                }
              >
                <Input
                  aria-label={t("rl.rewardEditor.referenceColumn")}
                  value={column}
                  onChange={(e) => setRule({ compare_to: e.target.value })}
                />
              </SectionField>
            )}
            {rule.type === "regex" && (
              <div className="grid gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                <SectionField label={t("rl.rewardEditor.pattern")}>
                  <Input
                    dir="ltr"
                    aria-label={t("rl.rewardEditor.pattern")}
                    className="font-mono text-xs"
                    value={String(rule.pattern ?? "")}
                    onChange={(e) => setRule({ pattern: e.target.value })}
                  />
                </SectionField>
                <OptionSelect
                  label={t("rl.rewardEditor.matchMode")}
                  value={(rule.mode as string) ?? "fullmatch"}
                  options={[
                    {
                      value: "fullmatch",
                      label: t("rl.rewardEditor.fullmatch"),
                    },
                    { value: "search", label: t("rl.rewardEditor.search") },
                  ]}
                  onChange={(mode) => setRule({ mode })}
                />
              </div>
            )}
            {rule.type === "json_schema" && (
              <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
                <OptionSelect
                  label={t("rl.rewardEditor.jsonType")}
                  value={schema.type}
                  options={["object", "array", "string", "number"].map(
                    (type) => ({ value: type, label: type }),
                  )}
                  onChange={(type) =>
                    setRule({
                      schema: {
                        type,
                        required: type === "object" ? schema.required : [],
                      },
                    })
                  }
                />
                {schema.type === "object" && (
                  <SectionField label={t("rl.rewardEditor.requiredKeys")}>
                    <KeyListInput
                      label={t("rl.rewardEditor.requiredKeys")}
                      keys={schema.required}
                      onChange={(required) =>
                        setRule({ schema: { type: "object", required } })
                      }
                    />
                  </SectionField>
                )}
              </div>
            )}
            {rule.type === "length" && (
              <NumberInput
                label={t("rl.rewardEditor.maxChars")}
                value={rule.max_chars}
                onChange={(max_chars) => setRule({ max_chars })}
              />
            )}
            {rule.type === "numeric" ? (
              <div className="flex flex-col gap-3">
                <p className="text-ui-11p5 text-muted-foreground">
                  {t("rl.rewardEditor.toleranceHint")}
                </p>
                {bands.map((band, i) => (
                  <div
                    // biome-ignore lint/suspicious/noArrayIndexKey: bands have no identity.
                    key={i}
                    className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2"
                  >
                    <NumberInput
                      label={t("rl.rewardEditor.within")}
                      value={band.within}
                      onChange={(within) => setBand(i, { within })}
                    />
                    <NumberInput
                      label={t("rl.rewardEditor.score")}
                      value={band.score}
                      onChange={(value) => setBand(i, { score: value })}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      disabled={bands.length === 1}
                      onClick={() =>
                        setRule({ bands: bands.filter((_, j) => j !== i) })
                      }
                    >
                      {t("rl.rewardEditor.remove")}
                    </Button>
                  </div>
                ))}
                <div className="grid grid-cols-2 items-end gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="justify-self-start"
                    onClick={() =>
                      setRule({ bands: [...bands, { within: "", score: "" }] })
                    }
                  >
                    {t("rl.rewardEditor.addBand")}
                  </Button>
                  <NumberInput
                    label={t("rl.rewardEditor.otherwiseScore")}
                    value={rule.else}
                    onChange={(value) => setRule({ else: value })}
                  />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <NumberInput
                  label={t(
                    `rl.rewardEditor.scores.${hitKey}` as TranslationKey,
                  )}
                  value={score?.[hitKey]}
                  onChange={(value) => setScore(hitKey, value)}
                />
                <NumberInput
                  label={t(
                    `rl.rewardEditor.scores.${missKey}` as TranslationKey,
                  )}
                  value={score?.[missKey]}
                  onChange={(value) => setScore(missKey, value)}
                />
              </div>
            )}
          </TabsContent>
          <TabsContent
            value="extraction"
            className="flex min-h-0 flex-col gap-4 overflow-y-auto"
          >
            {rule.type === "length" ? (
              <p className="text-ui-11p5 text-muted-foreground">
                {t("rl.rewardEditor.noExtraction")}
              </p>
            ) : (
              <ExtractionField
                label={t("rl.rewardEditor.responsePart")}
                value={(rule.extract ?? null) as Extraction}
                onChange={(extract) => setRule({ extract })}
              />
            )}
            {column !== null && (
              <>
                <ExtractionField
                  label={t("rl.rewardEditor.referencePart")}
                  value={(rule.reference_extract ?? null) as Extraction}
                  onChange={(reference_extract) =>
                    setRule({ reference_extract })
                  }
                />
                <NumberInput
                  label={t("rl.rewardEditor.missingScore")}
                  value={rule.missing}
                  onChange={(missing) => setRule({ missing })}
                />
              </>
            )}
            {rule.type === "exact_match" && (
              <SectionField label={t("rl.rewardEditor.normalization")}>
                <div className="grid gap-2 sm:grid-cols-2">
                  {NORMALIZERS.map((n) => (
                    <Label
                      key={n}
                      className="text-xs font-normal text-foreground"
                    >
                      <Checkbox
                        checked={normalize.includes(n)}
                        onCheckedChange={(on) =>
                          setRule({
                            // Order matters to the evaluator, so keep it and append new steps.
                            normalize: on
                              ? [...normalize, n]
                              : normalize.filter((x) => x !== n),
                          })
                        }
                      />
                      {t(`rl.rewardEditor.normalizers.${n}`)}
                    </Label>
                  ))}
                </div>
              </SectionField>
            )}
          </TabsContent>
          <TabsContent
            value="test"
            className="flex min-h-0 flex-col gap-4 overflow-y-auto"
          >
            <SectionField label={t("rl.rewards.tryCompletion")}>
              <Textarea
                rows={6}
                aria-label={t("rl.rewards.tryCompletion")}
                className="font-mono text-xs"
                value={completion}
                onChange={(e) => setCompletion(e.target.value)}
              />
            </SectionField>
            {column !== null && (
              <SectionField label={t("rl.rewards.tryReference")}>
                <Textarea
                  rows={2}
                  aria-label={t("rl.rewards.tryReference")}
                  className="font-mono text-xs"
                  value={reference ?? ""}
                  onChange={(e) => setReferenceEdit(e.target.value)}
                />
              </SectionField>
            )}
            {preview.error ? (
              <p role="alert" className="text-ui-11p5 text-destructive">
                {preview.error}
              </p>
            ) : (
              result && (
                <output
                  aria-live="polite"
                  className="flex items-center gap-2 text-xs text-muted-foreground"
                >
                  {t("rl.rewardEditor.score")}
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {formatScore(result.score)}
                  </span>
                  {result.explanation && (
                    <RewardExplanationHint explanation={result.explanation} />
                  )}
                </output>
              )
            )}
          </TabsContent>
        </Tabs>
        <DialogFooter className="items-center">
          {saveError && (
            <p
              role="alert"
              className="me-auto min-w-0 text-ui-11p5 text-destructive"
            >
              {saveError}
            </p>
          )}
          <Button
            type="button"
            variant="outline"
            disabled={saving}
            onClick={onClose}
          >
            {t("common.cancel")}
          </Button>
          <Button type="button" disabled={saving} onClick={save}>
            {t(saving ? "common.saving" : "common.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
