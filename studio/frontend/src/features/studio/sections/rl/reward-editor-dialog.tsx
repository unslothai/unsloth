// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useId, useRef, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import {
  Field as FormField,
  FieldDescription,
  FieldLabel,
} from "@/components/ui/field";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { Spinner } from "@/components/ui/spinner";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type RewardRecord,
  type DraftNumber,
  type DraftRule,
  type Extraction,
  type RewardDraft,
  type RuleType,
  importReward,
  useRewardPreview,
  NORMALIZERS,
  RULE_TYPES,
  RewardDraftError,
  SAMPLE_REWARD_COMPLETION,
  copyRewardName,
  defaultRewardRule,
  rewardDraftMarkdown,
  rewardReference,
  rewardReferenceColumn,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { RewardTestResult } from "./reward-test-result";

function Field({
  label,
  children,
  hint,
}: { label: string; children: (id: string) => ReactNode; hint?: string }) {
  const id = useId();
  return (
    <FormField className="min-w-0 gap-1.5">
      <FieldLabel htmlFor={id} className="text-xs">
        {label}
      </FieldLabel>
      {children(id)}
      {hint && (
        <FieldDescription className="text-start text-xs">
          {hint}
        </FieldDescription>
      )}
    </FormField>
  );
}

function NumberField({
  label,
  value,
  onChange,
  integer = false,
}: {
  label: string;
  value: DraftNumber | undefined;
  onChange: (value: string) => void;
  integer?: boolean;
}) {
  return (
    <Field label={label}>
      {(id) => (
        <Input
          id={id}
          type="number"
          step={integer ? 1 : "any"}
          required
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          className="tabular-nums"
        />
      )}
    </Field>
  );
}

function ExtractionFields({
  label,
  value,
  onChange,
}: {
  label: string;
  value: Extraction | null | undefined;
  onChange: (value: Extraction | null) => void;
}) {
  const t = useT();
  const mode = !value ? "whole" : "between" in value ? "between" : "regex";
  return (
    <div className="flex flex-col gap-3">
      <Field label={label}>
        {(id) => (
          <Select
            value={mode}
            onValueChange={(next) =>
              onChange(
                next === "whole"
                  ? null
                  : next === "between"
                    ? { between: ["", ""] }
                    : { regex: "" },
              )
            }
          >
            <SelectTrigger id={id} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="whole">
                {t("rl.rewardEditor.wholeText")}
              </SelectItem>
              <SelectItem value="between">
                {t("rl.rewardEditor.between")}
              </SelectItem>
              <SelectItem value="regex">
                {t("rl.rewardEditor.regex")}
              </SelectItem>
            </SelectContent>
          </Select>
        )}
      </Field>
      {value && "between" in value && (
        <div className="grid gap-3 sm:grid-cols-2">
          {[0, 1].map((index) => (
            <Field
              key={index}
              label={t(
                index === 0
                  ? "rl.rewardEditor.afterText"
                  : "rl.rewardEditor.beforeText",
              )}
            >
              {(id) => (
                <Input
                  id={id}
                  value={value.between[index]}
                  dir="auto"
                  required
                  onChange={(e) => {
                    const between: [string, string] = [...value.between];
                    between[index] = e.target.value;
                    onChange({ between });
                  }}
                  className="font-mono text-xs"
                />
              )}
            </Field>
          ))}
        </div>
      )}
      {value && "regex" in value && (
        <Field
          label={t("rl.rewardEditor.pattern")}
          hint={t("rl.rewardEditor.captureHint")}
        >
          {(id) => (
            <Textarea
              id={id}
              dir="ltr"
              rows={2}
              required
              className="font-mono text-xs"
              value={value.regex}
              onChange={(e) => onChange({ regex: e.target.value })}
            />
          )}
        </Field>
      )}
    </div>
  );
}

export function RewardEditorDialog({
  record,
  library,
  previewRow,
  mapping,
  onClose,
  onSaved,
}: {
  record?: RewardRecord;
  library: RewardRecord[];
  previewRow: Record<string, unknown> | null;
  mapping: Record<string, string>;
  onClose: () => void;
  onSaved: (record: RewardRecord) => void;
}) {
  const t = useT();
  const editing = record?.source === "user";
  const [draft, setDraft] = useState<RewardDraft>(() => ({
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
  const initial = useRef(JSON.stringify(draft));
  const [completion, setCompletion] = useState(SAMPLE_REWARD_COMPLETION);
  const [references, setReferences] = useState<Record<string, string>>({});
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [discardOpen, setDiscardOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("configuration");
  const nameInput = useRef<HTMLInputElement>(null);
  const [attemptedSave, setAttemptedSave] = useState(false);
  const formId = useId();
  const errorId = useId();
  const rule = draft.rule;
  const compareTo = rewardReferenceColumn(rule);
  const fromRow = compareTo
    ? rewardReference(compareTo, previewRow, mapping)
    : null;
  const manual = compareTo !== null && Object.hasOwn(references, compareTo);
  const reference = compareTo
    ? (references[compareTo] ??
      (fromRow?.source ? fromRow.value : compareTo === "answer" ? "72" : null))
    : null;
  const normalizedName = draft.name.normalize("NFKC").trim().toLowerCase();
  const collision = library.some(
    (r) => r.name === normalizedName && !(editing && r.name === record.name),
  );
  let markdown: string | null = null;
  let previewMarkdown: string | null = null;
  let validationError: string | null = null;
  let validationCode: RewardDraftError["code"] | null = null;
  try {
    // The preview API requires a manifest name, but does not persist the draft.
    previewMarkdown = rewardDraftMarkdown({ ...draft, name: "draft-preview" });
    markdown = rewardDraftMarkdown(draft);
    if (collision) validationError = t("rl.rewardEditor.duplicate");
  } catch (error) {
    validationCode = error instanceof RewardDraftError ? error.code : null;
    validationError =
      error instanceof RewardDraftError
        ? t(("rl.rewardEditor." + error.code) as TranslationKey)
        : String(error);
  }
  const preview = useRewardPreview(
    previewMarkdown
      ? [{ rewards: [{ markdown: previewMarkdown, weight: 1 }], reference }]
      : [],
    completion,
  );
  const update = (patch: Partial<RewardDraft>) => {
    setDraft((current) => ({ ...current, ...patch }));
    setSaveError(null);
  };
  const updateRule = (patch: Partial<DraftRule>) =>
    update({ rule: { ...rule, ...patch } });
  const updateScore = (key: string, value: string) =>
    updateRule({ score: { ...rule.score, [key]: value } });
  const close = () => {
    if (saving) return;
    if (JSON.stringify(draft) !== initial.current) setDiscardOpen(true);
    else onClose();
  };
  const save = async () => {
    if (saving) return;
    setAttemptedSave(true);
    if (!normalizedName || !markdown || validationError) {
      if (!normalizedName || validationCode === "invalidName" || collision) {
        setActiveTab("configuration");
        requestAnimationFrame(() => nameInput.current?.focus());
      } else if (validationCode === "invalidExtraction") {
        setActiveTab("extraction");
      }
      return;
    }
    setSaving(true);
    try {
      const saved = await importReward(markdown, editing);
      onSaved(saved);
      onClose();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveError(message);
    } finally {
      setSaving(false);
    }
  };
  const referenceLabel =
    fromRow?.source && !manual
      ? t("rl.rewardEditor.fromColumn", { column: fromRow.source })
      : t("rl.rewardEditor.manualReference");
  const error =
    saveError ??
    (!previewMarkdown
      ? validationError
      : attemptedSave && !normalizedName
        ? t("rl.rewardEditor.nameToSave")
        : attemptedSave || normalizedName
          ? validationError
          : null);

  return (
    <>
      <Dialog
        open
        onOpenChange={(open) => {
          if (!open) close();
        }}
      >
        <DialogContent
          className="flex h-[min(44rem,calc(100dvh-2rem))] flex-col gap-0 overflow-hidden p-0 sm:max-w-2xl"
          onCloseAutoFocus={(event) => event.preventDefault()}
        >
          <DialogHeader className="shrink-0 px-7 pt-8 pb-5 text-start">
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
          <form
            id={formId}
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              void save();
            }}
            className="flex min-h-0 flex-1 flex-col"
            aria-describedby={error ? errorId : undefined}
          >
            <fieldset
              disabled={saving}
              className="flex min-h-0 flex-1 flex-col"
            >
              <Tabs
                value={activeTab}
                onValueChange={setActiveTab}
                className="min-h-0 flex-1 gap-0"
              >
                <TabsList
                  unstyled
                  aria-label={t("rl.rewards.title")}
                  className="w-full shrink-0 justify-center gap-5 overflow-x-auto border-b border-border/60 px-7 pb-px text-ui-13 tracking-nav sm:gap-6"
                >
                  {[
                    {
                      value: "configuration",
                      label: t("rl.rewardEditor.configurationTab"),
                    },
                    {
                      value: "extraction",
                      label: t("rl.rewardEditor.extractionTab"),
                    },
                    { value: "test", label: t("rl.rewardEditor.testTab") },
                  ].map((item) => (
                    <TabsTrigger
                      key={item.value}
                      value={item.value}
                      indicatorClassName="pointer-events-none inset-x-0 top-auto bottom-0 h-0.5 rounded-full bg-foreground"
                      className={cn(
                        "relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 transition-colors",
                        activeTab === item.value
                          ? "font-semibold text-foreground"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {item.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
                <TabsContent
                  value="configuration"
                  className="min-h-0 space-y-5 overflow-y-auto px-7 py-5"
                >
                  <div className="space-y-4">
                    <Field
                      label={t("rl.rewardEditor.name")}
                      hint={t("rl.rewardEditor.nameHint")}
                    >
                      {(id) => (
                        <Input
                          ref={nameInput}
                          id={id}
                          dir="ltr"
                          autoComplete="off"
                          required
                          readOnly={editing}
                          value={draft.name}
                          onChange={(e) => update({ name: e.target.value })}
                          aria-invalid={Boolean(
                            collision ||
                              (attemptedSave &&
                                (!normalizedName ||
                                  validationCode === "invalidName")),
                          )}
                        />
                      )}
                    </Field>
                    <Field label={t("rl.rewardEditor.optionalDescription")}>
                      {(id) => (
                        <Input
                          id={id}
                          dir="auto"
                          value={draft.description}
                          onChange={(e) =>
                            update({ description: e.target.value })
                          }
                        />
                      )}
                    </Field>
                  </div>
                  <div className="space-y-4 border-t border-border/60 pt-5">
                    <Field label={t("rl.rewardEditor.type")}>
                      {(id) => (
                        <Select
                          value={rule.type}
                          onValueChange={(type) =>
                            update({
                              rule: defaultRewardRule(
                                type as RuleType,
                                library,
                              ),
                            })
                          }
                        >
                          <SelectTrigger id={id} className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {RULE_TYPES.map((type) => (
                              <SelectItem key={type} value={type}>
                                {t(
                                  ("rl.rewardEditor.types." +
                                    type) as TranslationKey,
                                )}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                    </Field>
                    {compareTo !== null && (
                      <Field
                        label={t("rl.rewardEditor.referenceColumn")}
                        hint={t("rl.rewardEditor.referenceHint")}
                      >
                        {(id) => (
                          <Input
                            id={id}
                            dir="auto"
                            required
                            value={rule.compare_to ?? ""}
                            onChange={(e) =>
                              updateRule({ compare_to: e.target.value })
                            }
                          />
                        )}
                      </Field>
                    )}
                    {rule.type === "regex" && (
                      <>
                        <Field
                          label={t("rl.rewardEditor.pattern")}
                          hint={t("rl.rewardEditor.patternHint")}
                        >
                          {(id) => (
                            <Textarea
                              id={id}
                              rows={3}
                              dir="ltr"
                              required
                              className="font-mono text-xs"
                              value={rule.pattern ?? ""}
                              onChange={(e) =>
                                updateRule({ pattern: e.target.value })
                              }
                            />
                          )}
                        </Field>
                        <Field label={t("rl.rewardEditor.matchMode")}>
                          {(id) => (
                            <Select
                              value={rule.mode ?? "fullmatch"}
                              onValueChange={(mode) =>
                                updateRule({
                                  mode: mode as "fullmatch" | "search",
                                })
                              }
                            >
                              <SelectTrigger id={id} className="w-full">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="fullmatch">
                                  {t("rl.rewardEditor.fullmatch")}
                                </SelectItem>
                                <SelectItem value="search">
                                  {t("rl.rewardEditor.search")}
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          )}
                        </Field>
                      </>
                    )}
                    {rule.type === "json_schema" && (
                      <>
                        <Field
                          label={t("rl.rewardEditor.jsonType")}
                          hint={t("rl.rewardEditor.jsonHint")}
                        >
                          {(id) => (
                            <Select
                              value={rule.schema?.type ?? "object"}
                              onValueChange={(type) =>
                                updateRule({
                                  schema: {
                                    type,
                                    required:
                                      type === "object"
                                        ? (rule.schema?.required ?? [])
                                        : [],
                                  },
                                })
                              }
                            >
                              <SelectTrigger id={id} className="w-full">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {["object", "array", "string", "number"].map(
                                  (type) => (
                                    <SelectItem key={type} value={type}>
                                      {t(
                                        ("rl.rewardEditor.jsonTypes." +
                                          type) as TranslationKey,
                                      )}
                                    </SelectItem>
                                  ),
                                )}
                              </SelectContent>
                            </Select>
                          )}
                        </Field>
                        {(rule.schema?.type ?? "object") === "object" && (
                          <div className="space-y-2">
                            <p className="text-xs font-medium">
                              {t("rl.rewardEditor.requiredKeys")}
                            </p>
                            {(rule.schema?.required ?? []).map((key, index) => (
                              <div className="flex gap-2" key={index}>
                                <Input
                                  aria-label={t("rl.rewardEditor.keyNumber", {
                                    number: index + 1,
                                  })}
                                  dir="auto"
                                  value={key}
                                  onChange={(e) => {
                                    const required = [
                                      ...(rule.schema?.required ?? []),
                                    ];
                                    required[index] = e.target.value;
                                    updateRule({
                                      schema: { type: "object", required },
                                    });
                                  }}
                                />
                                <Button
                                  variant="ghost"
                                  type="button"
                                  onClick={() =>
                                    updateRule({
                                      schema: {
                                        type: "object",
                                        required: rule.schema!.required.filter(
                                          (_, i) => i !== index,
                                        ),
                                      },
                                    })
                                  }
                                >
                                  {t("rl.rewardEditor.remove")}
                                </Button>
                              </div>
                            ))}
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              disabled={
                                (rule.schema?.required.length ?? 0) >= 64
                              }
                              onClick={() =>
                                updateRule({
                                  schema: {
                                    type: "object",
                                    required: [
                                      ...(rule.schema?.required ?? []),
                                      "",
                                    ],
                                  },
                                })
                              }
                            >
                              {t("rl.rewardEditor.addKey")}
                            </Button>
                          </div>
                        )}
                      </>
                    )}
                    {rule.type === "length" && (
                      <NumberField
                        label={t("rl.rewardEditor.maxCharacters")}
                        integer
                        value={rule.max_chars}
                        onChange={(max_chars) => updateRule({ max_chars })}
                      />
                    )}
                    {rule.type === "numeric" ? (
                      <div className="space-y-3">
                        <p className="text-xs text-muted-foreground">
                          {t("rl.rewardEditor.toleranceHint")}
                        </p>
                        {(rule.bands ?? []).map((band, index) => (
                          <div
                            key={index}
                            className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2"
                          >
                            <NumberField
                              label={t("rl.rewardEditor.within")}
                              value={band.within}
                              onChange={(within) =>
                                updateRule({
                                  bands: rule.bands!.map((b, i) =>
                                    i === index ? { ...b, within } : b,
                                  ),
                                })
                              }
                            />
                            <NumberField
                              label={t("rl.rewardEditor.score")}
                              value={band.score}
                              onChange={(score) =>
                                updateRule({
                                  bands: rule.bands!.map((b, i) =>
                                    i === index ? { ...b, score } : b,
                                  ),
                                })
                              }
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              disabled={rule.bands!.length === 1}
                              onClick={() =>
                                updateRule({
                                  bands: rule.bands!.filter(
                                    (_, i) => i !== index,
                                  ),
                                })
                              }
                            >
                              {t("rl.rewardEditor.remove")}
                            </Button>
                          </div>
                        ))}
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            updateRule({
                              bands: [
                                ...(rule.bands ?? []),
                                { within: "", score: "" },
                              ],
                            })
                          }
                        >
                          {t("rl.rewardEditor.addBand")}
                        </Button>
                        <NumberField
                          label={t("rl.rewardEditor.otherwiseScore")}
                          value={rule.else}
                          onChange={(value) => updateRule({ else: value })}
                        />
                      </div>
                    ) : (
                      <div className="grid gap-3 sm:grid-cols-2">
                        <NumberField
                          label={t(
                            rule.type === "length"
                              ? "rl.rewardEditor.atLimitScore"
                              : "rl.rewardEditor.matchScore",
                          )}
                          value={
                            rule.score?.[
                              rule.type === "length" ? "under" : "match"
                            ]
                          }
                          onChange={(value) =>
                            updateScore(
                              rule.type === "length" ? "under" : "match",
                              value,
                            )
                          }
                        />
                        <NumberField
                          label={t(
                            rule.type === "length"
                              ? "rl.rewardEditor.overLimitScore"
                              : "rl.rewardEditor.missScore",
                          )}
                          value={
                            rule.score?.[
                              rule.type === "length" ? "over" : "miss"
                            ]
                          }
                          onChange={(value) =>
                            updateScore(
                              rule.type === "length" ? "over" : "miss",
                              value,
                            )
                          }
                        />
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent
                  value="extraction"
                  className="min-h-0 space-y-5 overflow-y-auto px-7 py-5"
                >
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">
                      {t("rl.rewardEditor.extraction")}
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      {t(
                        compareTo !== null || rule.type === "json_schema"
                          ? "rl.rewardEditor.extractionHint"
                          : "rl.rewardEditor.noExtraction",
                      )}
                    </p>
                  </div>
                  {(compareTo !== null || rule.type === "json_schema") && (
                    <>
                      <ExtractionFields
                        label={t("rl.rewardEditor.responsePart")}
                        value={rule.extract}
                        onChange={(extract) => updateRule({ extract })}
                      />
                      {compareTo !== null && (
                        <>
                          <ExtractionFields
                            label={t("rl.rewardEditor.referencePart")}
                            value={rule.reference_extract}
                            onChange={(reference_extract) =>
                              updateRule({ reference_extract })
                            }
                          />
                          <NumberField
                            label={t("rl.rewardEditor.missingScore")}
                            value={rule.missing}
                            onChange={(missing) => updateRule({ missing })}
                          />
                        </>
                      )}
                      {rule.type === "exact_match" && (
                        <div className="space-y-2">
                          <p className="text-xs font-medium">
                            {t("rl.rewardEditor.normalization")}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {t("rl.rewardEditor.normalizationHint")}
                          </p>
                          {(rule.normalize ?? []).map((normalizer, index) => (
                            <div
                              className="flex flex-wrap items-center gap-1"
                              key={index}
                            >
                              <span className="me-auto text-xs">
                                {t(
                                  ("rl.rewardEditor.normalizers." +
                                    normalizer) as TranslationKey,
                                )}
                              </span>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                disabled={index === 0}
                                onClick={() => {
                                  const normalize = [...rule.normalize!];
                                  [normalize[index - 1], normalize[index]] = [
                                    normalize[index],
                                    normalize[index - 1],
                                  ];
                                  updateRule({ normalize });
                                }}
                              >
                                {t("rl.rewardEditor.moveUp")}
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                onClick={() =>
                                  updateRule({
                                    normalize: rule.normalize!.filter(
                                      (_, i) => i !== index,
                                    ),
                                  })
                                }
                              >
                                {t("rl.rewardEditor.remove")}
                              </Button>
                            </div>
                          ))}
                          <Select
                            value=""
                            onValueChange={(normalizer) =>
                              updateRule({
                                normalize: [
                                  ...(rule.normalize ?? []),
                                  normalizer,
                                ],
                              })
                            }
                            disabled={NORMALIZERS.every((n) =>
                              rule.normalize?.includes(n),
                            )}
                          >
                            <SelectTrigger
                              aria-label={t("rl.rewardEditor.addNormalization")}
                              className="w-full"
                            >
                              <SelectValue
                                placeholder={t(
                                  "rl.rewardEditor.addNormalization",
                                )}
                              />
                            </SelectTrigger>
                            <SelectContent>
                              {NORMALIZERS.filter(
                                (n) => !rule.normalize?.includes(n),
                              ).map((n) => (
                                <SelectItem key={n} value={n}>
                                  {t(
                                    ("rl.rewardEditor.normalizers." +
                                      n) as TranslationKey,
                                  )}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </>
                  )}
                </TabsContent>
                <TabsContent
                  value="test"
                  className="min-h-0 space-y-4 overflow-y-auto px-7 py-5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium">
                        {t("rl.rewardEditor.testTitle")}
                      </h3>
                    </div>
                    <Tooltip>
                      <TooltipTrigger asChild disableClickToggle>
                        <Button
                          type="button"
                          size="icon-sm"
                          variant="outline"
                          aria-label={t(
                            preview?.pending
                              ? "rl.rewardEditor.scoring"
                              : "rl.rewardEditor.run",
                          )}
                          onClick={() => void preview.run()}
                          disabled={!previewMarkdown || preview?.pending}
                        >
                          {preview?.pending ? (
                            <Spinner className="size-3.5" />
                          ) : (
                            <HugeiconsIcon
                              icon={PlayIcon}
                              className="size-3.5"
                            />
                          )}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        {t(
                          preview?.pending
                            ? "rl.rewardEditor.scoring"
                            : "rl.rewardEditor.run",
                        )}
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Field label={t("rl.rewardEditor.responseText")}>
                    {(id) => (
                      <Textarea
                        id={id}
                        rows={4}
                        dir="auto"
                        className="font-mono text-xs"
                        value={completion}
                        onChange={(e) => setCompletion(e.target.value)}
                      />
                    )}
                  </Field>
                  {compareTo !== null && (
                    <Field label={referenceLabel}>
                      {(id) => (
                        <Textarea
                          id={id}
                          rows={2}
                          dir="auto"
                          className="font-mono text-xs"
                          value={reference ?? ""}
                          onChange={(e) =>
                            setReferences({
                              ...references,
                              [compareTo]: e.target.value,
                            })
                          }
                        />
                      )}
                    </Field>
                  )}
                  <RewardTestResult
                    score={preview.data?.total}
                    error={preview.error}
                    explanation={preview.data?.scores[0]?.explanation}
                  />
                </TabsContent>
              </Tabs>
            </fieldset>
          </form>
          <div className="flex shrink-0 flex-col gap-3 border-t border-border px-7 pt-4 pb-7 sm:flex-row sm:items-center sm:justify-between">
            {error && (
              <p
                id={errorId}
                role="alert"
                className="min-w-0 text-xs text-destructive sm:flex-1"
              >
                {error}
              </p>
            )}
            <DialogFooter className="m-0 shrink-0 sm:ms-auto">
              <Button
                type="button"
                variant="outline"
                disabled={saving}
                onClick={close}
              >
                {t("rl.rewardEditor.cancel")}
              </Button>
              <Button type="submit" form={formId} disabled={saving}>
                {t(saving ? "rl.rewardEditor.saving" : "rl.rewardEditor.save")}
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
      <AlertDialog open={discardOpen} onOpenChange={setDiscardOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("rl.rewardEditor.discardTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("rl.rewardEditor.discardDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t("rl.rewardEditor.keepEditing")}
            </AlertDialogCancel>
            <AlertDialogAction onClick={onClose}>
              {t("rl.rewardEditor.discard")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
