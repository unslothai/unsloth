// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ComponentProps,
  type ReactNode,
  useId,
  useRef,
  useState,
} from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
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

type FieldProps = { label: string; hint?: string };

function Field({
  label,
  hint,
  children,
}: FieldProps & { children: (id: string) => ReactNode }) {
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

function InputField({
  label,
  hint,
  ...props
}: FieldProps & ComponentProps<typeof Input>) {
  return (
    <Field label={label} hint={hint}>
      {(id) => <Input id={id} {...props} />}
    </Field>
  );
}

function TextareaField({
  label,
  hint,
  ...props
}: FieldProps & ComponentProps<typeof Textarea>) {
  return (
    <Field label={label} hint={hint}>
      {(id) => <Textarea id={id} className="font-mono text-xs" {...props} />}
    </Field>
  );
}

function NumberField({
  integer = false,
  value,
  onChange,
  ...props
}: FieldProps & {
  value: DraftNumber | undefined;
  onChange: (value: string) => void;
  integer?: boolean;
}) {
  return (
    <InputField
      {...props}
      type="number"
      step={integer ? 1 : "any"}
      required
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value)}
      className="tabular-nums"
    />
  );
}

function SelectField({
  value,
  options,
  onChange,
  ...props
}: FieldProps & {
  value: string;
  options: [string, string][];
  onChange: (value: string) => void;
}) {
  return (
    <Field {...props}>
      {(id) => (
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger id={id} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map(([option, label]) => (
              <SelectItem key={option} value={option}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </Field>
  );
}

const SCORE_FIELDS = {
  length: [
    ["under", "atLimitScore"],
    ["over", "overLimitScore"],
  ],
  other: [
    ["match", "matchScore"],
    ["miss", "missScore"],
  ],
} as const;

function ActionButton(props: ComponentProps<typeof Button>) {
  return <Button type="button" variant="ghost" {...props} />;
}

const replaceAt = <T,>(list: T[], index: number, value: T) =>
  list.map((item, i) => (i === index ? value : item));
const removeAt = <T,>(list: T[], index: number) =>
  list.filter((_, i) => i !== index);

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
  return (
    <div className="flex flex-col gap-3">
      <SelectField
        label={label}
        value={!value ? "whole" : "between" in value ? "between" : "regex"}
        options={[
          ["whole", t("rl.rewardEditor.wholeText")],
          ["between", t("rl.rewardEditor.between")],
          ["regex", t("rl.rewardEditor.regex")],
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
          {(["afterText", "beforeText"] as const).map((key, index) => (
            <InputField
              key={key}
              label={t(`rl.rewardEditor.${key}`)}
              value={value.between[index]}
              dir="auto"
              required
              onChange={(e) =>
                onChange({
                  between: replaceAt(value.between, index, e.target.value) as [
                    string,
                    string,
                  ],
                })
              }
              className="font-mono text-xs"
            />
          ))}
        </div>
      )}
      {value && "regex" in value && (
        <TextareaField
          label={t("rl.rewardEditor.pattern")}
          hint={t("rl.rewardEditor.captureHint")}
          dir="ltr"
          rows={2}
          required
          value={value.regex}
          onChange={(e) => onChange({ regex: e.target.value })}
        />
      )}
    </div>
  );
}

function RuleFields({
  rule,
  library,
  onChange,
}: {
  rule: DraftRule;
  library: RewardRecord[];
  onChange: (rule: DraftRule) => void;
}) {
  const t = useT();
  const compareTo = rewardReferenceColumn(rule);
  const updateRule = (patch: Partial<DraftRule>) =>
    onChange({ ...rule, ...patch });
  const bands = rule.bands ?? [];
  const requiredKeys = rule.schema?.required ?? [];
  const setRequiredKeys = (required: string[]) =>
    updateRule({ schema: { type: "object", required } });
  return (
    <div className="space-y-4 border-t border-border/60 pt-5">
      <SelectField
        label={t("rl.rewardEditor.type")}
        value={rule.type}
        options={RULE_TYPES.map((type) => [
          type,
          t(`rl.rewardEditor.types.${type}`),
        ])}
        onChange={(type) =>
          onChange(defaultRewardRule(type as RuleType, library))
        }
      />
      {compareTo !== null && (
        <InputField
          label={t("rl.rewardEditor.referenceColumn")}
          hint={t("rl.rewardEditor.referenceHint")}
          dir="auto"
          required
          value={rule.compare_to ?? ""}
          onChange={(e) => updateRule({ compare_to: e.target.value })}
        />
      )}
      {rule.type === "regex" && (
        <>
          <TextareaField
            label={t("rl.rewardEditor.pattern")}
            hint={t("rl.rewardEditor.patternHint")}
            rows={3}
            dir="ltr"
            required
            value={rule.pattern ?? ""}
            onChange={(e) => updateRule({ pattern: e.target.value })}
          />
          <SelectField
            label={t("rl.rewardEditor.matchMode")}
            value={rule.mode ?? "fullmatch"}
            options={[
              ["fullmatch", t("rl.rewardEditor.fullmatch")],
              ["search", t("rl.rewardEditor.search")],
            ]}
            onChange={(mode) =>
              updateRule({ mode: mode as "fullmatch" | "search" })
            }
          />
        </>
      )}
      {rule.type === "json_schema" && (
        <>
          <SelectField
            label={t("rl.rewardEditor.jsonType")}
            hint={t("rl.rewardEditor.jsonHint")}
            value={rule.schema?.type ?? "object"}
            options={["object", "array", "string", "number"].map((type) => [
              type,
              t(`rl.rewardEditor.jsonTypes.${type}` as TranslationKey),
            ])}
            onChange={(type) =>
              updateRule({
                schema: {
                  type,
                  required: type === "object" ? requiredKeys : [],
                },
              })
            }
          />
          {(rule.schema?.type ?? "object") === "object" && (
            <div className="space-y-2">
              <p className="text-xs font-medium">
                {t("rl.rewardEditor.requiredKeys")}
              </p>
              {requiredKeys.map((key, index) => (
                <div className="flex gap-2" key={index}>
                  <Input
                    aria-label={t("rl.rewardEditor.keyNumber", {
                      number: index + 1,
                    })}
                    dir="auto"
                    value={key}
                    onChange={(e) =>
                      setRequiredKeys(
                        replaceAt(requiredKeys, index, e.target.value),
                      )
                    }
                  />
                  <ActionButton
                    onClick={() =>
                      setRequiredKeys(removeAt(requiredKeys, index))
                    }
                  >
                    {t("rl.rewardEditor.remove")}
                  </ActionButton>
                </div>
              ))}
              <ActionButton
                variant="outline"
                size="sm"
                disabled={requiredKeys.length >= 64}
                onClick={() => setRequiredKeys([...requiredKeys, ""])}
              >
                {t("rl.rewardEditor.addKey")}
              </ActionButton>
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
          {bands.map((band, index) => (
            <div
              key={index}
              className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2"
            >
              {(["within", "score"] as const).map((key) => (
                <NumberField
                  key={key}
                  label={t(`rl.rewardEditor.${key}`)}
                  value={band[key]}
                  onChange={(value) =>
                    updateRule({
                      bands: replaceAt(bands, index, { ...band, [key]: value }),
                    })
                  }
                />
              ))}
              <ActionButton
                disabled={bands.length === 1}
                onClick={() => updateRule({ bands: removeAt(bands, index) })}
              >
                {t("rl.rewardEditor.remove")}
              </ActionButton>
            </div>
          ))}
          <ActionButton
            variant="outline"
            size="sm"
            onClick={() =>
              updateRule({ bands: [...bands, { within: "", score: "" }] })
            }
          >
            {t("rl.rewardEditor.addBand")}
          </ActionButton>
          <NumberField
            label={t("rl.rewardEditor.otherwiseScore")}
            value={rule.else}
            onChange={(value) => updateRule({ else: value })}
          />
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {SCORE_FIELDS[rule.type === "length" ? "length" : "other"].map(
            ([key, label]) => (
              <NumberField
                key={key}
                label={t(`rl.rewardEditor.${label}`)}
                value={rule.score?.[key]}
                onChange={(value) =>
                  updateRule({
                    score: { ...rule.score, [key]: value },
                  })
                }
              />
            ),
          )}
        </div>
      )}
    </div>
  );
}

function ExtractionTab({
  rule,
  updateRule,
}: {
  rule: DraftRule;
  updateRule: (patch: Partial<DraftRule>) => void;
}) {
  const t = useT();
  const compareTo = rewardReferenceColumn(rule);
  const extractable = compareTo !== null || rule.type === "json_schema";
  const normalize = rule.normalize ?? [];
  return (
    <>
      <div className="space-y-2">
        <h3 className="text-sm font-medium">
          {t("rl.rewardEditor.extraction")}
        </h3>
        <p className="text-xs text-muted-foreground">
          {t(
            extractable
              ? "rl.rewardEditor.extractionHint"
              : "rl.rewardEditor.noExtraction",
          )}
        </p>
      </div>
      {extractable && (
        <ExtractionFields
          label={t("rl.rewardEditor.responsePart")}
          value={rule.extract}
          onChange={(extract) => updateRule({ extract })}
        />
      )}
      {compareTo !== null && (
        <>
          <ExtractionFields
            label={t("rl.rewardEditor.referencePart")}
            value={rule.reference_extract}
            onChange={(reference_extract) => updateRule({ reference_extract })}
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
          {normalize.map((normalizer, index) => (
            <div className="flex flex-wrap items-center gap-1" key={index}>
              <span className="me-auto text-xs">
                {t(
                  `rl.rewardEditor.normalizers.${normalizer}` as TranslationKey,
                )}
              </span>
              <ActionButton
                size="sm"
                disabled={index === 0}
                onClick={() =>
                  updateRule({
                    normalize: replaceAt(
                      replaceAt(normalize, index, normalize[index - 1]),
                      index - 1,
                      normalizer,
                    ),
                  })
                }
              >
                {t("rl.rewardEditor.moveUp")}
              </ActionButton>
              <ActionButton
                size="sm"
                onClick={() =>
                  updateRule({
                    normalize: removeAt(normalize, index),
                  })
                }
              >
                {t("rl.rewardEditor.remove")}
              </ActionButton>
            </div>
          ))}
          <Select
            value=""
            onValueChange={(normalizer) =>
              updateRule({ normalize: [...normalize, normalizer] })
            }
            disabled={NORMALIZERS.every((n) => normalize.includes(n))}
          >
            <SelectTrigger
              aria-label={t("rl.rewardEditor.addNormalization")}
              className="w-full"
            >
              <SelectValue
                placeholder={t("rl.rewardEditor.addNormalization")}
              />
            </SelectTrigger>
            <SelectContent>
              {NORMALIZERS.filter((n) => !normalize.includes(n)).map((n) => (
                <SelectItem key={n} value={n}>
                  {t(`rl.rewardEditor.normalizers.${n}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </>
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
    validationError = validationCode
      ? t(`rl.rewardEditor.${validationCode}`)
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
      onSaved(await importReward(markdown, editing));
      onClose();
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };
  const error =
    saveError ??
    (previewMarkdown && !normalizedName
      ? attemptedSave
        ? t("rl.rewardEditor.nameToSave")
        : null
      : validationError);
  const runLabel = t(
    preview.pending ? "rl.rewardEditor.scoring" : "rl.rewardEditor.run",
  );
  const tabClass = "min-h-0 space-y-5 overflow-y-auto px-7 py-5";

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
                  {(["configuration", "extraction", "test"] as const).map(
                    (tab) => (
                      <TabsTrigger
                        key={tab}
                        value={tab}
                        indicatorClassName="pointer-events-none inset-x-0 top-auto bottom-0 h-0.5 rounded-full bg-foreground"
                        className="relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 text-muted-foreground transition-colors hover:text-foreground data-active:font-semibold data-active:text-foreground"
                      >
                        {t(`rl.rewardEditor.${tab}Tab`)}
                      </TabsTrigger>
                    ),
                  )}
                </TabsList>
                <TabsContent value="configuration" className={tabClass}>
                  <div className="space-y-4">
                    <InputField
                      label={t("rl.rewardEditor.name")}
                      hint={t("rl.rewardEditor.nameHint")}
                      ref={nameInput}
                      dir="ltr"
                      autoComplete="off"
                      required
                      readOnly={editing}
                      value={draft.name}
                      onChange={(e) => update({ name: e.target.value })}
                      aria-invalid={
                        collision ||
                        (attemptedSave &&
                          (!normalizedName || validationCode === "invalidName"))
                      }
                    />
                    <InputField
                      label={t("rl.rewardEditor.optionalDescription")}
                      dir="auto"
                      value={draft.description}
                      onChange={(e) => update({ description: e.target.value })}
                    />
                  </div>
                  <RuleFields
                    rule={rule}
                    library={library}
                    onChange={(next) => update({ rule: next })}
                  />
                </TabsContent>
                <TabsContent value="extraction" className={tabClass}>
                  <ExtractionTab rule={rule} updateRule={updateRule} />
                </TabsContent>
                <TabsContent
                  value="test"
                  className="min-h-0 space-y-4 overflow-y-auto px-7 py-5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <h3 className="text-sm font-medium">
                      {t("rl.rewardEditor.testTitle")}
                    </h3>
                    <Button
                      type="button"
                      size="icon-sm"
                      variant="outline"
                      aria-label={runLabel}
                      title={runLabel}
                      onClick={() => void preview.run()}
                      disabled={!previewMarkdown || preview.pending}
                    >
                      {preview.pending ? (
                        <Spinner className="size-3.5" />
                      ) : (
                        <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
                      )}
                    </Button>
                  </div>
                  <TextareaField
                    label={t("rl.rewardEditor.responseText")}
                    rows={4}
                    dir="auto"
                    value={completion}
                    onChange={(e) => setCompletion(e.target.value)}
                  />
                  {compareTo !== null && (
                    <TextareaField
                      label={
                        fromRow?.source && !manual
                          ? t("rl.rewardEditor.fromColumn", {
                              column: fromRow.source,
                            })
                          : t("rl.rewardEditor.manualReference")
                      }
                      rows={2}
                      dir="auto"
                      value={reference ?? ""}
                      onChange={(e) =>
                        setReferences({
                          ...references,
                          [compareTo]: e.target.value,
                        })
                      }
                    />
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
