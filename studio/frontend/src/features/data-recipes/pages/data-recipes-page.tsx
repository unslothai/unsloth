// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
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
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { ShineBorder } from "@/components/ui/shine-border";
import { useI18n } from "@/features/i18n";
import type { TranslationKey } from "@/features/i18n/messages";
import { toastError } from "@/shared/toast";
import {
  Album02Icon,
  ArrowDown01Icon,
  CodeIcon,
  CookBookIcon,
  Database02Icon,
  Delete02Icon,
  DocumentAttachmentIcon,
  FunctionIcon,
  Plant01Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import {
  createRecipeDraft,
  createRecipeFromLearningRecipe,
  deleteRecipe,
  primeRecipeCache,
  useRecipes,
} from "../data/recipes-db";
import { LEARNING_RECIPES } from "../learning-recipes";

const OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY =
  "data-recipes:open-learning-recipes";

type TemplateCard = {
  titleKey: TranslationKey;
  descriptionKey: TranslationKey;
  icon: typeof CookBookIcon;
  difficulty: "easy" | "starter" | "intermediate" | "advanced";
  learningBadges: TranslationKey[];
  surfaceClassName: string;
  shineColor: string[];
  learningRecipeId?: string;
};

const TEMPLATE_CARDS: TemplateCard[] = [
  {
    titleKey: "dataRecipes.template.instructionFromAnswer.title",
    descriptionKey: "dataRecipes.template.instructionFromAnswer.description",
    icon: Plant01Icon,
    difficulty: "easy",
    learningBadges: [
      "dataRecipes.badge.seedDataset",
      "dataRecipes.badge.llmText",
      "dataRecipes.badge.prompting",
    ],
    surfaceClassName:
      "from-emerald-500/15 via-green-500/5 to-transparent dark:from-emerald-400/30 dark:via-green-400/14 dark:to-emerald-950/16",
    shineColor: [
      "rgb(16 185 129 / 0.45)",
      "rgb(34 197 94 / 0.4)",
      "rgb(52 211 153 / 0.45)",
    ],
    learningRecipeId: "instruction-from-answer",
  },
  {
    titleKey: "dataRecipes.template.pdfQa.title",
    descriptionKey: "dataRecipes.template.pdfQa.description",
    icon: DocumentAttachmentIcon,
    difficulty: "easy",
    learningBadges: ["dataRecipes.badge.unstructured", "dataRecipes.badge.llmText"],
    surfaceClassName:
      "from-violet-500/15 via-fuchsia-500/5 to-transparent dark:from-violet-400/30 dark:via-fuchsia-400/14 dark:to-violet-950/16",
    shineColor: [
      "rgb(139 92 246 / 0.45)",
      "rgb(217 70 239 / 0.4)",
      "rgb(168 85 247 / 0.45)",
    ],
    learningRecipeId: "pdf-grounded-qa",
  },
  {
    titleKey: "dataRecipes.template.ocr.title",
    descriptionKey: "dataRecipes.template.ocr.description",
    icon: Album02Icon,
    difficulty: "starter",
    learningBadges: [
      "dataRecipes.badge.vision",
      "dataRecipes.badge.llmText",
      "dataRecipes.badge.imageContext",
    ],
    surfaceClassName:
      "from-lime-500/15 via-emerald-500/5 to-transparent dark:from-lime-400/30 dark:via-emerald-400/14 dark:to-lime-950/16",
    shineColor: [
      "rgb(132 204 22 / 0.45)",
      "rgb(16 185 129 / 0.4)",
      "rgb(74 222 128 / 0.45)",
    ],
    learningRecipeId: "ocr-document-extraction",
  },
  {
    titleKey: "dataRecipes.template.textToPython.title",
    descriptionKey: "dataRecipes.template.textToPython.description",
    icon: CodeIcon,
    difficulty: "intermediate",
    learningBadges: [
      "dataRecipes.badge.llmJudge",
      "dataRecipes.badge.llmCode",
      "dataRecipes.badge.subcategory",
      "dataRecipes.badge.category",
    ],
    surfaceClassName:
      "from-amber-500/15 via-orange-500/5 to-transparent dark:from-amber-400/30 dark:via-orange-400/14 dark:to-amber-950/16",
    shineColor: [
      "rgb(245 158 11 / 0.45)",
      "rgb(249 115 22 / 0.4)",
      "rgb(251 146 60 / 0.45)",
    ],
    learningRecipeId: "text-to-python",
  },
  {
    titleKey: "dataRecipes.template.textToSql.title",
    descriptionKey: "dataRecipes.template.textToSql.description",
    icon: Database02Icon,
    difficulty: "intermediate",
    learningBadges: [
      "dataRecipes.badge.llmCode",
      "dataRecipes.badge.prompting",
      "dataRecipes.badge.dropColumns",
    ],
    surfaceClassName:
      "from-blue-500/15 via-indigo-500/5 to-transparent dark:from-blue-400/30 dark:via-indigo-400/14 dark:to-blue-950/16",
    shineColor: [
      "rgb(59 130 246 / 0.45)",
      "rgb(99 102 241 / 0.4)",
      "rgb(96 165 250 / 0.45)",
    ],
    learningRecipeId: "text-to-sql",
  },
  {
    titleKey: "dataRecipes.template.structuredJinja.title",
    descriptionKey: "dataRecipes.template.structuredJinja.description",
    icon: FunctionIcon,
    difficulty: "advanced",
    learningBadges: [
      "dataRecipes.badge.structuredLlm",
      "dataRecipes.badge.expression",
      "dataRecipes.badge.jinja",
    ],
    surfaceClassName:
      "from-cyan-500/15 via-sky-500/5 to-transparent dark:from-cyan-400/30 dark:via-sky-400/14 dark:to-cyan-950/16",
    shineColor: [
      "rgb(6 182 212 / 0.45)",
      "rgb(56 189 248 / 0.4)",
      "rgb(34 211 238 / 0.45)",
    ],
    learningRecipeId: "structured-outputs-jinja",
  },
];

const LEARNING_RECIPE_BY_ID = new Map(
  LEARNING_RECIPES.map((recipe) => [recipe.id, recipe]),
);

function formatRelativeTime(
  value: number,
  t: (key: TranslationKey) => string,
): string {
  const now = Date.now();
  const diffMs = Math.max(0, now - value);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;

  if (diffMs < minute) {
    return t("dataRecipes.time.justNow");
  }
  if (diffMs < hour) {
    const minutes = Math.floor(diffMs / minute);
    return t("dataRecipes.time.minutesAgo")
      .replace("{count}", String(minutes))
      .replace("{suffix}", minutes === 1 ? "" : "s");
  }
  if (diffMs < day) {
    const hours = Math.floor(diffMs / hour);
    return t("dataRecipes.time.hoursAgo")
      .replace("{count}", String(hours))
      .replace("{suffix}", hours === 1 ? "" : "s");
  }
  if (diffMs < week) {
    const days = Math.floor(diffMs / day);
    return t("dataRecipes.time.daysAgo")
      .replace("{count}", String(days))
      .replace("{suffix}", days === 1 ? "" : "s");
  }
  const weeks = Math.floor(diffMs / week);
  return t("dataRecipes.time.weeksAgo")
    .replace("{count}", String(weeks))
    .replace("{suffix}", weeks === 1 ? "" : "s");
}

function LearningRecipeCards({
  onSelect,
  loadingTemplateId,
}: {
  onSelect: (template: TemplateCard) => void;
  loadingTemplateId: string | null;
}): ReactElement {
  const { t } = useI18n();
  return (
    <div className="grid w-full gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {TEMPLATE_CARDS.map((template) => {
        const learningRecipe = template.learningRecipeId
          ? LEARNING_RECIPE_BY_ID.get(template.learningRecipeId)
          : undefined;
        const isReady = Boolean(learningRecipe);
        const isLoading =
          template.learningRecipeId !== undefined &&
          loadingTemplateId === template.learningRecipeId;
        const isDisabled = !isReady || isLoading || Boolean(loadingTemplateId);
        const visibleLearningBadges = template.learningBadges.slice(0, 4);
        const extraLearningBadgeCount = Math.max(
          0,
          template.learningBadges.length - 4,
        );
        return (
          <button
            key={template.titleKey}
            type="button"
            disabled={isDisabled}
            onClick={() => onSelect(template)}
            className={`group shadow-border relative overflow-hidden rounded-2xl bg-gradient-to-br text-left transition-transform ${template.surfaceClassName} enabled:cursor-pointer enabled:hover:-translate-y-0.5 enabled:hover:shadow-md disabled:cursor-not-allowed disabled:opacity-70`}
          >
            <ShineBorder
              borderWidth={1.2}
              duration={13}
              shineColor={template.shineColor}
            />
            <div className="relative flex h-full min-h-40 flex-col justify-between gap-3 p-4">
              <Badge
                className="absolute right-3 top-3"
                variant={
                  template.difficulty === "advanced" ? "secondary" : "outline"
                }
              >
                {template.difficulty === "easy"
                  ? t("dataRecipes.difficulty.easy")
                  : template.difficulty === "starter"
                    ? t("dataRecipes.difficulty.starter")
                    : template.difficulty === "intermediate"
                      ? t("dataRecipes.difficulty.intermediate")
                      : t("dataRecipes.difficulty.advanced")}
              </Badge>
              <div className="inline-flex size-10 items-center justify-center rounded-xl border border-foreground/10 bg-background/80">
                <HugeiconsIcon
                  icon={template.icon}
                  className="size-5 text-foreground/90"
                />
              </div>
              <div className="space-y-1">
                <p className="line-clamp-2 text-sm font-semibold leading-tight text-foreground">
                  {t(template.titleKey)}
                </p>
                <p className="line-clamp-2 text-xs text-muted-foreground">
                  {t(template.descriptionKey)}
                </p>
              </div>
              <div className="flex items-center gap-1 overflow-hidden whitespace-nowrap">
                {isLoading ? (
                  <Badge variant="outline">{t("dataRecipes.loadingShort")}</Badge>
                ) : (
                  <>
                    {visibleLearningBadges.map((badge) => (
                      <Badge
                        key={`${template.titleKey}-${badge}`}
                        variant="outline"
                        className="h-5 shrink-0 px-1.5 text-[10px]"
                      >
                        {t(badge)}
                      </Badge>
                    ))}
                    {extraLearningBadgeCount > 0 ? (
                      <Badge
                        variant="outline"
                        className="h-5 shrink-0 px-1.5 text-[10px]"
                      >
                        +{extraLearningBadgeCount}
                      </Badge>
                    ) : null}
                    {isReady ? null : (
                      <Badge
                        variant="secondary"
                        className="h-5 shrink-0 px-1.5 text-[10px]"
                      >
                        {t("dataRecipes.soon")}
                      </Badge>
                    )}
                  </>
                )}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

export function DataRecipesPage(): ReactElement {
  const { t } = useI18n();
  const navigate = useNavigate();
  const { recipes, ready } = useRecipes();
  const [creatingRecipe, setCreatingRecipe] = useState(false);
  const [learningDialogOpen, setLearningDialogOpen] = useState(false);
  const [loadingTemplateId, setLoadingTemplateId] = useState<string | null>(
    null,
  );

  useEffect(() => {
    if (sessionStorage.getItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY) !== "1") {
      return;
    }
    sessionStorage.removeItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY);
    setLearningDialogOpen(true);
  }, []);

  async function openNewRecipe(): Promise<void> {
    if (creatingRecipe || loadingTemplateId) {
      return;
    }
    setCreatingRecipe(true);
    try {
      const recipe = await createRecipeDraft();
      primeRecipeCache(recipe);
      await navigate({
        to: "/data-recipes/$recipeId",
        params: { recipeId: recipe.id },
      });
    } finally {
      setCreatingRecipe(false);
    }
  }

  async function openLearningRecipe(template: TemplateCard): Promise<void> {
    if (creatingRecipe || loadingTemplateId) {
      return;
    }
    if (!template.learningRecipeId) {
      toastError(t("dataRecipes.toast.learningNotReady"));
      return;
    }
    const recipeTemplate = LEARNING_RECIPE_BY_ID.get(template.learningRecipeId);
    if (!recipeTemplate) {
      toastError(t("dataRecipes.toast.learningNotFound"));
      return;
    }

    setLoadingTemplateId(template.learningRecipeId);
    try {
      const payload = await recipeTemplate.loadPayload();
      const recipe = await createRecipeFromLearningRecipe({
        templateId: recipeTemplate.id,
        templateTitle: recipeTemplate.title,
        payload,
      });
      primeRecipeCache(recipe);
      setLearningDialogOpen(false);
      await navigate({
        to: "/data-recipes/$recipeId",
        params: { recipeId: recipe.id },
      });
    } catch (error) {
      toastError(
        t("dataRecipes.toast.startFailed"),
        error instanceof Error ? error.message : undefined,
      );
    } finally {
      setLoadingTemplateId(null);
    }
  }

  function openRecipe(recipe: (typeof recipes)[number]): void {
    primeRecipeCache(recipe);
    navigate({
      to: "/data-recipes/$recipeId",
      params: { recipeId: recipe.id },
    }).catch(() => undefined);
  }

  async function handleDeleteRecipe(recipeId: string): Promise<void> {
    await deleteRecipe(recipeId);
  }

  const isBusy = creatingRecipe || Boolean(loadingTemplateId);

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto w-full max-w-7xl px-6 py-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              {t("dataRecipes.title")}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("dataRecipes.subtitle")}
            </p>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button type="button" disabled={isBusy}>
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                {t("dataRecipes.newRecipe")}
                <HugeiconsIcon icon={ArrowDown01Icon} className="size-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onSelect={() => {
                  openNewRecipe().catch(() => undefined);
                }}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                {t("dataRecipes.startEmpty")}
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={() => {
                  setLearningDialogOpen(true);
                }}
              >
                <HugeiconsIcon icon={CookBookIcon} className="size-4" />
                {t("dataRecipes.startFromLearning")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {!ready ? (
          <div className="mt-8 rounded-2xl border border-border/70 bg-card px-6 py-10 text-center">
            <p className="text-sm font-medium text-foreground">
              {t("dataRecipes.loading.title")}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {t("dataRecipes.loading.description")}
            </p>
          </div>
        ) : recipes.length === 0 ? (
          <Empty className="mt-8 border border-dashed border-border/70">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={CookBookIcon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>{t("dataRecipes.empty.title")}</EmptyTitle>
              <EmptyDescription>
                {t("dataRecipes.empty.description")}
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent className="max-w-6xl items-stretch">
              {/*<Button*/}
              {/*  type="button"*/}
              {/*  variant="secondary"*/}
              {/*  className="mx-auto"*/}
              {/*  onClick={() => setLearningDialogOpen(true)}*/}
              {/*  disabled={isBusy}*/}
              {/*>*/}
              {/*  <HugeiconsIcon icon={CookBookIcon} className="size-4" />*/}
              {/*  Start Tutorial*/}
              {/*</Button>*/}
              <LearningRecipeCards
                onSelect={(template) => {
                  openLearningRecipe(template).catch(() => undefined);
                }}
                loadingTemplateId={loadingTemplateId}
              />
            </EmptyContent>
          </Empty>
        ) : (
          <div className="mt-8 space-y-2">
            {recipes.map((recipe) => (
              <div
                key={recipe.id}
                className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3"
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-center gap-3 text-left"
                  onClick={() => openRecipe(recipe)}
                >
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-muted/20">
                    <HugeiconsIcon
                      icon={CookBookIcon}
                      className="size-4 text-muted-foreground"
                    />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="truncate text-sm font-medium">
                        {recipe.name}
                      </p>
                      {recipe.learningRecipeId ? (
                        <Badge variant="outline">
                          {t("dataRecipes.learningRecipeBadge")}
                        </Badge>
                      ) : null}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {t("dataRecipes.lastUpdated").replace(
                        "{time}",
                        formatRelativeTime(recipe.updatedAt, t),
                      )}{" "}
                      |{" "}
                      {t("dataRecipes.created").replace(
                        "{time}",
                        formatRelativeTime(recipe.createdAt, t),
                      )}
                    </p>
                  </div>
                </button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  onClick={() => {
                    handleDeleteRecipe(recipe.id).catch(() => undefined);
                  }}
                  aria-label={t("dataRecipes.deleteAria").replace(
                    "{name}",
                    recipe.name,
                  )}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </main>

      <Dialog open={learningDialogOpen} onOpenChange={setLearningDialogOpen}>
        <DialogContent
          className="sm:max-w-5xl"
          overlayClassName="bg-background/45 supports-backdrop-filter:backdrop-blur-[1px]"
        >
          <DialogHeader>
            <DialogTitle>{t("dataRecipes.learningDialog.title")}</DialogTitle>
            <DialogDescription>
              {t("dataRecipes.learningDialog.description")}
            </DialogDescription>
          </DialogHeader>
          <LearningRecipeCards
            onSelect={(template) => {
              openLearningRecipe(template).catch(() => undefined);
            }}
            loadingTemplateId={loadingTemplateId}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}
