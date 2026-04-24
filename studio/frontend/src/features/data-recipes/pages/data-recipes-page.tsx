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
  GithubIcon,
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
  title: string;
  description: string;
  icon: typeof CookBookIcon;
  difficulty: "Easy" | "Starter" | "Intermediate" | "Advanced";
  learningBadges: string[];
  surfaceClassName: string;
  shineColor: string[];
  learningRecipeId?: string;
};

const TEMPLATE_CARDS: TemplateCard[] = [
  {
    title: "Instruction from Answer",
    description:
      "Start from seed answer fields and generate matching user instructions for SFT pairs.",
    icon: Plant01Icon,
    difficulty: "Easy",
    learningBadges: ["Seed Dataset", "LLM Text", "Prompting"],
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
    title: "PDF Document QA",
    description:
      "Unstructured PDF chunks transformed into grounded question-answer training pairs.",
    icon: DocumentAttachmentIcon,
    difficulty: "Easy",
    learningBadges: ["Unstructured", "LLM Text"],
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
    title: "OCR Document Extraction",
    description:
      "Use image context from seed data to generate OCR-style extraction outputs.",
    icon: Album02Icon,
    difficulty: "Starter",
    learningBadges: ["Vision", "LLM Text", "Image Context"],
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
    title: "Text to Python",
    description:
      "Instruction-to-code pairs for training models that generate clean Python implementations.",
    icon: CodeIcon,
    difficulty: "Intermediate",
    learningBadges: ["LLM Judge", "LLM Code", "Subcategory", "Category"],
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
    title: "Text to SQL",
    description:
      "Natural language to SQL pairs, including schema-aware query construction patterns.",
    icon: Database02Icon,
    difficulty: "Intermediate",
    learningBadges: ["LLM Code", "Prompting", "Drop Columns"],
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
    title: "Structured Outputs + Jinja Expressions",
    description:
      "Support ticket triage dataset with structured JSON outputs and Jinja if/else refs.",
    icon: FunctionIcon,
    difficulty: "Advanced",
    learningBadges: ["Structured LLM", "Expression", "Jinja"],
    surfaceClassName:
      "from-cyan-500/15 via-sky-500/5 to-transparent dark:from-cyan-400/30 dark:via-sky-400/14 dark:to-cyan-950/16",
    shineColor: [
      "rgb(6 182 212 / 0.45)",
      "rgb(56 189 248 / 0.4)",
      "rgb(34 211 238 / 0.45)",
    ],
    learningRecipeId: "structured-outputs-jinja",
  },
  {
    title: "GitHub Crawler",
    description:
      "Crawl real GitHub issues and PRs and invert each thread into a {User, Assistant} training pair.",
    icon: GithubIcon,
    difficulty: "Intermediate",
    learningBadges: ["GitHub", "LLM Text", "Structured LLM"],
    surfaceClassName:
      "from-slate-500/15 via-zinc-500/5 to-transparent dark:from-slate-400/30 dark:via-zinc-400/14 dark:to-slate-950/16",
    shineColor: [
      "rgb(71 85 105 / 0.45)",
      "rgb(100 116 139 / 0.4)",
      "rgb(148 163 184 / 0.45)",
    ],
    learningRecipeId: "github-support-bot",
  },
];

const LEARNING_RECIPE_BY_ID = new Map(
  LEARNING_RECIPES.map((recipe) => [recipe.id, recipe]),
);

function formatRelativeTime(value: number): string {
  const now = Date.now();
  const diffMs = Math.max(0, now - value);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;

  if (diffMs < minute) {
    return "just now";
  }
  if (diffMs < hour) {
    const minutes = Math.floor(diffMs / minute);
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  }
  if (diffMs < day) {
    const hours = Math.floor(diffMs / hour);
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  }
  if (diffMs < week) {
    const days = Math.floor(diffMs / day);
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }
  const weeks = Math.floor(diffMs / week);
  return `${weeks} week${weeks === 1 ? "" : "s"} ago`;
}

function LearningRecipeCards({
  onSelect,
  loadingTemplateId,
}: {
  onSelect: (template: TemplateCard) => void;
  loadingTemplateId: string | null;
}): ReactElement {
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
            key={template.title}
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
                  template.difficulty === "Advanced" ? "secondary" : "outline"
                }
              >
                {template.difficulty}
              </Badge>
              <div className="inline-flex size-10 items-center justify-center rounded-xl border border-foreground/10 bg-background/80">
                <HugeiconsIcon
                  icon={template.icon}
                  className="size-5 text-foreground/90"
                />
              </div>
              <div className="space-y-1">
                <p className="line-clamp-2 text-sm font-semibold leading-tight text-foreground">
                  {template.title}
                </p>
                <p className="line-clamp-2 text-xs text-muted-foreground">
                  {template.description}
                </p>
              </div>
              <div className="flex items-center gap-1 overflow-hidden whitespace-nowrap">
                {isLoading ? (
                  <Badge variant="outline">Loading...</Badge>
                ) : (
                  <>
                    {visibleLearningBadges.map((badge) => (
                      <Badge
                        key={`${template.title}-${badge}`}
                        variant="outline"
                        className="h-5 shrink-0 px-1.5 text-[10px]"
                      >
                        {badge}
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
                        Soon
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
      toastError("Learning recipe not ready yet.");
      return;
    }
    const recipeTemplate = LEARNING_RECIPE_BY_ID.get(template.learningRecipeId);
    if (!recipeTemplate) {
      toastError("Learning recipe not found.");
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
        "Failed to start learning recipe.",
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
              Data Recipes
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Create and manage local recipe workflows.
            </p>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button type="button" disabled={isBusy}>
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                New Recipe
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
                Start Empty
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={() => {
                  setLearningDialogOpen(true);
                }}
              >
                <HugeiconsIcon icon={CookBookIcon} className="size-4" />
                Start from Learning Recipe
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {!ready ? (
          <div className="mt-8 rounded-2xl border border-border/70 bg-card px-6 py-10 text-center">
            <p className="text-sm font-medium text-foreground">
              Loading recipes
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Fetching your saved recipes and learning templates.
            </p>
          </div>
        ) : recipes.length === 0 ? (
          <Empty className="mt-8 border border-dashed border-border/70">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={CookBookIcon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>No recipes yet</EmptyTitle>
              <EmptyDescription>
                Browse Learning Recipes below to understand how recipe workflows
                work.
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
                        <Badge variant="outline">Learning Recipe</Badge>
                      ) : null}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Last updated {formatRelativeTime(recipe.updatedAt)} |
                      Created {formatRelativeTime(recipe.createdAt)}
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
                  aria-label={`Delete ${recipe.name}`}
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
            <DialogTitle>Learning Recipes</DialogTitle>
            <DialogDescription>
              Start from a prebuilt recipe to learn patterns, then edit and run.
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
