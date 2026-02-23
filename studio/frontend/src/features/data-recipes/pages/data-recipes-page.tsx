import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { ShineBorder } from "@/components/ui/shine-border";
import {
  AiChat02Icon,
  CodeIcon,
  CookBookIcon,
  Database02Icon,
  Delete02Icon,
  FunctionIcon,
  Plant01Icon,
  PlusSignIcon,
  Shield02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { useState } from "react";
import {
  createRecipeDraft,
  deleteRecipe,
  useRecipes,
} from "../data/recipes-db";

type TemplateCard = {
  title: string;
  description: string;
  icon: typeof CookBookIcon;
  surfaceClassName: string;
  shineColor: string[];
};

const TEMPLATE_CARDS: TemplateCard[] = [
  {
    title: "Structured Outputs + Jinja Expressions",
    description:
      "Support ticket triage dataset with structured JSON outputs and Jinja if/else refs.",
    icon: FunctionIcon,
    surfaceClassName:
      "from-cyan-500/15 via-sky-500/5 to-transparent border-cyan-500/30",
    shineColor: ["#06b6d4", "#38bdf8", "#22d3ee"],
  },
  {
    title: "Basic MCP Tool Use",
    description:
      "Agent workflow starter showing tool-call patterns and grounded tool result usage.",
    icon: Shield02Icon,
    surfaceClassName:
      "from-violet-500/15 via-fuchsia-500/5 to-transparent border-violet-500/30",
    shineColor: ["#8b5cf6", "#d946ef", "#a855f7"],
  },
  {
    title: "Seed Dataset",
    description:
      "Start from real rows, then expand with synthetic fields while preserving source context.",
    icon: Plant01Icon,
    surfaceClassName:
      "from-emerald-500/15 via-green-500/5 to-transparent border-emerald-500/30",
    shineColor: ["#10b981", "#22c55e", "#34d399"],
  },
  {
    title: "Text to Python",
    description:
      "Instruction-to-code pairs for training models that generate clean Python implementations.",
    icon: CodeIcon,
    surfaceClassName:
      "from-amber-500/15 via-orange-500/5 to-transparent border-amber-500/30",
    shineColor: ["#f59e0b", "#f97316", "#fb923c"],
  },
  {
    title: "Text to SQL",
    description:
      "Natural language to SQL pairs, including schema-aware query construction patterns.",
    icon: Database02Icon,
    surfaceClassName:
      "from-blue-500/15 via-indigo-500/5 to-transparent border-blue-500/30",
    shineColor: ["#3b82f6", "#6366f1", "#60a5fa"],
  },
  {
    title: "Multi-Turn Chat",
    description:
      "Role-based multi-turn conversations for assistant behavior, memory, and response quality.",
    icon: AiChat02Icon,
    surfaceClassName:
      "from-rose-500/15 via-pink-500/5 to-transparent border-rose-500/30",
    shineColor: ["#f43f5e", "#ec4899", "#fb7185"],
  },
];

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

export function DataRecipesPage(): ReactElement {
  const navigate = useNavigate();
  const recipes = useRecipes();
  const [creatingRecipe, setCreatingRecipe] = useState(false);

  async function openNewRecipe(): Promise<void> {
    if (creatingRecipe) {
      return;
    }
    setCreatingRecipe(true);
    try {
      const recipe = await createRecipeDraft();
      await navigate({
        to: "/data-recipes/$recipeId",
        params: { recipeId: recipe.id },
      });
    } finally {
      setCreatingRecipe(false);
    }
  }

  function openRecipe(recipeId: string): void {
    navigate({
      to: "/data-recipes/$recipeId",
      params: { recipeId },
    }).catch(() => undefined);
  }

  async function handleDeleteRecipe(recipeId: string): Promise<void> {
    await deleteRecipe(recipeId);
  }

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
          <Button
            type="button"
            onClick={() => {
              openNewRecipe().catch(() => undefined);
            }}
            disabled={creatingRecipe}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
            New Recipe
          </Button>
        </div>

        {recipes.length === 0 ? (
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
              <Button
                type="button"
                variant="secondary"
                className="mx-auto"
                disabled={true}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                Start Tutorial
              </Button>
              <div className="grid w-full gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {TEMPLATE_CARDS.map((template) => (
                  <div
                    key={template.title}
                    className={`group relative overflow-hidden rounded-2xl border bg-gradient-to-br ${template.surfaceClassName}`}
                  >
                    <ShineBorder
                      borderWidth={1.2}
                      duration={11}
                      shineColor={template.shineColor}
                    />
                    <div className="relative flex h-full min-h-40 flex-col justify-between gap-3 p-4 text-left">
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
                      <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
                        Template
                      </span>
                    </div>
                  </div>
                ))}
              </div>
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
                  onClick={() => openRecipe(recipe.id)}
                >
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-muted/20">
                    <HugeiconsIcon
                      icon={CookBookIcon}
                      className="size-4 text-muted-foreground"
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {recipe.name}
                    </p>
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
    </div>
  );
}
