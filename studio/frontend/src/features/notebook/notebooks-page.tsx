// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { InputGroupAddon } from "@/components/ui/input-group";
import { ShineBorder } from "@/components/ui/shine-border";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import { toastError } from "@/shared/toast";
import {
  AiBrain01Icon,
  BookOpen01Icon,
  CodeIcon,
  GithubIcon,
  Globe02Icon,
  Mic01Icon,
  Notebook01Icon,
  Search01Icon,
  SparklesIcon,
  TestTube01Icon,
  ViewIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchNotebookCatalog,
  notebookMatchesQuery,
  type NotebookCatalogEntry,
  useOpenNotebookInStudio,
} from "./notebooks-lib";

type CategoryStyle = {
  icon: typeof Notebook01Icon;
  surfaceClassName: string;
  shineColor: string[];
};

const CATEGORY_STYLES: Record<string, CategoryStyle> = {
  sft: {
    icon: BookOpen01Icon,
    surfaceClassName:
      "from-blue-500/15 via-indigo-500/5 to-transparent dark:from-blue-400/20 dark:via-indigo-400/10 dark:to-blue-950/16",
    shineColor: [
      "rgb(59 130 246 / 0.45)",
      "rgb(99 102 241 / 0.4)",
      "rgb(96 165 250 / 0.45)",
    ],
  },
  vision: {
    icon: ViewIcon,
    surfaceClassName:
      "from-violet-500/15 via-fuchsia-500/5 to-transparent dark:from-violet-400/20 dark:via-fuchsia-400/10 dark:to-violet-950/16",
    shineColor: [
      "rgb(139 92 246 / 0.45)",
      "rgb(217 70 239 / 0.4)",
      "rgb(168 85 247 / 0.45)",
    ],
  },
  grpo: {
    icon: SparklesIcon,
    surfaceClassName:
      "from-amber-500/15 via-orange-500/5 to-transparent dark:from-amber-400/20 dark:via-orange-400/10 dark:to-amber-950/16",
    shineColor: [
      "rgb(245 158 11 / 0.45)",
      "rgb(249 115 22 / 0.4)",
      "rgb(251 146 60 / 0.45)",
    ],
  },
  audio: {
    icon: Mic01Icon,
    surfaceClassName:
      "from-rose-500/15 via-pink-500/5 to-transparent dark:from-rose-400/20 dark:via-pink-400/10 dark:to-rose-950/16",
    shineColor: [
      "rgb(244 63 94 / 0.45)",
      "rgb(236 72 153 / 0.4)",
      "rgb(251 113 133 / 0.45)",
    ],
  },
  embedding: {
    icon: AiBrain01Icon,
    surfaceClassName:
      "from-cyan-500/15 via-sky-500/5 to-transparent dark:from-cyan-400/20 dark:via-sky-400/10 dark:to-cyan-950/16",
    shineColor: [
      "rgb(6 182 212 / 0.45)",
      "rgb(56 189 248 / 0.4)",
      "rgb(34 211 238 / 0.45)",
    ],
  },
  reasoning: {
    icon: AiBrain01Icon,
    surfaceClassName:
      "from-emerald-500/15 via-green-500/5 to-transparent dark:from-emerald-400/20 dark:via-green-400/10 dark:to-emerald-950/16",
    shineColor: [
      "rgb(16 185 129 / 0.45)",
      "rgb(34 197 94 / 0.4)",
      "rgb(52 211 153 / 0.45)",
    ],
  },
  code: {
    icon: CodeIcon,
    surfaceClassName:
      "from-slate-500/15 via-zinc-500/5 to-transparent dark:from-slate-400/20 dark:via-zinc-400/10 dark:to-slate-950/16",
    shineColor: [
      "rgb(71 85 105 / 0.45)",
      "rgb(100 116 139 / 0.4)",
      "rgb(148 163 184 / 0.45)",
    ],
  },
  inference: {
    icon: ZapIcon,
    surfaceClassName:
      "from-lime-500/15 via-emerald-500/5 to-transparent dark:from-lime-400/20 dark:via-emerald-400/10 dark:to-lime-950/16",
    shineColor: [
      "rgb(132 204 22 / 0.45)",
      "rgb(16 185 129 / 0.4)",
      "rgb(74 222 128 / 0.45)",
    ],
  },
};

const DEFAULT_CATEGORY_STYLE: CategoryStyle = {
  icon: Notebook01Icon,
  surfaceClassName:
    "from-slate-500/15 via-zinc-500/5 to-transparent dark:from-slate-400/20 dark:via-zinc-400/10 dark:to-slate-950/16",
  shineColor: [
    "rgb(71 85 105 / 0.45)",
    "rgb(100 116 139 / 0.4)",
    "rgb(148 163 184 / 0.45)",
  ],
};

function categoryStyle(category: string): CategoryStyle {
  return CATEGORY_STYLES[category] ?? DEFAULT_CATEGORY_STYLE;
}

function NotebookCard({
  notebook,
  onUseInStudio,
  studioDisabled,
  studioDisabledHint,
  highlighted = false,
  cardRef,
}: {
  notebook: NotebookCatalogEntry;
  onUseInStudio: (notebook: NotebookCatalogEntry) => void;
  studioDisabled: boolean;
  studioDisabledHint?: string;
  highlighted?: boolean;
  cardRef?: (node: HTMLElement | null) => void;
}): ReactElement {
  const t = useT();
  const style = categoryStyle(notebook.category);

  return (
    <article
      ref={cardRef}
      className={`group shadow-border relative overflow-hidden rounded-2xl bg-gradient-to-br dark:bg-white/[0.05] ${style.surfaceClassName} ${highlighted ? "ring-2 ring-primary/70" : ""}`}
    >
      <ShineBorder borderWidth={1.2} duration={13} shineColor={style.shineColor} />
      <div className="relative flex h-full min-h-44 flex-col gap-4 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="inline-flex size-10 items-center justify-center rounded-xl border border-foreground/10 bg-background/80">
            <HugeiconsIcon icon={style.icon} className="size-5 text-foreground/90" />
          </div>
          <div className="flex flex-wrap justify-end gap-1.5">
            {notebook.featured ? (
              <Badge variant="secondary">{t("notebooks.featured")}</Badge>
            ) : null}
            <Badge variant="outline">
              {t(`notebooks.categories.${notebook.category}` as "notebooks.categories.sft")}
            </Badge>
          </div>
        </div>

        <div className="space-y-1">
          <h2 className="line-clamp-2 text-sm font-semibold leading-tight text-foreground">
            {notebook.title}
          </h2>
          <p className="line-clamp-2 text-xs text-muted-foreground">
            {notebook.studio_model
              ? t("notebooks.studioModelReady", { model: notebook.studio_model })
              : t("notebooks.studioManualConfigure")}
          </p>
        </div>

        <div className="mt-auto flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            disabled={studioDisabled}
            title={studioDisabled ? studioDisabledHint : undefined}
            onClick={() => onUseInStudio(notebook)}
          >
            <HugeiconsIcon icon={TestTube01Icon} className="size-4" />
            {t("notebooks.useInStudio")}
          </Button>
          <Button type="button" size="sm" variant="outline" asChild>
            <a href={notebook.colab_url} target="_blank" rel="noreferrer">
              <HugeiconsIcon icon={Globe02Icon} className="size-4" />
              {t("notebooks.openInColab")}
            </a>
          </Button>
          <Button type="button" size="sm" variant="ghost" asChild>
            <a href={notebook.github_url} target="_blank" rel="noreferrer">
              <HugeiconsIcon icon={GithubIcon} className="size-4" />
              {t("notebooks.viewOnGitHub")}
            </a>
          </Button>
        </div>
      </div>
    </article>
  );
}

export function NotebooksPage(): ReactElement {
  const t = useT();
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const { openInStudio } = useOpenNotebookInStudio();

  const [loading, setLoading] = useState(true);
  const [notebooks, setNotebooks] = useState<NotebookCatalogEntry[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [pickedNotebookId, setPickedNotebookId] = useState<string | null>(null);
  const cardRefs = useRef(new Map<string, HTMLElement | null>());

  const studioDisabledHint =
    chatOnlyReason === "mlx_unavailable"
      ? t("notebooks.studioDisabled.mlxUnavailable")
      : chatOnlyReason === "intel_mac"
        ? t("notebooks.studioDisabled.intelMac")
        : chatOnlyReason === "no_gpu"
          ? t("notebooks.studioDisabled.noGpu")
          : t("notebooks.studioDisabled.generic");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchNotebookCatalog()
      .then((catalog) => {
        if (cancelled) return;
        setNotebooks(catalog.notebooks);
        setCategories(catalog.categories);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        toastError(
          t("notebooks.loadFailed"),
          error instanceof Error ? error.message : undefined,
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const filteredNotebooks = useMemo(() => {
    return notebooks.filter((notebook) => {
      if (activeCategory !== "all" && notebook.category !== activeCategory) {
        return false;
      }
      return notebookMatchesQuery(notebook, query);
    });
  }, [activeCategory, notebooks, query]);

  const notebookById = useMemo(
    () => new Map(notebooks.map((notebook) => [notebook.id, notebook])),
    [notebooks],
  );

  const pickerItems = useMemo(() => {
    const matches = notebooks.filter((notebook) => notebookMatchesQuery(notebook, query));
    return matches.map((notebook) => notebook.id);
  }, [notebooks, query]);

  const pickedNotebook = pickedNotebookId
    ? notebookById.get(pickedNotebookId) ?? null
    : null;

  const scrollToNotebook = (notebookId: string): void => {
    const node = cardRefs.current.get(notebookId);
    node?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const handlePickNotebook = (notebookId: string | null): void => {
    if (!notebookId) {
      setPickedNotebookId(null);
      return;
    }
    setPickedNotebookId(notebookId);
    setActiveCategory("all");
    window.requestAnimationFrame(() => scrollToNotebook(notebookId));
  };

  const featuredNotebooks = useMemo(
    () => filteredNotebooks.filter((notebook) => notebook.featured),
    [filteredNotebooks],
  );
  const otherNotebooks = useMemo(
    () => filteredNotebooks.filter((notebook) => !notebook.featured),
    [filteredNotebooks],
  );

  return (
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto w-full max-w-7xl px-5 py-8 sm:px-9">
        <div className="max-w-2xl">
          <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
            {t("notebooks.title")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("notebooks.subtitle")}
          </p>
        </div>

        <div className="mt-6 flex flex-col gap-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="w-full max-w-xl">
              <Combobox
                items={pickerItems}
                filteredItems={pickerItems}
                filter={null}
                value={pickedNotebookId}
                onValueChange={handlePickNotebook}
                onInputValueChange={(value) => {
                  setQuery(value);
                  if (pickedNotebookId && value !== pickedNotebook?.title) {
                    setPickedNotebookId(null);
                  }
                }}
                itemToStringValue={(id) => notebookById.get(id)?.title ?? id}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder={t("notebooks.pickerPlaceholder")}
                  className="w-full min-w-0"
                  showClear={true}
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent>
                  <ComboboxEmpty>{t("notebooks.pickerEmpty")}</ComboboxEmpty>
                  <ComboboxList className="max-h-72 p-1">
                    {(id: string) => {
                      const notebook = notebookById.get(id);
                      if (!notebook) return null;
                      return (
                        <ComboboxItem key={id} value={id} className="flex-col items-start gap-1">
                          <span className="font-medium">{notebook.title}</span>
                          <span className="text-xs text-muted-foreground">
                            {notebook.studio_model ?? notebook.notebook_file}
                          </span>
                        </ComboboxItem>
                      );
                    }}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
            {!loading ? (
              <p className="text-sm text-muted-foreground">
                {t("notebooks.resultsCount", { count: filteredNotebooks.length })}
              </p>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant={activeCategory === "all" ? "default" : "outline"}
              onClick={() => setActiveCategory("all")}
            >
              {t("notebooks.allCategories")}
            </Button>
            {categories.map((category) => (
              <Button
                key={category}
                type="button"
                size="sm"
                variant={activeCategory === category ? "default" : "outline"}
                onClick={() => setActiveCategory(category)}
              >
                {t(`notebooks.categories.${category}` as "notebooks.categories.sft")}
              </Button>
            ))}
          </div>
        </div>

        {pickedNotebook ? (
          <section className="mt-6 space-y-3">
            <h2 className="text-sm font-semibold text-foreground">
              {t("notebooks.pickedSection")}
            </h2>
            <div className="max-w-xl">
              <NotebookCard
                notebook={pickedNotebook}
                onUseInStudio={openInStudio}
                studioDisabled={chatOnly}
                studioDisabledHint={studioDisabledHint}
                highlighted={true}
              />
            </div>
          </section>
        ) : null}

        {loading ? (
          <div className="mt-10 flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-4" />
            {t("notebooks.loading")}
          </div>
        ) : filteredNotebooks.length === 0 ? (
          <Empty className="mt-10 border border-dashed border-border/70 dark:border-none">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={Notebook01Icon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>{t("notebooks.emptyTitle")}</EmptyTitle>
              <EmptyDescription>{t("notebooks.emptyDescription")}</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="mt-8 space-y-8">
            {featuredNotebooks.length > 0 ? (
              <section className="space-y-4">
                <h2 className="text-sm font-semibold text-foreground">
                  {t("notebooks.featuredSection")}
                </h2>
                <div className="grid w-full gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  {featuredNotebooks.map((notebook) => (
                    <NotebookCard
                      key={notebook.id}
                      notebook={notebook}
                      onUseInStudio={openInStudio}
                      studioDisabled={chatOnly}
                      studioDisabledHint={studioDisabledHint}
                      highlighted={pickedNotebookId === notebook.id}
                      cardRef={(node) => {
                        cardRefs.current.set(notebook.id, node);
                      }}
                    />
                  ))}
                </div>
              </section>
            ) : null}

            {otherNotebooks.length > 0 ? (
              <section className="space-y-4">
                {featuredNotebooks.length > 0 ? (
                  <h2 className="text-sm font-semibold text-foreground">
                    {t("notebooks.allSection")}
                  </h2>
                ) : null}
                <div className="grid w-full gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  {otherNotebooks.map((notebook) => (
                    <NotebookCard
                      key={notebook.id}
                      notebook={notebook}
                      onUseInStudio={openInStudio}
                      studioDisabled={chatOnly}
                      studioDisabledHint={studioDisabledHint}
                      highlighted={pickedNotebookId === notebook.id}
                      cardRef={(node) => {
                        cardRefs.current.set(notebook.id, node);
                      }}
                    />
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        )}
      </main>
    </div>
  );
}
