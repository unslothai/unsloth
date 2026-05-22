// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DocumentAttachmentIcon,
  DocumentCodeIcon,
  GithubIcon,
  Plant01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import type { SeedConfig } from "../../types";
import { HfDatasetCombobox } from "../shared/hf-dataset-combobox";
import { InlineField } from "./inline-field";

type InlineSeedProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
};

const GITHUB_REPO_LIST_SPLIT_RE = /[\n,]+/;

function parseGithubRepos(value: string | undefined): string[] {
  return (value ?? "")
    .split(GITHUB_REPO_LIST_SPLIT_RE)
    .map((repo) => repo.trim())
    .filter(Boolean);
}

function hasInvalidGithubRepo(repo: string): boolean {
  const parts = repo.split("/");
  return parts.length !== 2 || parts.some((part) => !part.trim());
}

export function InlineSeed({
  config,
  onUpdate,
}: InlineSeedProps): ReactElement {
  const mode = config.seed_source_type ?? "hf";

  if (mode === "github_repo") {
    const repos = parseGithubRepos(config.github_repo_slug);
    const invalidCount = repos.filter(hasInvalidGithubRepo).length;
    const summary =
      repos.length === 0
        ? "Add GitHub repos"
        : repos.length === 1
          ? repos[0]
          : `${repos.length} repositories`;
    const configuredItems = config.github_item_types;
    const items =
      configuredItems && configuredItems.length > 0
        ? configuredItems
        : ["issues", "pulls"];
    const itemsLabel = items.join(" · ");
    const limit = (config.github_limit ?? "100").trim() || "100";
    const commentsLabel =
      config.github_include_comments === false
        ? "comments off"
        : `comments ≤ ${config.github_max_comments_per_item ?? "30"}`;
    const tokenLabel = config.github_token?.trim() ? "PAT set" : "server token";
    const warning =
      repos.length === 0
        ? "No repos configured"
        : invalidCount > 0
          ? `${invalidCount} invalid repo${invalidCount === 1 ? "" : "s"}`
          : null;
    return (
      <div
        className={`corner-squircle flex items-center gap-2 rounded-md border px-2 py-2 ${warning ? "border-amber-500/50 bg-amber-500/10" : "border-border/60 bg-muted/30"}`}
      >
        <div className="corner-squircle rounded-md bg-primary/10 p-1.5 text-primary">
          <HugeiconsIcon icon={GithubIcon} className="size-3.5" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-xs font-medium">{summary}</p>
          <p className="truncate text-[11px] text-muted-foreground">
            {warning ??
              `${itemsLabel} · limit ${limit} · ${commentsLabel} · ${tokenLabel}`}
          </p>
        </div>
        <HugeiconsIcon
          icon={Plant01Icon}
          className="ml-auto size-3.5 text-muted-foreground/60"
        />
      </div>
    );
  }

  if (mode === "hf") {
    return (
      <div className="space-y-2">
        <InlineField label="Dataset">
          <HfDatasetCombobox
            value={config.hf_repo_id}
            accessToken={config.hf_token?.trim() || undefined}
            onValueChange={(next) =>
              onUpdate({
                hf_repo_id: next,
                hf_path: "",
                seed_columns: [],
                seed_drop_columns: [],
                seed_preview_rows: [],
              })
            }
            placeholder="org/repo"
          />
        </InlineField>
        <p className="text-[11px] text-muted-foreground">
          Load columns in dialog.
        </p>
      </div>
    );
  }

  const isLocal = mode === "local";
  const fileName = isLocal
    ? config.local_file_name?.trim()
    : config.unstructured_file_names?.length
      ? `${config.unstructured_file_names.length} file${config.unstructured_file_names.length !== 1 ? "s" : ""}`
      : undefined;

  return (
    <div className="corner-squircle flex items-center gap-2 rounded-md border border-border/60 bg-muted/30 px-2 py-2">
      <div className="corner-squircle rounded-md bg-primary/10 p-1.5 text-primary">
        <HugeiconsIcon
          icon={isLocal ? DocumentCodeIcon : DocumentAttachmentIcon}
          className="size-3.5"
        />
      </div>
      <div className="min-w-0">
        <p className="truncate text-xs font-medium">
          {fileName || "No file selected"}
        </p>
        <p className="text-[11px] text-muted-foreground">
          {isLocal ? "Structured file" : "Unstructured document"} · configure in
          dialog
        </p>
      </div>
      <HugeiconsIcon
        icon={Plant01Icon}
        className="ml-auto size-3.5 text-muted-foreground/60"
      />
    </div>
  );
}
