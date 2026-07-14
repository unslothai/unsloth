// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Alert02Icon,
  AlertDiamondIcon,
  CookBookIcon,
  FloppyDiskIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type KeyboardEvent, type ReactElement, useState } from "react";
import type { RecipeStudioView } from "../execution-types";
import type { GraphWarning } from "../utils/graph-warnings";
import {
  RECIPE_STUDIO_WARNING_BADGE_TONE,
  RECIPE_STUDIO_WARNING_ICON_TONE,
} from "../utils/ui-tones";

type StatusTone = "success" | "error";

type RecipeStudioHeaderProps = {
  activeView: RecipeStudioView;
  saveLoading: boolean;
  saveTone: StatusTone;
  savedAtLabel: string;
  workflowName: string;
  warnings?: GraphWarning[];
  supportsEasyMode?: boolean;
  onWorkflowNameChange: (value: string) => void;
  onViewChange: (view: RecipeStudioView) => void;
  onSaveRecipe: () => void;
};

const STATUS_MESSAGE_CLASS: Record<StatusTone, string> = {
  success: "Saved",
  error: "Needs saving",
};

export function RecipeStudioHeader({
  activeView,
  saveLoading,
  saveTone,
  savedAtLabel,
  workflowName,
  warnings = [],
  supportsEasyMode = false,
  onWorkflowNameChange,
  onViewChange,
  onSaveRecipe,
}: RecipeStudioHeaderProps): ReactElement {
  const [editingWorkflowName, setEditingWorkflowName] = useState(false);

  function handleViewValueChange(value: string): void {
    if (value === "easy" || value === "editor" || value === "executions") {
      onViewChange(value);
    }
  }

  function closeWorkflowNameEditor(): void {
    if (workflowName.trim().length === 0) {
      onWorkflowNameChange("Untitled recipe");
    }
    setEditingWorkflowName(false);
  }

  function handleWorkflowNameKeyDown(
    event: KeyboardEvent<HTMLInputElement>,
  ): void {
    if (event.key === "Enter") {
      closeWorkflowNameEditor();
      return;
    }
    if (event.key === "Escape") {
      setEditingWorkflowName(false);
    }
  }

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-4 border-b px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        <div
          className="flex size-8 shrink-0 items-center justify-center rounded-lg corner-squircle border border-border/70 bg-muted/20"
          aria-hidden={true}
        >
          <HugeiconsIcon
            icon={CookBookIcon}
            className="size-4 text-muted-foreground"
          />
        </div>
        <div className="flex min-w-0 items-center gap-2">
          {editingWorkflowName ? (
            <Input
              value={workflowName}
              onChange={(event) => onWorkflowNameChange(event.target.value)}
              onBlur={closeWorkflowNameEditor}
              onKeyDown={handleWorkflowNameKeyDown}
              autoFocus={true}
              className="h-7 w-full max-w-[min(22rem,50vw)]"
              aria-label="Recipe name"
            />
          ) : (
            <button
              type="button"
              onClick={() => setEditingWorkflowName(true)}
              className="max-w-[min(22rem,50vw)] truncate text-sm font-semibold text-foreground hover:text-primary"
              title={workflowName}
              aria-label={`Edit recipe name: ${workflowName}`}
            >
              {workflowName}
            </button>
          )}
          <Badge variant="secondary" className="h-6 shrink-0 text-[10px]">
            {STATUS_MESSAGE_CLASS[saveTone]}
          </Badge>
          <span
            className="hidden max-w-[12rem] truncate text-xs text-muted-foreground sm:inline"
            title={savedAtLabel}
          >
            {savedAtLabel}
          </span>
        </div>
      </div>
      <div className="justify-self-center">
        <Tabs value={activeView} onValueChange={handleViewValueChange}>
          <TabsList>
            {supportsEasyMode && (
              <TabsTrigger value="easy">Easy</TabsTrigger>
            )}
            <TabsTrigger value="editor">
              {supportsEasyMode ? "Advanced" : "Editor"}
            </TabsTrigger>
            <TabsTrigger value="executions">Runs</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <div className="flex items-center justify-self-end gap-2">
        {warnings.length > 0 && (
          <Popover>
            <PopoverTrigger asChild={true}>
              <button
                type="button"
                className={`inline-flex h-6 shrink-0 items-center gap-1 rounded-md border px-2 text-[10px] font-medium ${RECIPE_STUDIO_WARNING_BADGE_TONE}`}
              >
                <HugeiconsIcon icon={Alert02Icon} className="size-3" />
                {warnings.length}
              </button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 p-0">
              <div className="border-b px-3 py-2">
                <p className="text-xs font-semibold text-foreground">
                  Graph warnings ({warnings.length})
                </p>
              </div>
              <ul className="max-h-60 overflow-y-auto py-1">
                {warnings.map((w) => (
                  <li
                    key={`${w.nodeId ?? "global"}-${w.message}`}
                    className="flex items-start gap-2 px-3 py-1.5"
                  >
                    <HugeiconsIcon
                      icon={
                        w.severity === "error" ? AlertDiamondIcon : Alert02Icon
                      }
                      className={`mt-0.5 size-3 shrink-0 ${w.severity === "error" ? "text-destructive" : RECIPE_STUDIO_WARNING_ICON_TONE}`}
                    />
                    <span className="text-xs text-muted-foreground">
                      {(w.nodeName || w.nodeId) && (
                        <span className="font-medium text-foreground">
                          {w.nodeName || w.nodeId}:{" "}
                        </span>
                      )}
                      {w.message}
                    </span>
                  </li>
                ))}
              </ul>
            </PopoverContent>
          </Popover>
        )}
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onSaveRecipe}
          disabled={saveLoading}
        >
          <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
          {saveLoading ? "Saving..." : "Save"}
        </Button>
      </div>
    </div>
  );
}
