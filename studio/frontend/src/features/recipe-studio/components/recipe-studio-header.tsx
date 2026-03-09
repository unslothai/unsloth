// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { type KeyboardEvent, type ReactElement, useState } from "react";
import {
  CookBookIcon,
  FloppyDiskIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { RecipeStudioView } from "../execution-types";

type StatusTone = "success" | "error";

type RecipeStudioHeaderProps = {
  activeView: RecipeStudioView;
  saveLoading: boolean;
  saveTone: StatusTone;
  savedAtLabel: string;
  workflowName: string;
  onWorkflowNameChange: (value: string) => void;
  onViewChange: (view: RecipeStudioView) => void;
  onSaveRecipe: () => void;
};

const STATUS_MESSAGE_CLASS: Record<StatusTone, string> = {
  success: "Saved",
  error: "Unsaved changes",
};

export function RecipeStudioHeader({
  activeView,
  saveLoading,
  saveTone,
  savedAtLabel,
  workflowName,
  onWorkflowNameChange,
  onViewChange,
  onSaveRecipe,
}: RecipeStudioHeaderProps): ReactElement {
  const [editingWorkflowName, setEditingWorkflowName] = useState(false);

  function handleViewValueChange(value: string): void {
    if (value === "editor" || value === "executions") {
      onViewChange(value);
    }
  }

  function closeWorkflowNameEditor(): void {
    if (workflowName.trim().length === 0) {
      onWorkflowNameChange("Unnamed");
    }
    setEditingWorkflowName(false);
  }

  function handleWorkflowNameKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
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
        <button
          type="button"
          className="flex size-8 items-center justify-center rounded-lg corner-squircle border border-border/70 bg-muted/20"
          aria-label="Recipe icon"
        >
          <HugeiconsIcon icon={CookBookIcon} className="size-4 text-muted-foreground" />
        </button>
        <div className="flex min-w-0 items-center gap-2">
          {editingWorkflowName ? (
            <Input
              value={workflowName}
              onChange={(event) => onWorkflowNameChange(event.target.value)}
              onBlur={closeWorkflowNameEditor}
              onKeyDown={handleWorkflowNameKeyDown}
              autoFocus={true}
              className="h-7 w-[180px]"
            />
          ) : (
            <button
              type="button"
              onClick={() => setEditingWorkflowName(true)}
              className="truncate text-sm font-semibold text-foreground hover:text-primary"
            >
              {workflowName}
            </button>
          )}
          <Badge variant="secondary" className="h-6 text-[10px]">
            {STATUS_MESSAGE_CLASS[saveTone]}
          </Badge>
          <span className="text-xs text-muted-foreground">{savedAtLabel}</span>
        </div>
      </div>
      <div className="justify-self-center">
        <Tabs value={activeView} onValueChange={handleViewValueChange}>
          <TabsList>
            <TabsTrigger value="editor">Editor</TabsTrigger>
            <TabsTrigger value="executions">Executions</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <div className="flex items-center justify-self-end gap-2">
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onSaveRecipe}
          disabled={saveLoading}
        >
          <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
          {saveLoading ? "Saving..." : "Save Recipe"}
        </Button>
      </div>
    </div>
  );
}
