// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import { Trash2Icon, UploadIcon } from "lucide-react";
import {
  type ChangeEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  type AgentSkill,
  deleteSkill,
  importSkillBundle,
  listSkills,
  setSkillEnabled,
  subscribeSkillCatalogChanges,
} from "./api/skills-api";

type PendingAction =
  | { type: "delete"; skill: AgentSkill }
  | { type: "replace"; file: File };

export function ChatSkillsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [skills, setSkills] = useState<AgentSkill[]>([]);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [busyName, setBusyName] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(
    null,
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSkills(await listSkills());
    } catch (error) {
      toast.error("Could not load skills", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      refresh();
      return subscribeSkillCatalogChanges(refresh);
    }
  }, [open, refresh]);

  const importBundle = async (file: File, replace: boolean) => {
    setImporting(true);
    try {
      const skill = await importSkillBundle(file, replace);
      toast.success(replace ? "Skill replaced" : "Skill imported", {
        description: skill.name,
      });
      await refresh();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!replace && message.includes("already installed")) {
        setPendingAction({ type: "replace", file });
      } else {
        toast.error("Could not import skill", { description: message });
      }
    } finally {
      setImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const onImportFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      importBundle(file, false);
    }
  };

  const toggleSkill = async (skill: AgentSkill, enabled: boolean) => {
    setBusyName(skill.name);
    try {
      await setSkillEnabled(skill.name, enabled);
      await refresh();
    } catch (error) {
      toast.error("Could not update skill", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusyName(null);
    }
  };

  const removeSkill = async (skill: AgentSkill) => {
    setBusyName(skill.name);
    try {
      await deleteSkill(skill.name);
      setSkills((current) =>
        current.filter((entry) => entry.name !== skill.name),
      );
      toast.success("Skill deleted", { description: skill.name });
    } catch (error) {
      toast.error("Could not delete skill", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusyName(null);
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Agent Skills</DialogTitle>
            <DialogDescription>
              Import portable Agent Skills bundles for use in chat.
            </DialogDescription>
          </DialogHeader>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/zip,.zip"
            className="hidden"
            onChange={onImportFile}
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">
              ZIP bundles need one SKILL.md. Text scripts, references, and
              templates can be loaded in chat.
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="shrink-0"
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
            >
              {importing ? <Spinner /> : <UploadIcon size={14} />}
              Import bundle
            </Button>
          </div>
          {loading ? (
            <div className="flex justify-center py-8">
              <Spinner />
            </div>
          ) : skills.length === 0 ? (
            <div className="rounded-md border border-dashed py-8 text-center text-sm text-muted-foreground">
              No skills installed yet.
            </div>
          ) : (
            <ul className="flex max-h-[55vh] flex-col divide-y overflow-y-auto rounded-md border">
              {skills.map((skill) => (
                <li
                  key={skill.name}
                  className="flex items-start justify-between gap-3 px-3 py-3"
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">{skill.name}</div>
                    <div className="mt-0.5 text-xs text-muted-foreground">
                      {skill.description}
                    </div>
                    {skill.compatibility ? (
                      <div className="mt-1 text-xs text-muted-foreground">
                        {skill.compatibility}
                      </div>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Switch
                      checked={skill.enabled}
                      disabled={busyName === skill.name}
                      onCheckedChange={(enabled) => toggleSkill(skill, enabled)}
                      aria-label={`Enable ${skill.name}`}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      disabled={busyName === skill.name}
                      onClick={() =>
                        setPendingAction({ type: "delete", skill })
                      }
                      aria-label={`Delete ${skill.name}`}
                    >
                      <Trash2Icon size={14} />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={pendingAction !== null}
        onOpenChange={(next) => {
          if (!next) {
            setPendingAction(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {pendingAction?.type === "delete"
                ? "Delete skill"
                : "Replace installed skill?"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {pendingAction?.type === "delete" ? (
                <>
                  Delete &quot;{pendingAction.skill.name}&quot; and all of its
                  files? This cannot be undone.
                </>
              ) : (
                <>
                  A skill with this name is already installed. Replace its full
                  bundle with the uploaded version?
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant={
                pendingAction?.type === "delete" ? "destructive" : "default"
              }
              onClick={() => {
                const action = pendingAction;
                setPendingAction(null);
                if (action?.type === "delete") {
                  removeSkill(action.skill);
                } else if (action?.type === "replace") {
                  importBundle(action.file, true);
                }
              }}
            >
              {pendingAction?.type === "delete" ? "Delete" : "Replace"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
