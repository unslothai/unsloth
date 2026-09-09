// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

import {
  type ProjectInstructions,
  type ProjectSkills,
  getProjectInstructions,
  getProjectSkills,
  initializeProjectAgents,
  subscribeProjectGuidanceUpdated,
} from "../api/project-guidance-api";

export function ProjectGuidancePanel({
  projectId,
}: {
  projectId: string;
}) {
  const [instructions, setInstructions] = useState<ProjectInstructions | null>(
    null,
  );
  const [skills, setSkills] = useState<ProjectSkills | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRevision = useRef(0);
  const refresh = useCallback(async () => {
    const revision = ++requestRevision.current;
    try {
      const [nextInstructions, nextSkills] = await Promise.all([
        getProjectInstructions(projectId),
        getProjectSkills(projectId),
      ]);
      if (revision !== requestRevision.current) {
        return;
      }
      setInstructions(nextInstructions);
      setSkills(nextSkills);
      setError(null);
    } catch (nextError) {
      if (revision !== requestRevision.current) {
        return;
      }
      setError(
        nextError instanceof Error
          ? nextError.message
          : "Could not load project guidance.",
      );
    } finally {
      if (revision === requestRevision.current) {
        setLoading(false);
      }
    }
  }, [projectId]);

  useEffect(() => {
    let disposed = false;
    // A discarded Strict Mode mount should not start a network request.
    queueMicrotask(() => {
      if (!disposed) void refresh();
    });
    const unsubscribe = subscribeProjectGuidanceUpdated(projectId, () => {
      refresh().catch(() => undefined);
    });
    return () => {
      disposed = true;
      requestRevision.current += 1;
      unsubscribe();
    };
  }, [projectId, refresh]);

  async function createAgentsFile(): Promise<void> {
    if (creating) {
      return;
    }
    setCreating(true);
    try {
      const result = await initializeProjectAgents(projectId);
      setInstructions(result.instructions);
      toast.success(
        result.created ? "Created AGENTS.md" : `${result.path} already exists`,
        {
          description: result.created
            ? "Future agent turns will use these project instructions."
            : "The existing file was left unchanged.",
        },
      );
    } catch (nextError) {
      toast.error("Could not create AGENTS.md", {
        description: nextError instanceof Error ? nextError.message : undefined,
      });
    } finally {
      setCreating(false);
    }
  }

  const rootInstruction = instructions?.layers.find(
    (layer) =>
      layer.scope === "." &&
      (layer.path === "AGENTS.override.md" || layer.path === "AGENTS.md"),
  );

  return (
    <div className="mt-3 border-t border-border/70 pt-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-foreground">Agent context</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Repository instructions and project skills are added to agent turns.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={
              creating || loading || Boolean(rootInstruction)
            }
            onClick={createAgentsFile}
          >
            {creating
              ? "Creating..."
              : rootInstruction
                ? `${rootInstruction.path} found`
                : "Create AGENTS.md"}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={loading}
            onClick={() => {
              setLoading(true);
              void refresh();
            }}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </div>

      {error ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {error}
          </p>
        ) : (
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl bg-background/70 px-3 py-2">
              <p className="text-xs font-medium text-foreground">
                AGENTS.md
                {instructions
                  ? ` (${instructions.layers.length} layer${instructions.layers.length === 1 ? "" : "s"})`
                  : ""}
              </p>
              {instructions && instructions.layers.length > 0 ? (
                <div className="mt-2 space-y-2">
                  {instructions.layers.map((layer) => (
                    <details key={layer.path}>
                      <summary className="cursor-pointer text-xs text-muted-foreground">
                        {layer.path}
                        {layer.truncated ? " (truncated)" : ""}
                      </summary>
                      <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-muted-foreground">
                        {layer.content}
                      </pre>
                    </details>
                  ))}
                </div>
              ) : (
                <p className="mt-1 text-xs text-muted-foreground">
                  {loading
                    ? "Resolving instructions..."
                    : "No AGENTS.md found."}
                </p>
              )}
              {instructions && instructions.issues.length > 0 ? (
                <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                  {instructions.issues.length} instruction path
                  {instructions.issues.length === 1 ? " was" : "s were"}{" "}
                  excluded.
                </p>
              ) : null}
            </div>

            <div className="rounded-xl bg-background/70 px-3 py-2">
              <p className="text-xs font-medium text-foreground">
                Project skills
                {skills ? ` (${skills.skills.length})` : ""}
              </p>
              {skills && skills.skills.length > 0 ? (
                <div className="mt-2 space-y-2">
                  {skills.skills.map((skill) => (
                    <div key={`${skill.name}:${skill.path}`}>
                      <p className="text-xs font-medium text-foreground">
                        ${skill.name}
                        {skill.ambiguous ? " (duplicate name)" : ""}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {skill.description}
                      </p>
                      <code className="block break-all text-[10px] text-muted-foreground/80">
                        {skill.path}
                      </code>
                      {skill.ambiguous ? (
                        <p className="text-[10px] text-amber-600 dark:text-amber-400">
                          Explicit selection is disabled until this duplicate
                          skill name is resolved.
                        </p>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-1 text-xs text-muted-foreground">
                  {loading
                    ? "Discovering skills..."
                    : "No .agents/skills packages found."}
                </p>
              )}
              {skills && skills.issues.length > 0 ? (
                <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                  {skills.issues.length} skill path
                  {skills.issues.length === 1 ? " was" : "s were"} excluded.
                </p>
              ) : null}
            </div>
          </div>
      )}
    </div>
  );
}
