// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Badge } from "@/components/ui/badge";
import { type ReactElement, useMemo, useState } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { getAvailableVariableEntries } from "../../utils/variables";

type AvailableVariablesProps = {
  configId: string;
};

const USER_EXPANDED_FIELDS = [
  "first_name",
  "last_name",
  "sex",
  "city",
  "state",
  "age",
] as const;
const USER_BADGE_CLASS =
  "corner-squircle border-amber-500/25 bg-amber-500/10 font-mono text-[11px] text-amber-700 dark:text-amber-300";

export function AvailableVariables({
  configId,
}: AvailableVariablesProps): ReactElement | null {
  const [showUserFields, setShowUserFields] = useState(false);
  const configs = useRecipeStudioStore((state) => state.configs);
  const vars = getAvailableVariableEntries(configs, configId);
  const variableNames = useMemo(() => new Set(vars.map((entry) => entry.name)), [vars]);
  const hasUserRoot = variableNames.has("user");
  const userFieldEntries = useMemo(
    () =>
      USER_EXPANDED_FIELDS.map((field) => ({
        source: "column" as const,
        name: `user.${field}`,
      })).filter((entry) => !variableNames.has(entry.name)),
    [variableNames],
  );

  if (vars.length === 0) return null;

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 px-3 py-2">
      <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
        Available references
      </p>
      <div className="flex flex-wrap gap-1.5">
        {vars.map((v) => {
          const className =
            v.name === "user" || v.name.startsWith("user.")
              ? USER_BADGE_CLASS
              : v.source === "seed"
              ? "corner-squircle border-blue-500/25 bg-blue-500/10 font-mono text-[11px] text-blue-700 dark:text-blue-300"
              : "corner-squircle font-mono text-[11px]";
          if (v.name !== "user") {
            return (
              <Badge
                key={`${v.source}:${v.name}`}
                variant="secondary"
                className={className}
              >
                {`{{ ${v.name} }}`}
              </Badge>
            );
          }
          return (
            <button
              key={`${v.source}:${v.name}`}
              type="button"
              onClick={() => setShowUserFields((prev) => !prev)}
              className="cursor-pointer"
              aria-expanded={showUserFields}
            >
              <Badge variant="secondary" className={className}>
                {`{{ ${v.name} }}`}
              </Badge>
            </button>
          );
        })}
        {hasUserRoot && showUserFields &&
          userFieldEntries.map((entry) => (
            <Badge
              key={`user-expanded:${entry.name}`}
              variant="secondary"
              className={USER_BADGE_CLASS}
            >
              {`{{ ${entry.name} }}`}
            </Badge>
          ))}
      </div>
    </div>
  );
}
