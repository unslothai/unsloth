// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { getAvailableVariableEntries } from "../../utils/variables";
import { RECIPE_STUDIO_REFERENCE_BADGE_TONES } from "../../utils/ui-tones";

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
              ? RECIPE_STUDIO_REFERENCE_BADGE_TONES.user
              : v.source === "seed"
                ? RECIPE_STUDIO_REFERENCE_BADGE_TONES.seed
                : RECIPE_STUDIO_REFERENCE_BADGE_TONES.default;
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
              aria-label={showUserFields ? "Hide user fields" : "Show user fields"}
            >
              <Badge variant="secondary" className={className}>
                <span>{`{{ ${v.name} }}`}</span>
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  className={`size-3 transition-transform ${showUserFields ? "rotate-180" : ""}`}
                />
              </Badge>
            </button>
          );
        })}
        {hasUserRoot && showUserFields &&
          userFieldEntries.map((entry) => (
            <Badge
              key={`user-expanded:${entry.name}`}
              variant="secondary"
              className={RECIPE_STUDIO_REFERENCE_BADGE_TONES.user}
            >
              {`{{ ${entry.name} }}`}
            </Badge>
          ))}
      </div>
    </div>
  );
}
