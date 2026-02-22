import { Badge } from "@/components/ui/badge";
import type { ReactElement } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { getAvailableVariableEntries } from "../../utils/variables";

type AvailableVariablesProps = {
  configId: string;
};

export function AvailableVariables({
  configId,
}: AvailableVariablesProps): ReactElement | null {
  const configs = useRecipeStudioStore((state) => state.configs);
  const vars = getAvailableVariableEntries(configs, configId);

  if (vars.length === 0) return null;

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 px-3 py-2">
      <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
        Available references
      </p>
      <div className="flex flex-wrap gap-1.5">
        {vars.map((v) => (
          <Badge
            key={`${v.source}:${v.name}`}
            variant="secondary"
            className={
              v.source === "seed"
                ? "corner-squircle border-blue-500/25 bg-blue-500/10 font-mono text-[11px] text-blue-700 dark:text-blue-300"
                : "corner-squircle font-mono text-[11px]"
            }
          >
            {`{{ ${v.name} }}`}
          </Badge>
        ))}
      </div>
    </div>
  );
}
