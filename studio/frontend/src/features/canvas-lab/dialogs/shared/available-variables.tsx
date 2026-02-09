import { Badge } from "@/components/ui/badge";
import type { ReactElement } from "react";
import { useCanvasLabStore } from "../../stores/canvas-lab";
import { getAvailableVariables } from "../../utils/variables";

type AvailableVariablesProps = {
  configId: string;
};

export function AvailableVariables({
  configId,
}: AvailableVariablesProps): ReactElement | null {
  const configs = useCanvasLabStore((state) => state.configs);
  const vars = getAvailableVariables(configs, configId);

  if (vars.length === 0) return null;

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 px-3 py-2">
      <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
        Available references
      </p>
      <div className="flex flex-wrap gap-1.5">
        {vars.map((v) => (
          <Badge
            key={v}
            variant="secondary"
            className="corner-squircle font-mono text-[11px]"
          >
            {`{{ ${v} }}`}
          </Badge>
        ))}
      </div>
    </div>
  );
}
