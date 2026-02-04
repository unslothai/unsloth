import type { ReactElement } from "react";
import type { NodeConfig } from "../../types";
import { getConfigErrors } from "../../utils";

export function ValidationBanner({
  config,
}: {
  config: NodeConfig | null;
}): ReactElement | null {
  const errors = getConfigErrors(config);
  if (errors.length === 0) {
    return null;
  }
  return (
    <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
      <p className="font-semibold">Fix before run</p>
      <ul className="mt-1 list-disc pl-4">
        {errors.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
