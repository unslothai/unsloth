import { useUpdateNodeInternals } from "@xyflow/react";
import { useEffect } from "react";

type InternalsSyncProps = {
  nodeIds: string[];
};

export function InternalsSync({ nodeIds }: InternalsSyncProps): null {
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    if (nodeIds.length === 0) {
      return;
    }
    requestAnimationFrame(() => {
      updateNodeInternals(nodeIds);
      requestAnimationFrame(() => {
        updateNodeInternals(nodeIds);
      });
    });
  }, [nodeIds, updateNodeInternals]);

  return null;
}
