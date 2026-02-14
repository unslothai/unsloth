import { useUpdateNodeInternals } from "@xyflow/react";
import { useEffect, useMemo } from "react";

type InternalsSyncProps = {
  nodeIds: string[];
};

export function InternalsSync({ nodeIds }: InternalsSyncProps): null {
  const updateNodeInternals = useUpdateNodeInternals();
  const idsKey = useMemo(() => nodeIds.join("|"), [nodeIds]);
  const stableNodeIds = useMemo(() => nodeIds, [idsKey]);

  useEffect(() => {
    if (!idsKey) {
      return;
    }
    requestAnimationFrame(() => {
      updateNodeInternals(stableNodeIds);
    });
  }, [idsKey, stableNodeIds, updateNodeInternals]);

  return null;
}
