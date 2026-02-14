import { useUpdateNodeInternals } from "@xyflow/react";
import { useEffect, useMemo, useRef } from "react";

type InternalsSyncProps = {
  nodeIds: string[];
};

export function InternalsSync({ nodeIds }: InternalsSyncProps): null {
  const updateNodeInternals = useUpdateNodeInternals();
  const idsKey = useMemo(() => nodeIds.join("|"), [nodeIds]);
  const nodeIdsRef = useRef(nodeIds);
  nodeIdsRef.current = nodeIds;

  useEffect(() => {
    if (!idsKey) {
      return;
    }
    const raf = requestAnimationFrame(() => {
      updateNodeInternals(nodeIdsRef.current);
    });
    return () => cancelAnimationFrame(raf);
  }, [idsKey, updateNodeInternals]);

  return null;
}
