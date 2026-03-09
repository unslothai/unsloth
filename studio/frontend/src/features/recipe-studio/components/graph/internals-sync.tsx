// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

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
