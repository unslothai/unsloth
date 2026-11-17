// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useHfTokenStore,
  useOnlineStatus,
  useSelectedModelMetadata,
} from "@/features/hub";
import { modelInfoFacts } from "./model-info-facts";

/** Hub facts for a repo; mounted only while its section is open, so a closed one fetches nothing. */
export function ModelInfo({ repoId }: { repoId: string }) {
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const { result, error } = useSelectedModelMetadata(repoId, {
    accessToken: hfToken || undefined,
    enabled: true,
    online,
  });
  const facts = online && result ? modelInfoFacts(result) : null;
  if (!facts) {
    return (
      <p className="text-ui-11 text-muted-foreground">
        {!online
          ? "Connect to the internet to look up model details."
          : error
            ? "Could not reach Hugging Face for this model."
            : "Loading details..."}
      </p>
    );
  }
  return (
    <dl className="space-y-2">
      {facts.map(([label, value]) => (
        <div key={label} className="flex items-baseline justify-between gap-3">
          <dt className="shrink-0 text-ui-12 text-muted-foreground">{label}</dt>
          <dd className="min-w-0 break-all text-right text-ui-12 text-foreground">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
