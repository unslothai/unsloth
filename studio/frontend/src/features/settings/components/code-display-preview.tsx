// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Streamdown } from "streamdown";
import { unslothDarkTheme, unslothLightTheme } from "@/components/assistant-ui/code-themes";
import { codePlugin } from "@/components/assistant-ui/shared-code-plugin";

// One long line shows Word wrap; =>, >=, !== and && show ligatures.
const SAMPLE = `\`\`\`ts
const isActive = (user: User): boolean => user.visits >= 3 && user.status !== "banned";

export function formatUser(user: User): string {
  const tags = [user.role, user.team, user.location, user.timezone].filter((tag) => tag !== undefined).join(" · ");
  return \`\${user.name} <\${user.email}>\${tags ? \` (\${tags})\` : ""}\`;
}
\`\`\``;

/** A chat code block rendered by the chat's own renderer, so every code setting shows here first. */
export function CodeDisplayPreview() {
  return (
    <div data-code-preview="" className="min-w-0">
      <Streamdown
        mode="static"
        plugins={{ code: codePlugin }}
        controls={{ code: false }}
        shikiTheme={[unslothLightTheme, unslothDarkTheme]}
      >
        {SAMPLE}
      </Streamdown>
    </div>
  );
}
