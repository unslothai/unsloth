// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IncrementalMarkdownRender } from "../src/components/assistant-ui/streaming-render-schedule.ts";

export const committedBlockContents = (
  render: IncrementalMarkdownRender,
): string[] => render.chunks.flatMap((chunk) =>
  chunk.blocks.map((block) => block.content),
);

export const renderBlockContents = (
  render: IncrementalMarkdownRender,
): string[] => [
  ...committedBlockContents(render),
  ...render.tail.map((block) => block.content),
];
