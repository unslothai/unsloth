// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import FindBar, { type FindBarProps } from "./find-bar.tsx";

/**
 * A stable lazy entry that keeps the controller independent of the on-demand UI and engine.
 * Keeping the implementation static inside this entry lets Vite fetch every dependency in
 * parallel, rather than making the first Ctrl/Cmd+F wait through a second network waterfall.
 */
// biome-ignore lint/style/noDefaultExport: React.lazy requires the component as a default export.
export default function FindBarLoader(props: FindBarProps) {
  return <FindBar {...props} />;
}
