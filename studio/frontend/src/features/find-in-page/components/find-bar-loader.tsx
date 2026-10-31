// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import FindBar, { type FindBarProps } from "./find-bar.tsx";

/** Static implementation in a lazy entry lets Vite fetch all deps in parallel. */
// biome-ignore lint/style/noDefaultExport: React.lazy requires the component as a default export.
export default function FindBarLoader(props: FindBarProps) {
  return <FindBar {...props} />;
}
