// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, useState } from "react";
import { useCharacterLightbox } from "./lightbox-store";

const CharacterLightbox = lazy(() =>
  import("./components/character-lightbox").then((m) => ({ default: m.CharacterLightbox })),
);

/** Mounted once in the shell; the viewer itself loads the first time a picture is opened. */
export function CharacterLightboxHost() {
  const opened = useCharacterLightbox((s) => s.subject !== null);
  const [loaded, setLoaded] = useState(false);
  // Stays mounted after the first open, so closing can animate.
  if (opened && !loaded) setLoaded(true);
  if (!loaded && !opened) return null;
  return (
    <Suspense fallback={null}>
      <CharacterLightbox />
    </Suspense>
  );
}
