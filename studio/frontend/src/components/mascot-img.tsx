// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ComponentProps, useState } from "react";

// Inline data URI so the fallback can never 404, unlike public-folder originals.
import fallbackMascot from "@/assets/mascot-fallback.webp?inline";

const LEADING_SLASH = /^\//;

// Resolve against the deploy base (subpath mounts) and encode spaced filenames.
export function publicAssetUrl(path: string): string {
  return encodeURI(import.meta.env.BASE_URL + path.replace(LEADING_SLASH, ""));
}

type MascotImgProps = { src: string } & Omit<
  ComponentProps<"img">,
  "src" | "onError"
>;

// Keying on src remounts the inner component, resetting retry state per source.
export function MascotImg(props: MascotImgProps) {
  return <MascotImgInner key={props.src} {...props} />;
}

type Stage = "primary" | "retry" | "fallback";

// Retries once with a cache-buster, then swaps to the bundled fallback; empty alt by default.
function MascotImgInner({ src, alt = "", ...rest }: MascotImgProps) {
  const [stage, setStage] = useState<Stage>("primary");

  const url = publicAssetUrl(src);
  const effectiveSrc =
    stage === "primary"
      ? url
      : stage === "retry"
        ? `${url}${url.includes("?") ? "&" : "?"}retry=1`
        : fallbackMascot;

  return (
    <img
      draggable={false}
      {...rest}
      src={effectiveSrc}
      alt={alt}
      onError={() => setStage((s) => (s === "primary" ? "retry" : "fallback"))}
    />
  );
}
