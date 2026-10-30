// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import "./src/index.css";
import { createRoot } from "react-dom/client";
import { MascotImg } from "./src/components/mascot-img";
import { useAppearanceCustomStore } from "./src/features/settings/stores/appearance-custom-store";
import { OwnerAvatar } from "./src/features/hub/catalog/owner-avatar";
import { UserAvatar } from "./src/features/profile/components/user-avatar";

// The five production locations all render the shared decorative component.
// Avatar controls deliberately reuse sloth art: asset-name filtering would hide them.
const decorations = {
  greeting: "Sloth emojis/large sloth heart.png",
  authentication: "Sloth emojis/large sloth wave.png",
  notFound: "Sloth emojis/sloth shy large.png",
  canvas: "Sloth emojis/sloth w pc transparent.png",
  training: "unsloth-gem.png",
};

export function Smoke() {
  const enabled = useAppearanceCustomStore((s) => s.customization.showMascots);
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <main style={{ fontFamily: "sans-serif", padding: 24 }}>
      <h1>Decorative mascot component verification</h1>
      <button type="button" aria-pressed={enabled} onClick={() => patch({ showMascots: !enabled })}>
        Decorative mascots
      </button>
      <div style={{ display: "flex", gap: 32, margin: "30px 0" }}>
        {Object.entries(decorations).map(([name, src]) => (
          <section key={name} data-decoration={name}>
            <h2>{name}</h2><MascotImg src={src} width={96} height={96} />
          </section>
        ))}
      </div>
      <section data-identity="user"><h2>User avatar</h2>
        <UserAvatar name="Sloth user" imageUrl="/Sloth%20emojis/large%20sloth%20heart.png" size="sm" />
      </section>
      <section data-identity="model-owner"><h2>Model owner brand</h2>
        <OwnerAvatar owner="unsloth" remote={false} />
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<Smoke />);
