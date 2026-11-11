// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import * as three from "three";

import { SPLASH_RADIUS, type Verbs } from "./action-verbs";

const FLASH_S = 0.3;

export interface VerbFx {
  flash(x: number, y: number, z: number): void;
  sync(v: Verbs, dt: number): void;
  dispose(): void;
}

export function createVerbFx(scene: three.Scene, slots: number): VerbFx {
  const rockGeo = new three.IcosahedronGeometry(0.09, 0);
  const rockMat = new three.MeshLambertMaterial({ color: "#8d8a83" });
  const flashGeo = new three.SphereGeometry(1, 12, 8);
  const rocks: three.Mesh[] = [];
  const flashes: {
    mesh: three.Mesh<three.SphereGeometry, three.MeshBasicMaterial>;
    t: number;
  }[] = [];
  for (let i = 0; i < slots; i++) {
    const r = new three.Mesh(rockGeo, rockMat);
    r.visible = false;
    scene.add(r);
    rocks.push(r);
    const f = new three.Mesh(
      flashGeo,
      new three.MeshBasicMaterial({
        color: "#ffb43a",
        transparent: true,
        depthWrite: false,
      }),
    );
    f.visible = false;
    scene.add(f);
    flashes.push({ mesh: f, t: 1 });
  }
  return {
    flash(x, y, z) {
      const f = flashes.find((e) => e.t >= 1);
      if (!f) return;
      f.t = 0;
      f.mesh.position.set(x, z, y);
      f.mesh.visible = true;
    },
    sync(v, dt) {
      for (let i = 0; i < rocks.length; i++) {
        const p = v.shots[i];
        rocks[i].visible = p.live;
        if (p.live) rocks[i].position.set(p.x, p.z, p.y);
        const f = flashes[i];
        if (f.t < 1) {
          f.t = Math.min(1, f.t + dt / FLASH_S);
          f.mesh.scale.setScalar(0.3 + f.t * SPLASH_RADIUS);
          f.mesh.material.opacity = 0.55 * (1 - f.t);
          f.mesh.visible = f.t < 1;
        }
      }
    },
    dispose() {
      for (const m of [...rocks, ...flashes.map((f) => f.mesh)])
        m.removeFromParent();
      for (const f of flashes) f.mesh.material.dispose();
      rockGeo.dispose();
      rockMat.dispose();
      flashGeo.dispose();
    },
  };
}
