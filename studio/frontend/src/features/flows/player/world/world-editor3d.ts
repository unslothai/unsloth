// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import * as three from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

import type { ModelAsset, WorldProp3d } from "../../types";
import type { WorldMap } from "./procgen";
import type { PropLayer } from "./props-scene3d";
import {
  type EditorState,
  type GizmoMode,
  MAX_PROPS,
  type PropHistory,
  type WorldEditor,
  clampScale,
  commit,
  freeTileNear,
  makeProp,
  redo,
  removeProp,
  snapRotation,
  snapToTile,
  startHistory,
  undo,
  updateProp,
} from "./props3d";

export interface EditorContext {
  scene: three.Scene;
  camera: three.PerspectiveCamera;
  dom: HTMLCanvasElement;
  layer: PropLayer;
  exportRoot: three.Object3D;
  map: WorldMap;
  groundAt(x: number, y: number): number;
  player(): { x: number; y: number; yaw: number };
  openTile(x: number, y: number): boolean;
  onProps(props: WorldProp3d[]): void;
}

export interface WorldEditorRuntime extends WorldEditor {
  frame(): void;
}

const DEG = 180 / Math.PI;

export function createWorldEditor(ctx: EditorContext): WorldEditorRuntime {
  const { camera, dom, layer, map } = ctx;
  const saved = {
    pos: camera.position.clone(),
    rot: camera.rotation.clone(),
  };
  let h: PropHistory = startHistory(ctx.map.props ?? []);
  let selected: string | null = null;
  let mode: GizmoMode = "translate";
  const listeners = new Set<() => void>();
  const emit = () => {
    for (const cb of listeners) cb();
  };

  const me = ctx.player();
  const fx = -Math.sin(me.yaw);
  const fz = -Math.cos(me.yaw);
  const target = new three.Vector3(me.x + fx * 3, 0, me.y + fz * 3);
  target.y = ctx.groundAt(target.x, target.z);
  camera.position.set(target.x - fx * 7, target.y + 7, target.z - fz * 7);
  const orbit = new OrbitControls(camera, dom);
  orbit.target.copy(target);
  orbit.enableDamping = true;
  orbit.maxPolarAngle = 1.45;
  orbit.minDistance = 2;
  orbit.maxDistance = 80;
  orbit.update();

  const gizmo = new TransformControls(camera, dom);
  gizmo.setTranslationSnap(null);
  gizmo.setRotationSnap(Math.PI / 12);
  const helper = gizmo.getHelper();
  ctx.scene.add(helper);
  const applyMode = () => {
    gizmo.setMode(mode);
    gizmo.showX = mode !== "rotate";
    gizmo.showZ = mode !== "rotate";
    gizmo.showY = mode !== "translate";
  };
  applyMode();

  let scaleBase = 1;
  const onDragging = (e: { value: unknown }) => {
    orbit.enabled = !e.value;
    const obj = selected ? layer.objectOf(selected) : undefined;
    if (e.value) {
      scaleBase = obj?.scale.x ?? 1;
      return;
    }
    if (!obj || !selected) return;
    const snap = snapToTile(
      obj.position.x,
      obj.position.z,
      map.width,
      map.height,
    );
    const yaw = new three.Euler().setFromQuaternion(obj.quaternion, "YXZ").y;
    apply(
      updateProp(h.props, selected, {
        x: snap.x,
        y: snap.y,
        rot: snapRotation(yaw * DEG),
        scale: clampScale(obj.scale.x),
      }),
    );
  };
  const onObjectChange = () => {
    const obj = selected ? layer.objectOf(selected) : undefined;
    if (!obj) return;
    if (mode === "translate")
      obj.position.y = ctx.groundAt(obj.position.x, obj.position.z);
    if (mode === "scale") {
      const s = [obj.scale.x, obj.scale.y, obj.scale.z].reduce((a, b) =>
        Math.abs(b - scaleBase) > Math.abs(a - scaleBase) ? b : a,
      );
      obj.scale.setScalar(clampScale(s));
    }
  };
  gizmo.addEventListener("dragging-changed", onDragging);
  gizmo.addEventListener("objectChange", onObjectChange);

  const attach = () => {
    const obj = selected ? layer.objectOf(selected) : undefined;
    if (obj) gizmo.attach(obj);
    else gizmo.detach();
  };
  function apply(next: WorldProp3d[], keepHistory = false) {
    h = keepHistory ? { ...h, props: next } : commit(h, next);
    layer.sync(h.props);
    ctx.onProps(h.props);
    if (selected && !h.props.some((p) => p.id === selected)) selected = null;
    attach();
    emit();
  }

  const ray = new three.Raycaster();
  let down: { x: number; y: number } | null = null;
  const onDown = (e: PointerEvent) => {
    down = { x: e.clientX, y: e.clientY };
  };
  const onUp = (e: PointerEvent) => {
    const start = down;
    down = null;
    if (!start || gizmo.dragging || gizmo.axis) return;
    if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 5) return;
    const rect = dom.getBoundingClientRect();
    ray.setFromCamera(
      new three.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      ),
      camera,
    );
    const hit = ray.intersectObject(layer.root, true)[0];
    api.select(layer.idOf(hit?.object ?? null));
  };
  dom.addEventListener("pointerdown", onDown);
  dom.addEventListener("pointerup", onUp);

  const onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)
    )
      return;
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    if (mod && k === "z") api[e.shiftKey ? "redo" : "undo"]();
    else if (mod && k === "y") api.redo();
    else if (mod) return;
    else if (k === "g") api.setMode("translate");
    else if (k === "r") api.setMode("rotate");
    else if (k === "s") api.setMode("scale");
    else if (k === "x" || k === "delete" || k === "backspace") api.remove();
    else if (k === "escape") api.select(null);
    else return;
    e.preventDefault();
    e.stopPropagation();
  };
  window.addEventListener("keydown", onKey, true);

  const api: WorldEditorRuntime = {
    state(): EditorState {
      return {
        props: h.props,
        selected,
        mode,
        canUndo: h.past.length > 0,
        canRedo: h.future.length > 0,
      };
    },
    subscribe(cb) {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    add(kind) {
      if (h.props.length >= MAX_PROPS) return;
      const at = freeTileNear(
        map,
        { x: Math.floor(orbit.target.x), y: Math.floor(orbit.target.z) },
        ctx.openTile,
      );
      if (!at) return;
      const p = makeProp(
        kind,
        at,
        h.props.map((x) => x.id),
      );
      selected = p.id;
      apply([...h.props, p]);
    },
    async addModel(asset: ModelAsset, data: ArrayBuffer, name?: string) {
      if (h.props.length >= MAX_PROPS) return;
      await layer.addModel(asset.ref, data);
      const at = freeTileNear(
        map,
        { x: Math.floor(orbit.target.x), y: Math.floor(orbit.target.z) },
        ctx.openTile,
      );
      if (!at) return;
      const p = makeProp(
        "model",
        at,
        h.props.map((x) => x.id),
        {
          asset: { kind: "model", ref: asset.ref, mime: "model/gltf-binary" },
          ...(name ? { name: name.slice(0, 60) } : {}),
        },
      );
      selected = p.id;
      apply([...h.props, p]);
    },
    select(id) {
      selected = id && h.props.some((p) => p.id === id) ? id : null;
      attach();
      emit();
    },
    setMode(m) {
      mode = m;
      applyMode();
      emit();
    },
    update(id, patch) {
      apply(updateProp(h.props, id, patch));
    },
    remove(id = selected) {
      if (!id) return;
      apply(removeProp(h.props, id));
    },
    undo() {
      h = undo(h);
      apply(h.props, true);
    },
    redo() {
      h = redo(h);
      apply(h.props, true);
    },
    async exportGlb() {
      const was = helper.visible;
      helper.visible = false;
      try {
        const out = await new GLTFExporter().parseAsync(ctx.exportRoot, {
          binary: true,
          onlyVisible: true,
        });
        return new Blob([out as ArrayBuffer], { type: "model/gltf-binary" });
      } finally {
        helper.visible = was;
      }
    },
    frame() {
      orbit.update();
    },
    close() {
      window.removeEventListener("keydown", onKey, true);
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
      gizmo.removeEventListener("dragging-changed", onDragging);
      gizmo.removeEventListener("objectChange", onObjectChange);
      gizmo.detach();
      helper.removeFromParent();
      gizmo.dispose();
      orbit.dispose();
      listeners.clear();
      camera.position.copy(saved.pos);
      camera.rotation.copy(saved.rot);
    },
  };
  return api;
}
