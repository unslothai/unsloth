// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A deterministic stand-in for React's hooks, for components loaded with `loadWithStubs`.
// Hooks keep their slot across `render()` calls like React does; effects run synchronously
// when their deps change, `render()` repeats while they set state, and `unmount()` runs cleanups.

type Cleanup = (() => void) | undefined | void;

export function fakeReact({ runEffects = true }: { runEffects?: boolean } = {}) {
  const slots: unknown[] = [];
  const cleanups = new Map<number, Cleanup>();
  let cursor = 0;
  let dirty = false;
  const depsChanged = (prev: unknown[] | undefined, next: unknown[] | undefined) =>
    !prev || !next || prev.length !== next.length || prev.some((d, i) => !Object.is(d, next[i]));

  const react = {
    useState<T>(initial: T | (() => T)) {
      const i = cursor++;
      if (!(i in slots))
        slots[i] = typeof initial === "function" ? (initial as () => T)() : initial;
      const set = (next: T | ((prev: T) => T)) => {
        slots[i] =
          typeof next === "function" ? (next as (prev: T) => T)(slots[i] as T) : next;
        dirty = true;
      };
      return [slots[i] as T, set] as const;
    },
    useRef<T>(initial: T) {
      const i = cursor++;
      if (!(i in slots)) slots[i] = { current: initial };
      return slots[i] as { current: T };
    },
    useMemo<T>(make: () => T, deps?: unknown[]) {
      const i = cursor++;
      const held = slots[i] as { deps?: unknown[]; value: T } | undefined;
      if (!held || depsChanged(held.deps, deps)) slots[i] = { deps, value: make() };
      return (slots[i] as { value: T }).value;
    },
    useCallback<T>(fn: T, deps?: unknown[]) {
      return react.useMemo(() => fn, deps);
    },
    useEffect(effect: () => Cleanup, deps?: unknown[]) {
      const i = cursor++;
      if (!runEffects) return;
      const held = slots[i] as { deps?: unknown[] } | undefined;
      if (held && !depsChanged(held.deps, deps)) return;
      slots[i] = { deps };
      cleanups.get(i)?.();
      cleanups.set(i, effect());
    },
    useLayoutEffect(effect: () => Cleanup, deps?: unknown[]) {
      react.useEffect(effect, deps);
    },
    useContext: () => null,
    memo: <F>(fn: F) => fn,
  };
  return {
    react,
    /** Render a component until it settles: state set by its effects re-renders, as React would. */
    render<P, R>(component: (props: P) => R, props: P): R {
      for (let pass = 0; ; pass++) {
        if (pass > 20) throw new Error("render loop: effects keep setting state");
        cursor = 0;
        dirty = false;
        const out = component(props);
        if (!dirty) return out;
      }
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup?.();
      cleanups.clear();
    },
  };
}

/** A zustand-style hook over a fixed state: `useX(selector)` and `useX.getState()`. */
export function selectorStore<S>(state: S) {
  return Object.assign(<T>(select: (s: S) => T) => select(state), {
    getState: () => state,
  });
}
