// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::OsString;

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";

#[derive(Debug, PartialEq, Eq)]
enum RenderingPlan {
    ForceSharedMemory,
    PreserveEnvironment,
}

fn rendering_plan(env: impl Fn(&str) -> Option<OsString>) -> RenderingPlan {
    // Presence is an operator override, including `=0` and an empty value. Do
    // not combine the modern shared-memory switch with legacy instructions a
    // user may already carry in a launcher or environment.d file.
    if env(FORCE_SHARED_MEMORY).is_some() || env(DISABLE_DMABUF).is_some() {
        return RenderingPlan::PreserveEnvironment;
    }

    let Some(display) = env(WAYLAND_DISPLAY) else {
        return RenderingPlan::PreserveEnvironment;
    };
    if display.is_empty() {
        return RenderingPlan::PreserveEnvironment;
    }

    // WAYLAND_DISPLAY is inherited in XWayland sessions too. Respect an
    // explicit GDK X11 selection rather than penalizing that rendering path.
    if let Some(backends) = env(GDK_BACKEND) {
        let allows_wayland = backends
            .to_string_lossy()
            .split(',')
            .any(|backend| backend.trim().eq_ignore_ascii_case("wayland"));
        if !allows_wayland {
            return RenderingPlan::PreserveEnvironment;
        }
    }

    RenderingPlan::ForceSharedMemory
}

/// Select WebKitGTK's shared-memory transport before GTK/WebKit initialization.
///
/// Returns true when this launch applied the compatibility setting.
pub fn configure_wayland_renderer() -> bool {
    match rendering_plan(|name| std::env::var_os(name)) {
        RenderingPlan::ForceSharedMemory => {
            std::env::set_var(FORCE_SHARED_MEMORY, "1");
            true
        }
        RenderingPlan::PreserveEnvironment => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(vars: &[(&str, &str)]) -> RenderingPlan {
        rendering_plan(|name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        })
    }

    #[test]
    fn wayland_uses_shared_memory_transport() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::ForceSharedMemory
        );
    }

    #[test]
    fn x11_session_keeps_webkit_defaults() {
        assert_eq!(plan(&[]), RenderingPlan::PreserveEnvironment);
    }

    #[test]
    fn an_empty_wayland_display_is_not_a_wayland_session() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn an_explicit_x11_gdk_backend_wins() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (GDK_BACKEND, "x11")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn an_explicit_force_shm_value_is_preserved() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (FORCE_SHARED_MEMORY, "0")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn the_legacy_dmabuf_override_is_preserved() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (DISABLE_DMABUF, "1")]),
            RenderingPlan::PreserveEnvironment
        );
    }
}
