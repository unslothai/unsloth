// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::{OsStr, OsString};

const APPIMAGE: &str = "APPIMAGE";
const GLES_V2: &[u8] = b"libGLESv2.so.2\0";

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
// disable-nvidia-dmabuf.patch's own opt-out, read inside isNVIDIA(). WebKit returns on
// DISABLE_DMABUF first, so it never gets there unless we honour it ourselves.
const FORCE_DMABUF: &str = "WEBKIT_FORCE_DMABUF_RENDERER";
// Names the variable we set, so a relaunch tells our own inherited output from an
// operator's value. Tauri's process::restart does not env_clear. WebKit never reads it.
const APPLIED_WORKAROUND: &str = "UNSLOTH_WEBKIT_RENDERER_WORKAROUND";
const FORCE_SHARED_MEMORY_MIN_VERSION: (u32, u32) = (2, 44);
// both the proprietary and open nvidia modules publish this; nouveau does not and is unaffected
const NVIDIA_DRIVER_VERSION_PATH: &str = "/proc/driver/nvidia/version";

const NVIDIA_REASON: &str = "NVIDIA driver loaded (no Wayland session)";
const NVIDIA_WAYLAND_REASON: &str = "NVIDIA driver loaded (Wayland session)";
const NVIDIA_APPIMAGE_GLES_REASON: &str =
    "NVIDIA driver loaded; AppImage without a usable GLES library";
const WAYLAND_REASON: &str = "Wayland session";
const APPIMAGE_GLES_REASON: &str = "AppImage without a usable GLES library";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderingWorkaround {
    ForceSharedMemory,
    DisableDmabuf,
}

impl RenderingWorkaround {
    fn variable(self) -> &'static str {
        match self {
            Self::ForceSharedMemory => FORCE_SHARED_MEMORY,
            Self::DisableDmabuf => DISABLE_DMABUF,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RenderingPlan {
    /// The workaround to apply, and why, for the startup log.
    Apply(RenderingWorkaround, &'static str),
    PreserveEnvironment,
}

fn backend_allows_wayland(backends: &OsStr) -> bool {
    backends.to_string_lossy().split(',').any(|backend| {
        let backend = backend.trim();
        backend == "*" || backend.eq_ignore_ascii_case("wayland")
    })
}

fn supports_force_shared_memory((major, minor, _micro): (u32, u32, u32)) -> bool {
    (major, minor) >= FORCE_SHARED_MEMORY_MIN_VERSION
}

/// `forceDMABuf && *forceDMABuf != '0'` from disable-nvidia-dmabuf.patch: first byte
/// only, so an empty value is a request and `0` is not.
fn force_dmabuf_requested(value: &OsStr) -> bool {
    value.as_encoded_bytes().first() != Some(&b'0')
}

fn rendering_plan(
    env: impl Fn(&str) -> Option<OsString>,
    webkit_version: (u32, u32, u32),
    gles_usable: bool,
    nvidia_driver_loaded: bool,
) -> RenderingPlan {
    // Either renderer variable is an operator override when present, `=0` and an empty
    // value included, unless the marker says we wrote it. Without that test a launch
    // reads its own inherited output back as an instruction and the first decision is
    // pinned for the life of the process tree, even after the display server changes.
    //
    // DISABLE_DMABUF=0 is how a host this over-triggers on gets its accelerated path
    // back: WebKit reads it as unset (`g_strcmp0(v, "0")`), and a patched library's own
    // GL_VENDOR probe then declines on an iGPU. An empty value is not that request.
    let applied_by_this_app = env(APPLIED_WORKAROUND);
    let ours = |variable: &str| applied_by_this_app.as_deref() == Some(OsStr::new(variable));

    if env(DISABLE_DMABUF).is_some() && !ours(DISABLE_DMABUF) {
        return RenderingPlan::PreserveEnvironment;
    }
    if env(FORCE_SHARED_MEMORY).is_some() && !ours(FORCE_SHARED_MEMORY) {
        return RenderingPlan::PreserveEnvironment;
    }

    // Stands the NVIDIA branch down and nothing else: it answers that patch's question,
    // not the missing-GLES one below, and that launch cannot render without its fallback.
    let force_dmabuf = env(FORCE_DMABUF)
        .as_deref()
        .is_some_and(force_dmabuf_requested);

    let is_appimage = env(APPIMAGE).is_some();
    let configured_backends = env(GDK_BACKEND);
    let explicit_wayland = configured_backends
        .as_deref()
        .map(backend_allows_wayland)
        .unwrap_or(false);
    // GDK can connect to the default wayland-0 socket when its backend is explicitly
    // Wayland even if WAYLAND_DISPLAY is absent. Conversely, WAYLAND_DISPLAY is inherited
    // by XWayland apps, so an X11-only GDK_BACKEND must take precedence.
    let wayland_session = explicit_wayland
        || (configured_backends.is_none()
            && env(WAYLAND_DISPLAY)
                .map(|display| !display.is_empty())
                .unwrap_or(false));

    // The DMA-BUF transport breaks on the proprietary driver on either display server, so
    // this cannot be gated on a Wayland session. Upstream declined the fix (bug 262607
    // WONTFIX, PR 18614 closed), so Debian and Ubuntu carry disable-nvidia-dmabuf.patch
    // and Fedora, Arch and the tarballs do not. Both shipped artifacts get a patched
    // library anyway (the .deb from the host, the AppImage bundles Ubuntu 22.04's), so
    // here this covers hosts where that patch's GL_VENDOR probe disagrees with the module
    // probe, plus the GBM open and throwaway GL context isNVIDIA() does at startup.
    //
    // The two switches are not interchangeable, so pick per failure mode, not per GPU:
    //   Wayland  DISABLE_DMABUF. The failure is the explicit-sync disconnect, and
    //            FORCE_SHM routes every commit down the wl_shm path that trips it
    //            (bug 315436). It is also the switch reported to fix Error 71.
    //   X11      FORCE_SHM. No explicit-sync protocol there, the failure is hardware
    //            DMA-BUF allocation, and shared memory fixes it without the empty set.
    // The empty set is not just slower: DISABLE_DMABUF returns before the SharedMemory
    // add, so checkRequirements() is false, AcceleratedBackingStore::create() returns
    // nullptr, and webkitWebViewBaseEnterAcceleratedCompositingMode() dereferences it
    // behind an ASSERT release builds drop. block/buzz#3654 hits that SIGSEGV on NVIDIA
    // X11, on the same iGPU-presenting topology the probe below over-triggers on.
    //
    // That probe is module presence, not the GPU that will render, so a PRIME laptop on
    // its iGPU takes a workaround it does not need. Deliberate: reading the rendering GPU
    // needs a GL context and this runs before GTK init so that none exists. Those hosts
    // opt out with WEBKIT_DISABLE_DMABUF_RENDERER=0.
    if nvidia_driver_loaded && !force_dmabuf {
        let missing_appimage_gles = is_appimage && !gles_usable;
        let reason = if missing_appimage_gles {
            NVIDIA_APPIMAGE_GLES_REASON
        } else if wayland_session {
            NVIDIA_WAYLAND_REASON
        } else {
            NVIDIA_REASON
        };
        // FORCE_SHM still reaches the failing libepoxy path in AppImages, and 2.44 is
        // where WebKitGTK started reading it at all, so both keep the stronger switch.
        let workaround = if wayland_session
            || missing_appimage_gles
            || !supports_force_shared_memory(webkit_version)
        {
            RenderingWorkaround::DisableDmabuf
        } else {
            RenderingWorkaround::ForceSharedMemory
        };
        return RenderingPlan::Apply(workaround, reason);
    }

    // libepoxy loads GLES at runtime; disabling DMA-BUF handles a missing copy (#8343).
    if is_appimage && !gles_usable {
        return RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON);
    }

    if !wayland_session {
        return RenderingPlan::PreserveEnvironment;
    }

    let workaround = if is_appimage {
        // FORCE_SHM still reaches the failing libepoxy path in AppImages.
        RenderingWorkaround::DisableDmabuf
    } else if supports_force_shared_memory(webkit_version) {
        RenderingWorkaround::ForceSharedMemory
    } else {
        // FORCE_SHM was added with WebKitGTK 2.44. Older host libraries ignore
        // it, so use the renderer-disable workaround supported by 2.42.
        RenderingWorkaround::DisableDmabuf
    };
    RenderingPlan::Apply(workaround, WAYLAND_REASON)
}

fn nvidia_driver_loaded() -> bool {
    std::path::Path::new(NVIDIA_DRIVER_VERSION_PATH).exists()
}

fn gles_is_usable() -> bool {
    // Match libepoxy's runtime load, including unresolved dependencies.
    unsafe {
        let handle = libc::dlopen(GLES_V2.as_ptr().cast(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            false
        } else {
            libc::dlclose(handle);
            true
        }
    }
}

fn runtime_webkit_version() -> (u32, u32, u32) {
    // These accessors read compile-time version constants from the already
    // linked WebKitGTK library; they do not initialize GTK or a web view.
    unsafe {
        (
            webkit2gtk_sys::webkit_get_major_version(),
            webkit2gtk_sys::webkit_get_minor_version(),
            webkit2gtk_sys::webkit_get_micro_version(),
        )
    }
}

/// Select a compatible WebKitGTK rendering transport before GTK initialization.
///
/// Returns the variable applied by this launch and the reason for it, if any.
pub fn configure_renderer() -> Option<(&'static str, &'static str)> {
    let gles_usable = std::env::var_os(APPIMAGE).is_none() || gles_is_usable();
    match rendering_plan(
        |name| std::env::var_os(name),
        runtime_webkit_version(),
        gles_usable,
        nvidia_driver_loaded(),
    ) {
        RenderingPlan::Apply(workaround, reason) => {
            let variable = workaround.variable();
            // A relaunch deciding the other way inherited the previous choice, and WebKit
            // reads DISABLE_DMABUF before FORCE_SHM, so a stale one would outrank it.
            // Only a variable we claimed is cleared; an operator's never reaches here.
            if let Some(previous) = std::env::var_os(APPLIED_WORKAROUND) {
                if previous != OsStr::new(variable) {
                    std::env::remove_var(previous);
                }
            }
            std::env::set_var(variable, "1");
            // Claim it, so the next launch re-decides rather than reading it as an override.
            std::env::set_var(APPLIED_WORKAROUND, variable);
            Some((variable, reason))
        }
        RenderingPlan::PreserveEnvironment => {
            // Nothing is ours now; a stale claim would suppress an override next relaunch.
            std::env::remove_var(APPLIED_WORKAROUND);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODERN_WEBKIT: (u32, u32, u32) = (2, 44, 0);

    fn plan_on_host(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
        nvidia_driver_loaded: bool,
    ) -> RenderingPlan {
        rendering_plan(
            |name| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(value))
            },
            webkit_version,
            gles_usable,
            nvidia_driver_loaded,
        )
    }

    fn plan_with_version_and_gles(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
    ) -> RenderingPlan {
        plan_on_host(vars, webkit_version, gles_usable, false)
    }

    fn plan_with_version(vars: &[(&str, &str)], webkit_version: (u32, u32, u32)) -> RenderingPlan {
        plan_with_version_and_gles(vars, webkit_version, true)
    }

    fn plan(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_with_version(vars, MODERN_WEBKIT)
    }

    fn plan_on_nvidia(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_on_host(vars, MODERN_WEBKIT, true, true)
    }

    #[test]
    fn wayland_uses_shared_memory_transport() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn explicit_wayland_without_a_display_variable_still_applies() {
        assert_eq!(
            plan(&[(GDK_BACKEND, "wayland")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn an_x11_session_off_nvidia_keeps_webkit_defaults() {
        assert_eq!(plan(&[]), RenderingPlan::PreserveEnvironment);
    }

    #[test]
    fn nvidia_on_x11_drops_to_shared_memory_rather_than_the_empty_transport_set() {
        // No explicit-sync protocol to violate here, and DISABLE_DMABUF leaves the null
        // backing store release builds dereference (block/buzz#3654).
        assert_eq!(
            plan_on_nvidia(&[]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_under_an_explicit_x11_backend_still_applies() {
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "x11")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_on_x11_below_244_keeps_the_legacy_renderer_switch() {
        // FORCE_SHM arrived in 2.44; older libraries ignore it entirely.
        assert_eq!(
            plan_on_host(&[], (2, 42, 7), true, true),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_appimage_without_gles_reports_both_causes() {
        assert_eq!(
            plan_on_host(
                &[(APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false,
                true
            ),
            RenderingPlan::Apply(
                RenderingWorkaround::DisableDmabuf,
                NVIDIA_APPIMAGE_GLES_REASON
            )
        );
    }

    #[test]
    fn the_distribution_force_dmabuf_opt_out_outranks_the_nvidia_default() {
        // Checked inside isNVIDIA(), which WebKit never reaches once DISABLE_DMABUF is set.
        for value in ["1", "", "yes"] {
            assert_eq!(
                plan_on_nvidia(&[(FORCE_DMABUF, value)]),
                RenderingPlan::PreserveEnvironment,
                "WEBKIT_FORCE_DMABUF_RENDERER={value} must stand the workaround down"
            );
        }
    }

    #[test]
    fn force_dmabuf_does_not_defeat_the_missing_gles_fallback() {
        // That fallback answers a packaging failure, not this variable's question, and the
        // AppImage cannot render without it (#8343).
        assert_eq!(
            plan_on_host(
                &[(FORCE_DMABUF, "1"), (APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false,
                true
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON)
        );
    }

    #[test]
    fn force_dmabuf_still_leaves_the_wayland_workaround_in_place() {
        // It speaks to the NVIDIA patch, not the Wayland rule that predates it.
        assert_eq!(
            plan_on_nvidia(&[(FORCE_DMABUF, "1"), (WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn a_zero_force_dmabuf_value_does_not_stand_the_workaround_down() {
        // The patch tests the first byte against '0', so "0" is not a request.
        assert_eq!(
            plan_on_nvidia(&[(FORCE_DMABUF, "0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_on_wayland_takes_the_stronger_switch_over_shared_memory() {
        assert_eq!(
            plan_on_nvidia(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn an_operator_force_shm_value_is_preserved_on_nvidia_too() {
        // Unmarked, so it came from a launcher or environment.d rather than from us.
        for value in ["0", "1", "", "true"] {
            assert_eq!(
                plan_on_nvidia(&[(FORCE_SHARED_MEMORY, value)]),
                RenderingPlan::PreserveEnvironment,
                "operator FORCE_SHM={value} must not be overridden on NVIDIA"
            );
        }
    }

    #[test]
    fn nvidia_relaunch_re_decides_over_its_own_inherited_force_shm() {
        // Marked, so it is our own state from an earlier launch, and Wayland escalates it.
        assert_eq!(
            plan_on_nvidia(&[
                (WAYLAND_DISPLAY, "wayland-0"),
                (FORCE_SHARED_MEMORY, "1"),
                (APPLIED_WORKAROUND, FORCE_SHARED_MEMORY),
            ]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn a_relaunch_re_reaches_the_same_decision_it_made_first_time() {
        // process::restart inherits the environment, so the plan must be a fixed point:
        // feed a launch's own output back in and it must not drift.
        let hosts = [
            (&[][..], true),
            (&[(WAYLAND_DISPLAY, "wayland-0")][..], true),
            (&[(GDK_BACKEND, "x11")][..], true),
            (&[][..], false),
            (&[(WAYLAND_DISPLAY, "wayland-0")][..], false),
        ];
        for (session, nvidia) in hosts {
            let first = plan_on_host(session, MODERN_WEBKIT, true, nvidia);
            let RenderingPlan::Apply(workaround, reason) = first else {
                continue;
            };
            let mut inherited = session.to_vec();
            inherited.push((workaround.variable(), "1"));
            inherited.push((APPLIED_WORKAROUND, workaround.variable()));
            assert_eq!(
                plan_on_host(&inherited, MODERN_WEBKIT, true, nvidia),
                RenderingPlan::Apply(workaround, reason),
                "relaunch drifted for session {session:?} nvidia={nvidia}"
            );
        }
    }

    #[test]
    fn a_marker_for_the_other_variable_does_not_release_force_shm() {
        // Only a claim naming FORCE_SHM identifies FORCE_SHM as ours.
        assert_eq!(
            plan_on_nvidia(&[
                (FORCE_SHARED_MEMORY, "1"),
                (APPLIED_WORKAROUND, DISABLE_DMABUF),
            ]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn explicit_disable_override_preserves_force_shm_on_nvidia() {
        assert_eq!(
            plan_on_nvidia(&[(FORCE_SHARED_MEMORY, "1"), (DISABLE_DMABUF, "0")]),
            RenderingPlan::PreserveEnvironment
        );
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
    fn a_gdk_wildcard_can_select_wayland_without_a_display_variable() {
        assert_eq!(
            plan(&[(GDK_BACKEND, "*")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn appimage_without_usable_gles_disables_dmabuf_on_x11() {
        assert_eq!(
            plan_with_version_and_gles(
                &[(APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON)
        );
    }

    #[test]
    fn native_package_without_gles_keeps_existing_x11_behavior() {
        assert_eq!(
            plan_with_version_and_gles(&[], MODERN_WEBKIT, false),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn appimage_wayland_disables_dmabuf_completely() {
        assert_eq!(
            plan(&[
                (APPIMAGE, "/tmp/Unsloth.AppImage"),
                (WAYLAND_DISPLAY, "wayland-0")
            ]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, WAYLAND_REASON)
        );
    }

    #[test]
    fn old_webkit_uses_the_legacy_renderer_switch() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 42, 7)),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, WAYLAND_REASON)
        );
    }

    #[test]
    fn webkit_244_uses_force_shm() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 44, 0)),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
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
