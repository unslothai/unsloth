// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::{OsStr, OsString};

const APPIMAGE: &str = "APPIMAGE";
const GLES_V2: &[u8] = b"libGLESv2.so.2\0";

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
// disable-nvidia-dmabuf.patch reads this first inside isNVIDIA() and stands the
// distribution workaround down for it, so on Debian and Ubuntu it is the only way
// a user can ask for the hardware transport back. WebKit reads DISABLE_DMABUF
// before it ever calls isNVIDIA(), so anything set here outranks it silently.
const FORCE_DMABUF: &str = "WEBKIT_FORCE_DMABUF_RENDERER";
// Written next to the workaround so a relaunch can tell state this application
// wrote from a value the operator set. Tauri's process::restart spawns without
// env_clear, so the replacement inherits both. WebKit never reads this.
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

/// Match `forceDMABuf && *forceDMABuf != '0'` from disable-nvidia-dmabuf.patch:
/// the first byte only, so an empty value asks for the hardware transport and
/// `0` does not.
fn force_dmabuf_requested(value: &OsStr) -> bool {
    value.as_encoded_bytes().first() != Some(&b'0')
}

fn rendering_plan(
    env: impl Fn(&str) -> Option<OsString>,
    webkit_version: (u32, u32, u32),
    gles_usable: bool,
    nvidia_driver_loaded: bool,
) -> RenderingPlan {
    // Either renderer variable is an operator override when it is present, `=0` and an
    // empty value included, unless the marker says this application wrote it for an
    // earlier launch of the same process tree. Tauri's process::restart inherits the
    // environment, so without the marker a launch reads its own output back as an
    // instruction and the first decision is pinned for the life of the process tree,
    // even after the display server or the library under it changes.
    //
    // DISABLE_DMABUF=0 is how a host the probe below over-triggers on asks for its
    // accelerated path back: WebKit reads it as unset (`g_strcmp0(v, "0")`), and on a
    // patched distribution the library's own GL_VENDOR probe then declines on an
    // iGPU-presenting host, so the hardware transport really is restored. An empty
    // value is not that request; WebKit reads it as set.
    let applied_by_this_app = env(APPLIED_WORKAROUND);
    let ours = |variable: &str| applied_by_this_app.as_deref() == Some(OsStr::new(variable));

    if env(DISABLE_DMABUF).is_some() && !ours(DISABLE_DMABUF) {
        return RenderingPlan::PreserveEnvironment;
    }

    // The distribution patch's own opt-out. It has to be read here, because WebKit
    // reads DISABLE_DMABUF before isNVIDIA() and so would never reach it once a
    // workaround is applied below. It speaks only to the NVIDIA question the patch
    // asks, so it stands the NVIDIA branch down and nothing else; the missing-GLES
    // fallback below is a packaging failure, not a GPU policy, and an operator asking
    // for the hardware transport back is not asking for a launch that cannot render.
    let force_dmabuf_requested_by_operator = env(FORCE_DMABUF)
        .as_deref()
        .is_some_and(force_dmabuf_requested);

    // Same rule for the shared-memory switch: do not combine it with legacy
    // instructions a user may already carry in a launcher or environment.d file.
    if env(FORCE_SHARED_MEMORY).is_some() && !ours(FORCE_SHARED_MEMORY) {
        return RenderingPlan::PreserveEnvironment;
    }

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

    // The DMA-BUF transport breaks on the proprietary NVIDIA driver on either display
    // server, so this cannot be gated on a Wayland session. Upstream declined the fix
    // (bug 262607 is WONTFIX, WebKit PR 18614 closed unmerged), so Debian and Ubuntu
    // carry disable-nvidia-dmabuf.patch while Fedora, Arch and the upstream tarballs do
    // not. Where it is present it already empties the transport set on NVIDIA by itself:
    // isNVIDIA() sits before mode.add(SharedMemory) in 2.50.4 and 2.52.6, and 2.53.90
    // moved it after, leaving {SharedMemory}.
    //
    // Both shipped artifacts run a patched library, though: the .deb resolves the host
    // one and the AppImage bundles Ubuntu 22.04's. So on those the branch below matters
    // where the library's GL_VENDOR probe and this module probe disagree, and it also
    // skips the GBM device open and throwaway GL context isNVIDIA() performs at startup.
    // Building against an unpatched system library is the case the branch covers outright.
    //
    // The two switches are not interchangeable. DISABLE_DMABUF is read before the
    // SharedMemory add, so the set stays empty, checkRequirements() is false,
    // AcceleratedBackingStore::create() returns nullptr, and
    // webkitWebViewBaseEnterAcceleratedCompositingMode() dereferences that behind an
    // ASSERT that release builds compile out. FORCE_SHM is read after the add, leaves
    // {SharedMemory}, and keeps a valid backing store and accelerated compositing.
    //
    // So the choice is per failure mode rather than per GPU:
    //   Wayland   DISABLE_DMABUF. The failure there is the explicit-sync protocol
    //             disconnect, and FORCE_SHM routes every commit down the wl_shm path
    //             that trips it (WebKit bug 315436). It is also the switch with
    //             reports of fixing Error 71.
    //   X11       FORCE_SHM where the runtime supports it. X11 has no explicit-sync
    //             protocol, the failure there is hardware DMA-BUF allocation, and
    //             dropping to shared memory fixes that without the empty set.
    // The empty set is not merely slower on hosts where the library's own probe does
    // not fire: block/buzz#3654 reports a SIGSEGV on NVIDIA X11 with DISABLE_DMABUF
    // that FORCE_SHM does not reproduce, on exactly the iGPU-presenting topology the
    // probe below over-triggers on.
    //
    // The probe is kernel-module presence, not the GPU that will render. A PRIME or
    // Optimus laptop presenting on the integrated GPU reads as NVIDIA here and takes a
    // workaround it does not need. That is deliberate: reading the rendering GPU needs a
    // GL context, and this runs before GTK is initialized precisely so that no GL state
    // exists yet. Those hosts opt out with WEBKIT_DISABLE_DMABUF_RENDERER=0.
    if nvidia_driver_loaded && !force_dmabuf_requested_by_operator {
        let missing_appimage_gles = is_appimage && !gles_usable;
        let reason = if missing_appimage_gles {
            NVIDIA_APPIMAGE_GLES_REASON
        } else if wayland_session {
            NVIDIA_WAYLAND_REASON
        } else {
            NVIDIA_REASON
        };
        // FORCE_SHM still reaches the failing libepoxy path in AppImages, and WebKitGTK
        // before 2.44 ignores it outright, so both keep the stronger switch.
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
            // A relaunch that re-decides the other way has inherited the previous
            // choice. WebKit reads DISABLE_DMABUF before FORCE_SHM, so leaving the old
            // one set would silently outrank the new one. Only a variable this
            // application claimed is cleared; an operator's value never reaches here.
            if let Some(previous) = std::env::var_os(APPLIED_WORKAROUND) {
                if previous != OsStr::new(variable) {
                    std::env::remove_var(previous);
                }
            }
            std::env::set_var(variable, "1");
            // Claim it, so the next launch re-decides instead of reading it back as an
            // operator override.
            std::env::set_var(APPLIED_WORKAROUND, variable);
            Some((variable, reason))
        }
        RenderingPlan::PreserveEnvironment => {
            // Nothing here is ours any more; a stale claim would suppress an operator
            // override on the next relaunch.
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
        // X11 has no explicit-sync protocol to violate, and DISABLE_DMABUF leaves a null
        // backing store that release builds dereference (block/buzz#3654).
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
        // disable-nvidia-dmabuf.patch checks this inside isNVIDIA(), which WebKit never
        // reaches once DISABLE_DMABUF is set, so it has to be honoured here instead.
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
        // That fallback answers a packaging failure, not the NVIDIA question this
        // variable asks. Without it the AppImage cannot render at all (#8343).
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
        // The variable speaks to the NVIDIA patch, not to the Wayland transport rule
        // that predates it.
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
        // Marked, so it is this application's own state from an earlier launch. On
        // Wayland the re-decision escalates it rather than inheriting it.
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
        // Tauri's process::restart inherits the environment, so the plan has to be a
        // fixed point: feed a launch's own output back in and it must not drift.
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
