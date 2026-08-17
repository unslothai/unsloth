mod backend;
mod managed;
mod pid_records;
mod types;
mod version;

use crate::desktop_backend_owner::{
    OwnedBackendProbe, OwnedBackendReadiness, VerifiedOwnedBackend,
};
use backend::probe_existing_backends;
use log::{info, warn};
pub use managed::managed_install_ready;
use managed::probe_managed_install;
use std::path::PathBuf;
use types::{BackendProbe, ManagedProbe};
pub use types::{DesktopPreflightDisposition, DesktopPreflightResult, ExternalBackendConflict};
#[cfg(test)]
pub(crate) use version::DESKTOP_MANAGEABILITY_VERSION;
pub(crate) use version::{
    backend_version_stale_reason, DESKTOP_BACKEND_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};

#[cfg(test)]
use backend::{backend_desktop_auth_status, backend_health};
#[cfg(test)]
use managed::probe_managed_bin;
#[cfg(test)]
use version::{
    backend_version_compatible, backend_version_outdated_reason, expected_backend_version,
    managed_backend_version_stale_reason, MIN_DESKTOP_BACKEND_VERSION,
};

fn release_auto_repair() -> bool {
    !cfg!(debug_assertions)
}

/// Whether the managed install sits on a profile the app cannot reach:
/// `Unavailable` if the binary was never looked up, `Stale` if it has nowhere to run.
fn managed_profile_unreachable(managed: &ManagedProbe) -> bool {
    match managed {
        ManagedProbe::Unavailable { reason } | ManagedProbe::Stale { reason, .. } => {
            // Either context failure: repair needs the same profile or path setting.
            managed::is_context_reason(reason)
        }
        ManagedProbe::Ready { .. } | ManagedProbe::Missing => false,
    }
}

/// Repair reinstalls through the managed CLI, which needs the profile the probe
/// just failed to reach, and stops a backend that still answers to do it.
fn stale_auto_repair(managed: &ManagedProbe) -> bool {
    release_auto_repair() && !managed_profile_unreachable(managed)
}

/// What to tell the user about a stale backend: the unreachable profile wins,
/// since the frontend answers every other reason with "update", which needs it.
fn stale_reason(managed: &ManagedProbe, reason: &str) -> String {
    if managed_profile_unreachable(managed) {
        info!("Desktop preflight: stale backend ({reason}) reported as an unusable managed context");
        return match managed {
            ManagedProbe::Unavailable { reason } | ManagedProbe::Stale { reason, .. } => {
                reason.clone()
            }
            _ => managed::WORKING_DIRECTORY_UNAVAILABLE.to_string(),
        };
    }
    reason.to_string()
}

fn managed_bin_for_result(managed: &ManagedProbe) -> Option<PathBuf> {
    match managed {
        ManagedProbe::Ready { bin } | ManagedProbe::Stale { bin, .. } => Some(bin.clone()),
        ManagedProbe::Missing | ManagedProbe::Unavailable { .. } => None,
    }
}

fn choose_preflight(managed: ManagedProbe, backend: BackendProbe) -> DesktopPreflightResult {
    match (backend, managed) {
        (BackendProbe::ExternalConflict { port, reason }, managed) => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::ExternalConflict,
            reason: Some(reason),
            port: Some(port),
            can_auto_repair: false,
            managed_bin: managed_bin_for_result(&managed),
        },
        (BackendProbe::Ready { port }, managed) => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::AttachedReady,
            reason: None,
            port: Some(port),
            can_auto_repair: false,
            managed_bin: managed_bin_for_result(&managed),
        },
        (_, managed) => match managed {
            ManagedProbe::Ready { bin } => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::ManagedReady,
                reason: None,
                port: None,
                can_auto_repair: false,
                managed_bin: Some(bin),
            },
            ManagedProbe::Stale { bin, reason } => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::ManagedStale,
                // The repair needs the same home directory, so do not offer it.
                can_auto_repair: release_auto_repair() && !managed::is_context_reason(&reason),
                reason: Some(reason),
                port: None,
                managed_bin: Some(bin),
            },
            ManagedProbe::Missing => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::NotInstalled,
                reason: None,
                port: None,
                can_auto_repair: false,
                managed_bin: None,
            },
            // Not "install Unsloth": the install may exist on an unmounted profile.
            ManagedProbe::Unavailable { reason } => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::ManagedStale,
                reason: Some(reason),
                port: None,
                can_auto_repair: false,
                managed_bin: None,
            },
        },
    }
}

fn owned_unmanageable_reason(reason: &str) -> String {
    format!("desktop_owned_backend_unmanageable:{reason}")
}

fn choose_owned_preflight(
    managed: &ManagedProbe,
    owned: &VerifiedOwnedBackend,
) -> DesktopPreflightResult {
    match &owned.readiness {
        OwnedBackendReadiness::Ready => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::OwnedReady,
            reason: None,
            port: Some(owned.port),
            can_auto_repair: false,
            managed_bin: managed_bin_for_result(managed),
        },
        OwnedBackendReadiness::Stale { reason } => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::OwnedStale,
            reason: Some(stale_reason(managed, reason)),
            port: Some(owned.port),
            can_auto_repair: stale_auto_repair(managed),
            managed_bin: managed_bin_for_result(managed),
        },
    }
}

fn choose_unmanageable_owned_preflight(
    managed: &ManagedProbe,
    port: u16,
    reason: String,
) -> DesktopPreflightResult {
    DesktopPreflightResult {
        disposition: DesktopPreflightDisposition::ExternalConflict,
        reason: Some(owned_unmanageable_reason(&reason)),
        port: Some(port),
        can_auto_repair: false,
        managed_bin: managed_bin_for_result(managed),
    }
}

fn choose_owned_transitional_preflight(
    managed: &ManagedProbe,
    port: Option<u16>,
) -> DesktopPreflightResult {
    DesktopPreflightResult {
        disposition: DesktopPreflightDisposition::ExternalConflict,
        reason: Some("desktop_owned_backend_starting".to_string()),
        port,
        can_auto_repair: false,
        managed_bin: managed_bin_for_result(managed),
    }
}

fn choose_ownerless_spawned_preflight(
    managed: &ManagedProbe,
    backend: &BackendProbe,
    port: Option<u16>,
) -> DesktopPreflightResult {
    match (port, backend) {
        (Some(owned_port), BackendProbe::Ready { port }) if owned_port == *port => {
            DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::OwnedReady,
                reason: None,
                port: Some(*port),
                can_auto_repair: false,
                managed_bin: managed_bin_for_result(managed),
            }
        }
        (Some(owned_port), BackendProbe::Old { port, reason }) if owned_port == *port => {
            DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::OwnedStale,
                reason: Some(stale_reason(managed, reason)),
                port: Some(*port),
                can_auto_repair: stale_auto_repair(managed),
                managed_bin: managed_bin_for_result(managed),
            }
        }
        _ => choose_owned_transitional_preflight(managed, port),
    }
}

fn mutation_blocker_from_probe(
    probe: BackendProbe,
    live_local_backend: &dyn Fn(u16) -> Option<u32>,
) -> Option<ExternalBackendConflict> {
    match probe {
        BackendProbe::ExternalConflict { port, reason } => {
            Some(ExternalBackendConflict { port, reason })
        }
        // An id-less backend may be this install serving from a terminal, in
        // which case rewriting the venv underneath it would break it. It may
        // equally be a remote Studio behind a port forward, and refusing on
        // that leaves a stale install with no way to repair itself, since
        // repair is what the app runs automatically. The local per-port record
        // is what tells the two apart, so it, not the port, decides.
        BackendProbe::Unrelated { port, reason } => match live_local_backend(port) {
            Some(pid) => {
                info!("Desktop preflight: mutation blocked by local backend {pid} on port {port}");
                Some(ExternalBackendConflict { port, reason })
            }
            None => {
                info!("Desktop preflight: mutation may proceed past port {port} ({reason}): no live backend of this install is recorded there");
                None
            }
        },
        BackendProbe::Ready { port } => Some(ExternalBackendConflict {
            port,
            reason: "same_root_external_backend_active".to_string(),
        }),
        _ => None,
    }
}

pub async fn mutation_blocking_backend_ignoring(
    ignored_ports: &[u16],
) -> Option<ExternalBackendConflict> {
    backend::probe_backend_ports(ignored_ports)
        .await
        .into_iter()
        .find_map(|probe| {
            mutation_blocker_from_probe(probe, &pid_records::live_backend_pid_on_port)
        })
}

pub async fn desktop_preflight_result() -> DesktopPreflightResult {
    let (managed, backend) = tokio::join!(probe_managed_install(), probe_existing_backends(&[]));
    choose_preflight(managed, backend)
}

pub async fn desktop_preflight_result_with_state(
    state: &crate::process::BackendState,
) -> Result<(DesktopPreflightResult, Option<(u64, bool)>), String> {
    let (managed, backend, owned) = tokio::join!(
        probe_managed_install(),
        probe_existing_backends(&[]),
        crate::desktop_backend_owner::probe_verified_owned_backend()
    );

    if let Some(snapshot) = crate::process::owned_backend_snapshot(state)? {
        let Some(owner) = snapshot.owner.clone() else {
            // Defensive fallback for legacy ownerless handles. Wait for full
            // health so auth and bootstrap are ready.

            let probe = match snapshot.port {
                Some(port) => backend::probe_ownerless_spawned_backend(port).await,
                None => backend,
            };
            return Ok((
                choose_ownerless_spawned_preflight(&managed, &probe, snapshot.port),
                None,
            ));
        };
        match crate::desktop_backend_owner::probe_owned_backend_state(
            owner,
            snapshot.port,
            snapshot.is_adopted,
        )
        .await
        {
            OwnedBackendProbe::Verified(verified) => {
                if snapshot.port.is_none() {
                    crate::process::record_owned_backend_port_if_current(
                        state,
                        snapshot.generation,
                        verified.port,
                    );
                }
                let result = choose_owned_preflight(&managed, &verified);
                let watchdog_generation = if snapshot.is_adopted
                    && result.disposition == DesktopPreflightDisposition::OwnedReady
                {
                    Some((snapshot.generation, false))
                } else {
                    None
                };
                return Ok((result, watchdog_generation));
            }
            OwnedBackendProbe::Unmanageable { port, reason } => {
                return Ok((
                    choose_unmanageable_owned_preflight(&managed, port, reason),
                    None,
                ));
            }
            OwnedBackendProbe::NoMetadata
            | OwnedBackendProbe::RemovedMalformed
            | OwnedBackendProbe::NotVerified { .. } => {
                if snapshot.is_adopted {
                    crate::process::clear_adopted_backend_if_current(
                        state,
                        snapshot.generation,
                        snapshot.port,
                        "state owner probe no longer verifies",
                    );
                    return Ok((choose_preflight(managed, backend), None));
                }
                return Ok((
                    choose_owned_transitional_preflight(&managed, snapshot.port),
                    None,
                ));
            }
        }
    }

    let owned = match owned {
        Ok(owned) => owned,
        Err(error) => {
            warn!(
                "Desktop-owned backend probe failed; continuing without adoption: {}",
                error
            );
            return Ok((choose_preflight(managed, backend), None));
        }
    };

    match owned {
        OwnedBackendProbe::Verified(verified) => {
            let result = choose_owned_preflight(&managed, &verified);
            let adopted = crate::process::adopt_verified_backend(state, verified)?;
            let watchdog_generation =
                if result.disposition == DesktopPreflightDisposition::OwnedReady {
                    Some((adopted.generation, adopted.newly_adopted))
                } else {
                    None
                };
            Ok((result, watchdog_generation))
        }
        OwnedBackendProbe::Unmanageable { port, reason } => Ok((
            choose_unmanageable_owned_preflight(&managed, port, reason),
            None,
        )),
        OwnedBackendProbe::NoMetadata
        | OwnedBackendProbe::RemovedMalformed
        | OwnedBackendProbe::NotVerified { .. } => Ok((choose_preflight(managed, backend), None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn choose_preflight_classifies_core_cases() {
        let bin = || PathBuf::from("/managed/unsloth");
        let ready = || ManagedProbe::Ready { bin: bin() };
        let stale = || ManagedProbe::Stale {
            bin: bin(),
            reason: "old cli".to_string(),
        };
        let old_backend = || BackendProbe::Old {
            port: 8001,
            reason: "missing endpoint".to_string(),
        };
        let cases = [
            (
                stale(),
                BackendProbe::Ready { port: 8000 },
                DesktopPreflightDisposition::AttachedReady,
                Some(8000),
                None,
                false,
                Some(bin()),
            ),
            (
                ready(),
                BackendProbe::Ready { port: 8000 },
                DesktopPreflightDisposition::AttachedReady,
                Some(8000),
                None,
                false,
                Some(bin()),
            ),
            (
                ManagedProbe::Missing,
                BackendProbe::Ready { port: 8000 },
                DesktopPreflightDisposition::AttachedReady,
                Some(8000),
                None,
                false,
                None,
            ),
            (
                ready(),
                old_backend(),
                DesktopPreflightDisposition::ManagedReady,
                None,
                None,
                false,
                Some(bin()),
            ),
            (
                stale(),
                old_backend(),
                DesktopPreflightDisposition::ManagedStale,
                None,
                Some("old cli"),
                release_auto_repair(),
                Some(bin()),
            ),
            (
                ready(),
                BackendProbe::Missing,
                DesktopPreflightDisposition::ManagedReady,
                None,
                None,
                false,
                Some(bin()),
            ),
            (
                stale(),
                BackendProbe::Missing,
                DesktopPreflightDisposition::ManagedStale,
                None,
                Some("old cli"),
                release_auto_repair(),
                Some(bin()),
            ),
            (
                ManagedProbe::Missing,
                BackendProbe::Missing,
                DesktopPreflightDisposition::NotInstalled,
                None,
                None,
                false,
                None,
            ),
            // An unreachable profile is not a missing install.
            (
                ManagedProbe::Unavailable {
                    reason: managed::WORKING_DIRECTORY_UNAVAILABLE.to_string(),
                },
                BackendProbe::Missing,
                DesktopPreflightDisposition::ManagedStale,
                None,
                Some(managed::WORKING_DIRECTORY_UNAVAILABLE),
                false,
                None,
            ),
        ];

        for (managed, backend, disposition, port, reason, can_auto_repair, managed_bin) in cases {
            let result = choose_preflight(managed, backend);
            assert_eq!(result.disposition, disposition);
            assert_eq!(result.port, port);
            assert_eq!(result.reason.as_deref(), reason);
            assert_eq!(result.can_auto_repair, can_auto_repair);
            assert_eq!(result.managed_bin, managed_bin);
        }
    }

    #[test]
    fn repair_is_withheld_from_stale_backends_on_an_unreachable_profile() {
        let bin = PathBuf::from("/managed/unsloth");
        let unreachable = [
            ManagedProbe::Unavailable {
                reason: managed::WORKING_DIRECTORY_UNAVAILABLE.to_string(),
            },
            ManagedProbe::Stale {
                bin: bin.clone(),
                reason: managed::WORKING_DIRECTORY_UNAVAILABLE.to_string(),
            },
        ];
        let reachable = [
            ManagedProbe::Ready { bin: bin.clone() },
            ManagedProbe::Missing,
            ManagedProbe::Stale {
                bin: bin.clone(),
                reason: "cli_unusable".to_string(),
            },
        ];

        let owned = |managed: &ManagedProbe| {
            choose_owned_preflight(
                managed,
                &VerifiedOwnedBackend {
                    owner: crate::desktop_backend_owner::test_owner_state("root", "token", 8000),
                    port: 8000,
                    backend_pid: 2,
                    generation: 3,
                    readiness: OwnedBackendReadiness::Stale {
                        reason: "backend_outdated".to_string(),
                    },
                },
            )
        };
        let ownerless = |managed: &ManagedProbe| {
            choose_ownerless_spawned_preflight(
                managed,
                &BackendProbe::Old {
                    port: 8000,
                    reason: "backend_outdated".to_string(),
                },
                Some(8000),
            )
        };

        for managed in &unreachable {
            assert!(managed_profile_unreachable(managed), "{managed:?}");
            assert!(!stale_auto_repair(managed), "{managed:?}");
            // Repair would stop a working backend for an install that cannot run.
            for result in [owned(managed), ownerless(managed)] {
                assert_eq!(result.disposition, DesktopPreflightDisposition::OwnedStale);
                assert_eq!(result.port, Some(8000));
                assert!(!result.can_auto_repair, "{managed:?}");
                // The frontend answers "backend_outdated" with "update", which needs
                // the profile.
                assert_eq!(
                    result.reason.as_deref(),
                    Some(managed::WORKING_DIRECTORY_UNAVAILABLE),
                    "{managed:?}"
                );
            }
        }

        for managed in &reachable {
            assert!(!managed_profile_unreachable(managed), "{managed:?}");
            assert_eq!(stale_auto_repair(managed), release_auto_repair());
            for result in [owned(managed), ownerless(managed)] {
                assert_eq!(result.can_auto_repair, release_auto_repair(), "{managed:?}");
                assert_eq!(
                    result.reason.as_deref(),
                    Some("backend_outdated"),
                    "{managed:?}"
                );
            }
        }
    }

    #[test]
    fn external_conflict_blocks_managed_flow() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::ExternalConflict {
                port: 8888,
                reason: "same_root_external_backend_active".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ExternalConflict
        );
        assert_eq!(result.port, Some(8888));
        assert_eq!(
            result.reason,
            Some("same_root_external_backend_active".to_string())
        );
        assert!(!result.can_auto_repair);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn mutation_blocker_blocks_ready_external_backends() {
        assert_eq!(
            mutation_blocker_from_probe(BackendProbe::Ready { port: 8890 }, &|_| None),
            Some(ExternalBackendConflict {
                port: 8890,
                reason: "same_root_external_backend_active".to_string(),
            })
        );
    }

    #[test]
    fn backend_version_gate_classifies_core_cases() {
        for version in [
            MIN_DESKTOP_BACKEND_VERSION,
            "2026.8.5",
            "2027.1.0",
            "2026.8.4.post1",
            "2026.8.4+local",
        ] {
            assert!(backend_version_compatible(Some(version)), "{version}");
        }
        for (version, reason) in [
            (None, "desktop_backend_version_missing"),
            (Some("not-a-version"), "desktop_backend_version_invalid"),
            (Some("2026.8.4.1"), "desktop_backend_version_invalid"),
            (Some("2026.8.4foo"), "desktop_backend_version_invalid"),
            (Some("2026.8.4.devx"), "desktop_backend_version_invalid"),
            (Some("2026.5.2"), "desktop_backend_version_too_old"),
            (Some("2026.8.4rc1"), "desktop_backend_version_too_old"),
            (Some("2026.8.4.dev1"), "desktop_backend_version_too_old"),
        ] {
            assert_eq!(
                backend_version_stale_reason(version).as_deref(),
                Some(reason)
            );
        }
        assert_eq!(
            backend_version_compatible(Some("dev")),
            cfg!(debug_assertions)
        );
    }

    #[test]
    fn managed_venv_behind_the_shipped_backend_is_outdated() {
        // Above the floor but below what this build shipped: the exact case the
        // standalone installer leaves behind in the shared venv.
        assert_eq!(
            backend_version_outdated_reason(Some("2026.8.4"), "2026.8.5").as_deref(),
            Some("desktop_backend_version_outdated")
        );
        for version in ["2026.8.5", "2026.8.6", "2027.1.0"] {
            assert_eq!(
                backend_version_outdated_reason(Some(version), "2026.8.5"),
                None,
                "{version}"
            );
        }
        // The floor still speaks first, so its reasons keep reaching the UI.
        for (version, reason) in [
            (None, "desktop_backend_version_missing"),
            (Some("not-a-version"), "desktop_backend_version_invalid"),
            (Some("2026.5.2"), "desktop_backend_version_too_old"),
        ] {
            assert_eq!(
                backend_version_outdated_reason(version, "2026.8.5").as_deref(),
                Some(reason)
            );
        }
        // Unstamped builds fall back to the floor, so the managed gate reduces
        // to the shared one for dev and CI.
        assert_eq!(expected_backend_version(), MIN_DESKTOP_BACKEND_VERSION);
        assert_eq!(
            managed_backend_version_stale_reason(Some(MIN_DESKTOP_BACKEND_VERSION)),
            None
        );
        assert_eq!(
            managed_backend_version_stale_reason(Some("2026.5.2")).as_deref(),
            Some("desktop_backend_version_too_old")
        );
    }

    #[cfg(unix)]
    struct FakeCli {
        bin: PathBuf,
        dir: PathBuf,
    }

    #[cfg(unix)]
    impl Drop for FakeCli {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[cfg(unix)]
    fn fake_cli(test_name: &str, script: &str) -> FakeCli {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-preflight-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("unsloth");
        fs::write(&bin, script).unwrap();
        let mut perms = fs::metadata(&bin).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).unwrap();
        FakeCli { bin, dir }
    }

    #[cfg(unix)]
    static MANAGED_CAPABILITY_CACHE_TEST_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

    #[cfg(unix)]
    struct ManagedCapabilityCacheHome {
        path: PathBuf,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(unix)]
    impl ManagedCapabilityCacheHome {
        fn new(test_name: &str) -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};

            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "unsloth-preflight-cache-{test_name}-{}-{nanos}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            let previous = std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME");
            std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", &path);
            Self { path, previous }
        }
    }

    #[cfg(unix)]
    impl Drop for ManagedCapabilityCacheHome {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", previous);
            } else {
                std::env::remove_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME");
            }
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[cfg(unix)]
    fn managed_capability_cache_path_for_test() -> PathBuf {
        std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(dirs::home_dir)
            .unwrap()
            .join(".unsloth")
            .join("studio")
            .join("desktop_capability_cache.json")
    }

    #[cfg(unix)]
    fn remove_managed_capability_cache() {
        let _ = std::fs::remove_file(managed_capability_cache_path_for_test());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_capability_probe_classifies_core_cases() {
        let _cache_guard = MANAGED_CAPABILITY_CACHE_TEST_LOCK.lock().await;
        let _cache_home = ManagedCapabilityCacheHome::new("core-cases");
        remove_managed_capability_cache();

        for (name, script, stale_reason) in [
            (
                "cap-missing",
                r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
                Some("desktop_capability_probe_failed"),
            ),
            (
                "cap-true-helper-missing",
                r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":2,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_install_ok":true,"version":"2026.8.4"}'
  exit 0
fi
exit 1
"#,
                None,
            ),
            (
                "cap-false-helper-ready",
                r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":2,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","studio_install_ok":true,"version":"2026.8.4"}'
  exit 0
fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
                Some("cap_false"),
            ),
        ] {
            let fake = fake_cli(name, script);
            let bin = fake.bin.clone();
            match (probe_managed_bin(bin.clone()).await, stale_reason) {
                (ManagedProbe::Ready { bin: actual }, None) => assert_eq!(actual, bin),
                (
                    ManagedProbe::Stale {
                        bin: actual,
                        reason,
                    },
                    Some(expected),
                ) => {
                    assert_eq!((actual, reason.as_str()), (bin, expected));
                }
                (probe, expected) => panic!("unexpected probe {probe:?}, expected {expected:?}"),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_capability_help_probe_runs_before_cache() {
        use std::fs;

        let _cache_guard = MANAGED_CAPABILITY_CACHE_TEST_LOCK.lock().await;
        let _cache_home = ManagedCapabilityCacheHome::new("cache-hit");

        remove_managed_capability_cache();
        // `-h` always succeeds unless `modeh` exists; the desktop-capabilities
        // probe always succeeds unless `modecap` exists. Toggling those lets us
        // prove the ordering: -h runs on every probe (even a cache hit), while
        // the heavier capability probe is skipped once the cache is warm.
        let fake = fake_cli(
            "cap-cache-hit",
            r#"#!/bin/sh
log="$0.calls"
modeh="$0.modeh"
modecap="$0.modecap"
printf '%s\n' "$*" >> "$log"
if [ "$1" = "-h" ]; then
  if [ -f "$modeh" ]; then exit 42; fi
  exit 0
fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  if [ -f "$modecap" ]; then exit 42; fi
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":2,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_install_ok":true,"version":"2026.8.4"}'
  exit 0
fi
exit 1
"#,
        );
        let bin = fake.bin.clone();
        let calls = bin.with_extension("calls");
        let modeh = bin.with_extension("modeh");
        let modecap = bin.with_extension("modecap");

        // Cold probe: runs -h and the capability probe, then caches the result.
        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Ready { .. }
        ));
        let first_calls = fs::read_to_string(&calls).unwrap();
        assert!(first_calls.contains("-h"));
        assert!(first_calls.contains("studio desktop-capabilities --json"));

        // Cache hit: -h still runs, but the capability probe is skipped (breaking
        // it via `modecap` proves it is not invoked).
        fs::write(&modecap, "broken").unwrap();
        fs::write(&calls, "").unwrap();
        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Ready { .. }
        ));
        assert_eq!(fs::read_to_string(&calls).unwrap(), "-h\n");

        // A non-launchable CLI is caught by the -h probe even with a warm cache:
        // preflight reports Stale (for repair) and never trusts the cache.
        fs::write(&modeh, "broken").unwrap();
        fs::write(&calls, "").unwrap();
        assert!(matches!(
            probe_managed_bin(bin).await,
            ManagedProbe::Stale { .. }
        ));
        assert_eq!(fs::read_to_string(&calls).unwrap(), "-h\n");

        remove_managed_capability_cache();
    }

    const EXPECTED_ROOT_ID: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_ROOT_ID: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const OWNER_TOKEN: &str = "desktop-owner-token";

    fn install_test_owner() {
        crate::desktop_backend_owner::install_test_owner(EXPECTED_ROOT_ID, OWNER_TOKEN);
    }

    fn desktop_ready_health(root_id: &str) -> String {
        desktop_ready_health_with_owner(root_id, true)
    }

    fn desktop_owner_json(include_owner: bool) -> String {
        if include_owner {
            format!(
                r#", "desktop_owner":{{"kind":"tauri","token_sha256":"{}"}}"#,
                crate::desktop_backend_owner::token_sha256(OWNER_TOKEN)
            )
        } else {
            String::new()
        }
    }

    fn desktop_ready_health_with_owner(root_id: &str, include_owner: bool) -> String {
        let owner = desktop_owner_json(include_owner);
        // Tied to the owner on purpose: the secret comes from the desktop spawn,
        // so an ownerless (terminal-started) backend can never report it. Both at
        // once describes a backend that cannot exist.
        let leases = if include_owner {
            r#""native_path_leases_supported":true,"#
        } else {
            ""
        };
        format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,{leases}"studio_root_id":"{root_id}"{owner}}}"#
        )
    }

    async fn backend_server(health_body: impl Into<String>, route_status: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let health_body = health_body.into();

        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0; 2048];
                let n = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..n]);
                let (status, body) = if request.starts_with("GET /api/health ") {
                    ("200 OK", health_body.as_str())
                } else if request.starts_with("POST /api/auth/desktop-login ") {
                    (route_status, "")
                } else {
                    ("404 Not Found", "")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        port
    }

    async fn probe_test_backend(
        health_body: impl Into<String>,
        route_status: &'static str,
    ) -> BackendProbe {
        install_test_owner();
        let port = backend_server(health_body, route_status).await;
        let client = crate::loopback_http::client(std::time::Duration::from_secs(2)).unwrap();
        let health = backend_health(&client, port).await.unwrap();
        backend_desktop_auth_status(&client, port, &health, Some(EXPECTED_ROOT_ID)).await
    }

    #[tokio::test]
    async fn backend_health_without_desktop_capability_fields_is_still_candidate() {
        let port = backend_server(
            r#"{"status":"healthy","service":"Unsloth UI Backend"}"#,
            "401 Unauthorized",
        )
        .await;
        let client = crate::loopback_http::client(std::time::Duration::from_secs(2)).unwrap();

        assert!(backend_health(&client, port).await.is_some());
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_missing_protocol_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_health_with_desktop_capability_fields_and_401_is_ready() {
        let probe =
            probe_test_backend(desktop_ready_health(EXPECTED_ROOT_ID), "401 Unauthorized").await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn legacy_manageability_same_root_backend_is_still_ready() {
        // Same migration window as the owned-backend case: a server from the
        // release before the CLI gained studio_install_ok reports manageability
        // 1. That capability is CLI-side, so it must not turn a live,
        // protocol-compatible backend into a conflict the user has to kill.
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"native_path_leases_supported":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn backend_without_any_manageability_field_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::Old { reason, .. } if reason == "desktop_manageability_unsupported"
        ));
    }

    #[tokio::test]
    async fn compatible_same_root_without_desktop_owner_is_ready() {
        let probe = probe_test_backend(
            desktop_ready_health_with_owner(EXPECTED_ROOT_ID, false),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn terminal_started_backend_without_native_path_leases_stays_adoptable() {
        // What EVERY terminal-started server looks like, not an edge case.
        // Refusing it would drop the attach use-tauri-backend.ts supports on
        // purpose, to grey out one button that use-linked-folders.ts already
        // greys out from the same capability.
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"native_path_leases_supported":false,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn backend_predating_the_lease_capability_field_stays_adoptable() {
        // Absent, not false: a backend predating the field reports nothing, and
        // Option<bool> makes that indistinguishable from `false` untested.
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn owned_backend_without_native_path_leases_is_stale_not_a_conflict() {
        // A defect in our own spawn, and ours to restart, so Stale (repairable)
        // rather than a conflict the user has to resolve by hand.
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"native_path_leases_supported":false,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::Old { reason, .. } if reason == "native_path_leases_unsupported"
        ));
    }

    #[tokio::test]
    async fn a_stale_version_is_still_reported_as_a_version_problem() {
        // Ordering guard: the lease check used to run first, so a backend that
        // really needed an update reported the wrong cause to diagnostics.
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.1","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"native_path_leases_supported":false,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::ExternalConflict { reason, .. }
                if reason == "desktop_backend_version_too_old"
        ));
    }

    #[tokio::test]
    async fn stale_same_root_without_desktop_owner_is_external_conflict() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.1","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "desktop_backend_version_too_old"
        ));
    }

    #[tokio::test]
    async fn backend_root_id_mismatch_is_old_before_auth_probe() {
        let probe =
            probe_test_backend(desktop_ready_health(OTHER_ROOT_ID), "401 Unauthorized").await;

        assert!(matches!(
            probe,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "studio_root_id_mismatch"
        ));
    }

    #[tokio::test]
    async fn backend_missing_root_id_is_unrelated_before_auth_probe() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::Unrelated {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
    }

    #[tokio::test]
    async fn backend_expected_root_id_missing_is_unrelated_before_auth_probe() {
        install_test_owner();
        let port = backend_server(desktop_ready_health(EXPECTED_ROOT_ID), "401 Unauthorized").await;
        let client = crate::loopback_http::client(std::time::Duration::from_secs(2)).unwrap();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health, None).await,
            BackendProbe::Unrelated {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
    }

    /// The report this came from: an id-less Studio answered on a candidate
    /// port, and a perfectly healthy install refused to launch at all.
    #[test]
    fn an_unrelated_backend_does_not_block_a_launch() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/bin/unsloth"),
            },
            BackendProbe::Unrelated {
                port: 8888,
                reason: "ambiguous_root_external_backend_active".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedReady
        );
        assert_eq!(result.port, None);
    }

    /// ...and a venv rewrite is still refused while a local backend of this
    /// install is recorded on the port, because it would break that backend.
    #[test]
    fn an_unrelated_backend_blocks_mutations_when_it_is_recorded_locally() {
        assert_eq!(
            mutation_blocker_from_probe(
                BackendProbe::Unrelated {
                    port: 8899,
                    reason: "ambiguous_root_external_backend_active".to_string(),
                },
                &|port| (port == 8899).then_some(4242)
            ),
            Some(ExternalBackendConflict {
                port: 8899,
                reason: "ambiguous_root_external_backend_active".to_string(),
            })
        );
    }

    /// The follow-up report: a stale install auto-runs a repair, and an id-less
    /// backend reached over a port forward left it erroring on every attempt
    /// with nothing the user could stop locally.
    #[test]
    fn an_unrecorded_unrelated_backend_does_not_block_a_repair() {
        assert_eq!(
            mutation_blocker_from_probe(
                BackendProbe::Unrelated {
                    port: 8888,
                    reason: "ambiguous_root_external_backend_active".to_string(),
                },
                &|_| None
            ),
            None
        );
    }

    /// A local record is only consulted for the unattributable case: a backend
    /// that identified itself as a conflict blocks either way.
    #[test]
    fn an_external_conflict_blocks_mutations_without_a_local_record() {
        assert_eq!(
            mutation_blocker_from_probe(
                BackendProbe::ExternalConflict {
                    port: 8890,
                    reason: "desktop_backend_version_too_old".to_string(),
                },
                &|_| None
            ),
            Some(ExternalBackendConflict {
                port: 8890,
                reason: "desktop_backend_version_too_old".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn backend_route_404_is_old() {
        let probe =
            probe_test_backend(desktop_ready_health(EXPECTED_ROOT_ID), "404 Not Found").await;

        assert!(matches!(
            probe,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "desktop_login_not_found"
        ));
    }

    #[tokio::test]
    async fn backend_capability_false_is_old_even_when_route_401() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":2,"supports_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "cap_false"
        ));
    }
}
