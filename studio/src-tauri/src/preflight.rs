mod backend;
mod managed;
mod types;
mod version;

use crate::desktop_backend_owner::{
    OwnedBackendProbe, OwnedBackendReadiness, VerifiedOwnedBackend,
};
use backend::probe_existing_backends;
use log::warn;
pub use managed::managed_install_ready;
use managed::probe_managed_install;
use std::path::PathBuf;
use types::{BackendProbe, ManagedProbe};
pub use types::{DesktopPreflightDisposition, DesktopPreflightResult, ExternalBackendConflict};
pub(crate) use version::{
    backend_version_stale_reason, DESKTOP_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};

#[cfg(test)]
use backend::{backend_desktop_auth_status, backend_health};
#[cfg(test)]
use managed::probe_managed_bin;
#[cfg(test)]
use version::{backend_version_compatible, MIN_DESKTOP_BACKEND_VERSION};

fn release_auto_repair() -> bool {
    !cfg!(debug_assertions)
}

fn managed_bin_for_result(managed: &ManagedProbe) -> Option<PathBuf> {
    match managed {
        ManagedProbe::Ready { bin } | ManagedProbe::Stale { bin, .. } => Some(bin.clone()),
        ManagedProbe::Missing => None,
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
                reason: Some(reason),
                port: None,
                can_auto_repair: release_auto_repair(),
                managed_bin: Some(bin),
            },
            ManagedProbe::Missing => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::NotInstalled,
                reason: None,
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
            reason: Some(reason.clone()),
            port: Some(owned.port),
            can_auto_repair: release_auto_repair(),
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
                reason: Some(reason.clone()),
                port: Some(*port),
                can_auto_repair: release_auto_repair(),
                managed_bin: managed_bin_for_result(managed),
            }
        }
        _ => choose_owned_transitional_preflight(managed, port),
    }
}

fn mutation_blocker_from_probe(probe: BackendProbe) -> Option<ExternalBackendConflict> {
    match probe {
        BackendProbe::ExternalConflict { port, reason } => {
            Some(ExternalBackendConflict { port, reason })
        }
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
    mutation_blocker_from_probe(probe_existing_backends(ignored_ports).await)
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
            mutation_blocker_from_probe(BackendProbe::Ready { port: 8890 }),
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
            "2026.5.4",
            "2027.1.0",
            "2026.5.3.post1",
            "2026.5.3+local",
            "2026.5.3.post1",
        ] {
            assert!(backend_version_compatible(Some(version)), "{version}");
        }
        for (version, reason) in [
            (None, "desktop_backend_version_missing"),
            (Some("not-a-version"), "desktop_backend_version_invalid"),
            (Some("2026.5.3.1"), "desktop_backend_version_invalid"),
            (Some("2026.5.3foo"), "desktop_backend_version_invalid"),
            (Some("2026.5.3.devx"), "desktop_backend_version_invalid"),
            (Some("2026.5.2"), "desktop_backend_version_too_old"),
            (Some("2026.5.3rc1"), "desktop_backend_version_too_old"),
            (Some("2026.5.3.dev1"), "desktop_backend_version_too_old"),
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
    #[tokio::test]
    async fn managed_cli_capability_probe_classifies_core_cases() {
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
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"version":"2026.5.3"}'
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
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","version":"2026.5.3"}'
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
        format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.3","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{root_id}"{owner}}}"#
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
        let client = reqwest::Client::new();
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
        let client = reqwest::Client::new();

        assert!(backend_health(&client, port).await.is_some());
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_missing_protocol_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.3","desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
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
    async fn compatible_same_root_without_desktop_owner_is_ready() {
        let probe = probe_test_backend(
            desktop_ready_health_with_owner(EXPECTED_ROOT_ID, false),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn stale_same_root_without_desktop_owner_is_external_conflict() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.1","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
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
    async fn backend_missing_root_id_is_external_conflict_before_auth_probe() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
    }

    #[tokio::test]
    async fn backend_expected_root_id_missing_is_external_conflict_before_auth_probe() {
        install_test_owner();
        let port = backend_server(desktop_ready_health(EXPECTED_ROOT_ID), "401 Unauthorized").await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health, None).await,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
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
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.3","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
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
