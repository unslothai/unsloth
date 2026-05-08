mod backend;
mod managed;
mod types;
mod version;

use backend::probe_existing_backends;
pub use managed::managed_install_ready;
use managed::probe_managed_install;
use std::path::PathBuf;
use types::{BackendProbe, ManagedProbe};
pub use types::{DesktopPreflightDisposition, DesktopPreflightResult, ExternalBackendConflict};

#[cfg(test)]
use backend::{backend_desktop_auth_status, backend_health};
#[cfg(test)]
use managed::probe_managed_bin;
#[cfg(test)]
use version::{
    backend_version_compatible, backend_version_stale_reason, MIN_DESKTOP_BACKEND_VERSION,
};

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
    let _ = crate::desktop_backend_owner::cleanup_verified_desktop_orphan().await;
    let (managed, backend) = tokio::join!(probe_managed_install(), probe_existing_backends(&[]));
    choose_preflight(managed, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn compatible_backend_wins_over_stale_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Ready { port: 8000 },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.reason, None);
        assert!(!result.can_auto_repair);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn compatible_backend_wins_over_ready_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Ready { port: 8000 },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn compatible_backend_wins_over_missing_managed_install() {
        let result = choose_preflight(ManagedProbe::Missing, BackendProbe::Ready { port: 8000 });

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.managed_bin, None);
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_falls_back_to_ready_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Old {
                port: 8001,
                reason: "missing endpoint".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedReady
        );
        assert_eq!(result.reason, None);
        assert_eq!(result.port, None);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_falls_back_to_stale_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Old {
                port: 8001,
                reason: "missing endpoint".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedStale
        );
        assert_eq!(result.reason, Some("old cli".to_string()));
        assert_eq!(result.port, None);
        assert_eq!(result.can_auto_repair, release_auto_repair());
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn managed_ready_when_no_backend() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Missing,
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedReady
        );
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn managed_stale_when_no_backend() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Missing,
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedStale
        );
        assert_eq!(result.reason, Some("old cli".to_string()));
        assert_eq!(result.can_auto_repair, release_auto_repair());
    }

    #[test]
    fn not_installed_when_no_backend_no_managed_binary() {
        let result = choose_preflight(ManagedProbe::Missing, BackendProbe::Missing);

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::NotInstalled
        );
        assert_eq!(result.managed_bin, None);
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_with_no_managed_install_uses_install_flow() {
        let result = choose_preflight(
            ManagedProbe::Missing,
            BackendProbe::Old {
                port: 8002,
                reason: "old version".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::NotInstalled
        );
        assert_eq!(result.reason, None);
        assert_eq!(result.port, None);
        assert!(!result.can_auto_repair);
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
    fn backend_version_gate_accepts_minimum_and_newer_versions() {
        assert!(backend_version_compatible(Some(
            MIN_DESKTOP_BACKEND_VERSION
        )));
        assert!(backend_version_compatible(Some("2026.5.3")));
        assert!(backend_version_compatible(Some("2027.1.0")));
    }

    #[test]
    fn backend_version_gate_accepts_pep_style_suffixes() {
        assert!(backend_version_compatible(Some("2026.5.2.post1")));
        assert!(backend_version_compatible(Some("2026.5.2+local")));
        assert!(backend_version_compatible(Some("2026.5.2rc1")));
        assert!(backend_version_compatible(Some("2026.5.2.dev1")));
    }

    #[test]
    fn backend_version_gate_rejects_missing_invalid_and_older_versions() {
        assert_eq!(
            backend_version_stale_reason(None),
            Some("desktop_backend_version_missing".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("not-a-version")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.2.1")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.2foo")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.1")),
            Some("desktop_backend_version_too_old".to_string())
        );
    }

    #[test]
    fn backend_version_gate_allows_dev_only_in_debug_builds() {
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
    async fn managed_cli_stale_when_desktop_capabilities_missing() {
        let fake = fake_cli(
            "cap-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale { bin: actual_bin, .. } if actual_bin == bin
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_stale_when_desktop_capabilities_command_missing() {
        let fake = fake_cli(
            "missing-helper",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale { bin: actual_bin, .. } if actual_bin == bin
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_stale_when_help_broken() {
        let fake = fake_cli(
            "broken-help",
            r#"#!/bin/sh
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cli_unusable".to_string()
            }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_ready_when_desktop_capabilities_compatible() {
        let fake = fake_cli(
            "cap-true-helper-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"version":"2026.5.2"}'
  exit 0
fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Ready { bin }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn capability_false_reason_used_when_legacy_helper_missing() {
        let fake = fake_cli(
            "cap-false-helper-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","version":"2026.5.2"}'
  exit 0
fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cap_false".to_string()
            }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn capability_false_overrides_working_legacy_helper() {
        let fake = fake_cli(
            "cap-false-helper-ready",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","version":"2026.5.2"}'
  exit 0
fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cap_false".to_string()
            }
        );
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
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{root_id}"{owner}}}"#
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
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_unsupported_protocol_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":2,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
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
    async fn backend_route_500_is_old() {
        let probe = probe_test_backend(
            desktop_ready_health(EXPECTED_ROOT_ID),
            "500 Internal Server Error",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_capability_false_is_old_even_when_route_401() {
        install_test_owner();
        let port = backend_server(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health, Some(EXPECTED_ROOT_ID)).await,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "cap_false"
        ));
    }
}
