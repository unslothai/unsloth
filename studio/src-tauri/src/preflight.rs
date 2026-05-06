use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopPreflightDisposition {
    NotInstalled,
    ManagedReady,
    ManagedStale,
    AttachedReady,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopPreflightResult {
    pub disposition: DesktopPreflightDisposition,
    pub reason: Option<String>,
    pub port: Option<u16>,
    pub can_auto_repair: bool,
    pub managed_bin: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ManagedProbe {
    Missing,
    Ready { bin: PathBuf },
    Stale { bin: PathBuf, reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BackendProbe {
    Missing,
    Ready { port: u16 },
    Old { port: u16, reason: String },
}

#[derive(Debug, Deserialize)]
struct DesktopCapability {
    desktop_protocol_version: Option<u16>,
    supports_api_only: Option<bool>,
    supports_provision_desktop_auth: Option<bool>,
    desktop_auth_stale_reason: Option<String>,
}

#[derive(Debug)]
struct BackendHealth {
    desktop_protocol_version: Option<u16>,
    supports_desktop_auth: Option<bool>,
    stale_reason: Option<String>,
}

fn release_auto_repair() -> bool {
    !cfg!(debug_assertions)
}

fn choose_preflight(managed: ManagedProbe, backend: BackendProbe) -> DesktopPreflightResult {
    match (backend, managed) {
        (BackendProbe::Ready { port }, ManagedProbe::Ready { bin }) => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::AttachedReady,
            reason: None,
            port: Some(port),
            can_auto_repair: false,
            managed_bin: Some(bin),
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

async fn run_cli_probe(bin: &std::path::Path, args: &[&str]) -> bool {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = cmd.spawn() else {
        return false;
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    }
}

async fn probe_cli_capability(bin: &std::path::Path) -> Option<DesktopCapability> {
    let mut cmd = Command::new(bin);
    cmd.args(["studio", "desktop-capabilities", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = cmd.spawn() else {
        return None;
    };
    let Some(mut stdout) = child.stdout.take() else {
        return None;
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) if status.success() => {}
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return None;
        }
        _ => return None,
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return None;
    }

    serde_json::from_slice::<DesktopCapability>(&output).ok()
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    capability.desktop_protocol_version == Some(1)
        && capability.supports_api_only == Some(true)
        && capability.supports_provision_desktop_auth == Some(true)
}

fn desktop_capability_stale_reason(capability: &DesktopCapability) -> String {
    capability
        .desktop_auth_stale_reason
        .clone()
        .unwrap_or_else(|| "desktop_capability_incompatible".to_string())
}

async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    if !run_cli_probe(&bin, &["-h"]).await {
        return ManagedProbe::Stale {
            bin,
            reason: "cli_unusable".to_string(),
        };
    }

    let capability = probe_cli_capability(&bin).await;
    if let Some(capability) = capability {
        if desktop_capability_ready(&capability) {
            return ManagedProbe::Ready { bin };
        }
        return ManagedProbe::Stale {
            bin,
            reason: desktop_capability_stale_reason(&capability),
        };
    }

    ManagedProbe::Stale {
        bin,
        reason: "desktop_capability_probe_failed".to_string(),
    }
}

async fn probe_managed_install() -> ManagedProbe {
    match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        None => ManagedProbe::Missing,
    }
}

pub async fn managed_install_ready() -> bool {
    matches!(probe_managed_install().await, ManagedProbe::Ready { .. })
}

async fn backend_health(client: &reqwest::Client, port: u16) -> Option<BackendHealth> {
    let url = format!("http://127.0.0.1:{port}/api/health");
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let json = response.json::<serde_json::Value>().await.ok()?;
    let healthy = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "healthy")
        .unwrap_or(false);
    let service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);
    if !healthy || !service {
        return None;
    }

    let desktop_protocol_version = json
        .get("desktop_protocol_version")
        .and_then(|v| v.as_u64())
        .and_then(|v| u16::try_from(v).ok());
    let supports_desktop_auth = json.get("supports_desktop_auth").and_then(|v| v.as_bool());
    if desktop_protocol_version.is_none() && supports_desktop_auth.is_none() {
        return None;
    }
    let stale_reason = match supports_desktop_auth {
        Some(false) => json
            .get("desktop_auth_stale_reason")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    };
    Some(BackendHealth {
        desktop_protocol_version,
        supports_desktop_auth,
        stale_reason,
    })
}

fn backend_capability_stale_reason(health: &BackendHealth) -> Option<String> {
    if health.desktop_protocol_version != Some(1) {
        return health
            .stale_reason
            .clone()
            .or_else(|| Some("desktop_protocol_incompatible".to_string()));
    }
    if health.supports_desktop_auth != Some(true) {
        return health
            .stale_reason
            .clone()
            .or_else(|| Some("desktop_auth_unsupported".to_string()));
    }
    None
}

#[derive(Serialize)]
struct DesktopLoginProbe<'a> {
    secret: &'a str,
}

async fn backend_desktop_auth_status(
    client: &reqwest::Client,
    port: u16,
    health: &BackendHealth,
) -> BackendProbe {
    if let Some(reason) = backend_capability_stale_reason(health) {
        return BackendProbe::Old { port, reason };
    }

    let url = format!("http://127.0.0.1:{port}/api/auth/desktop-login");
    let response = client
        .post(url)
        .json(&DesktopLoginProbe {
            secret: "desktop-preflight-invalid-secret",
        })
        .send()
        .await;

    let Ok(response) = response else {
        return BackendProbe::Old {
            port,
            reason: backend_capability_stale_reason(health)
                .unwrap_or_else(|| "desktop_login_probe_failed".to_string()),
        };
    };

    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => BackendProbe::Ready { port },
        reqwest::StatusCode::NOT_FOUND => BackendProbe::Old {
            port,
            reason: "desktop_login_not_found".to_string(),
        },
        _ => BackendProbe::Old {
            port,
            reason: backend_capability_stale_reason(health)
                .unwrap_or_else(|| "desktop_login_probe_failed".to_string()),
        },
    }
}

async fn probe_existing_backends() -> BackendProbe {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return BackendProbe::Missing,
    };

    // Fan out health probes concurrently. The desktop-auth probe is still
    // sequential per candidate because it has auth-log side effects.
    let ports: Vec<u16> = (8888u16..=8908).collect();
    let mut health_futs = Vec::with_capacity(ports.len());
    for port in ports {
        // why: reqwest::Client is internally Arc-wrapped; clone is a refcount bump
        // (documented cheap). tokio::spawn needs 'static, so each task owns its own clone.
        let c = client.clone();
        health_futs.push(tokio::spawn(async move {
            backend_health(&c, port).await.map(|h| (port, h))
        }));
    }

    let mut candidates: Vec<(u16, BackendHealth)> = Vec::new();
    for fut in health_futs {
        if let Ok(Some(pair)) = fut.await {
            candidates.push(pair);
        }
    }

    let mut first_old = None;
    for (port, health) in candidates {
        match backend_desktop_auth_status(&client, port, &health).await {
            ready @ BackendProbe::Ready { .. } => return ready,
            old @ BackendProbe::Old { .. } if first_old.is_none() => first_old = Some(old),
            _ => {}
        }
    }

    first_old.unwrap_or(BackendProbe::Missing)
}

pub async fn desktop_preflight_result() -> DesktopPreflightResult {
    let (managed, backend) = tokio::join!(probe_managed_install(), probe_existing_backends());
    choose_preflight(managed, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn compatible_backend_does_not_win_over_stale_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Ready { port: 8000 },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedStale
        );
        assert_eq!(result.port, None);
        assert_eq!(result.reason, Some("old cli".to_string()));
        assert_eq!(result.can_auto_repair, release_auto_repair());
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
    fn compatible_backend_does_not_win_over_missing_managed_install() {
        let result = choose_preflight(ManagedProbe::Missing, BackendProbe::Ready { port: 8000 });

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::NotInstalled
        );
        assert_eq!(result.port, None);
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
  printf '{"desktop_protocol_version":1,"supports_api_only":true,"supports_provision_desktop_auth":true}'
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
  printf '{"desktop_protocol_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"desktop_auth_stale_reason":"cap_false"}'
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
  printf '{"desktop_protocol_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"desktop_auth_stale_reason":"cap_false"}'
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

    async fn backend_server(health_body: &'static str, route_status: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0; 2048];
                let n = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..n]);
                let (status, body) = if request.starts_with("GET /api/health ") {
                    ("200 OK", health_body)
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
        health_body: &'static str,
        route_status: &'static str,
    ) -> BackendProbe {
        let port = backend_server(health_body, route_status).await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();
        backend_desktop_auth_status(&client, port, &health).await
    }

    #[tokio::test]
    async fn backend_health_without_desktop_capability_fields_is_not_compatible() {
        let port = backend_server(
            r#"{"status":"healthy","service":"Unsloth UI Backend"}"#,
            "401 Unauthorized",
        )
        .await;
        let client = reqwest::Client::new();

        assert!(backend_health(&client, port).await.is_none());
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_missing_protocol_is_old() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_unsupported_protocol_is_old() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":2,"supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_health_with_desktop_capability_fields_and_401_is_ready() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn backend_route_404_is_old() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "404 Not Found",
        )
        .await;

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
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "500 Internal Server Error",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_capability_false_is_old_even_when_route_401() {
        let port = backend_server(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":false,"desktop_auth_stale_reason":"cap_false"}"#,
            "401 Unauthorized",
        )
        .await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health).await,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "cap_false"
        ));
    }
}
