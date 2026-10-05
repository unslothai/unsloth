use super::types::BackendProbe;
use super::version::{
    backend_version_stale_reason, DESKTOP_BACKEND_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};

use log::info;
use std::time::Duration;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct DesktopOwnerHealth {
    kind: Option<String>,
    token_sha256: Option<String>,
}

#[derive(Debug)]
pub(super) struct BackendHealth {
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    native_path_leases_supported: Option<bool>,
    studio_root_id: Option<String>,
    desktop_owner: Option<DesktopOwnerHealth>,
    version: Option<String>,
    stale_reason: Option<String>,
}

pub(super) async fn backend_health(client: &reqwest::Client, port: u16) -> Option<BackendHealth> {
    let started = Instant::now();
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
    info!(
        "Desktop preflight: health probe on port {} healthy={} service={} in {}ms",
        port,
        healthy,
        service,
        started.elapsed().as_millis()
    );

    if !healthy || !service {
        return None;
    }

    let desktop_protocol_version = json
        .get("desktop_protocol_version")
        .and_then(|v| v.as_u64())
        .and_then(|v| u16::try_from(v).ok());
    let desktop_manageability_version = json
        .get("desktop_manageability_version")
        .and_then(|v| v.as_u64())
        .and_then(|v| u16::try_from(v).ok());
    let supports_desktop_auth = json.get("supports_desktop_auth").and_then(|v| v.as_bool());
    let supports_desktop_backend_ownership = json
        .get("supports_desktop_backend_ownership")
        .and_then(|v| v.as_bool());
    let native_path_leases_supported = json
        .get("native_path_leases_supported")
        .and_then(|v| v.as_bool());
    let studio_root_id = json
        .get("studio_root_id")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let desktop_owner = json
        .get("desktop_owner")
        .and_then(|v| serde_json::from_value::<DesktopOwnerHealth>(v.clone()).ok());
    let version = json
        .get("version")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let stale_reason = match supports_desktop_auth {
        Some(false) => json
            .get("desktop_auth_stale_reason")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    };
    Some(BackendHealth {
        desktop_protocol_version,
        desktop_manageability_version,
        supports_desktop_auth,
        supports_desktop_backend_ownership,
        native_path_leases_supported,
        studio_root_id,
        desktop_owner,
        version,
        stale_reason,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendRootStatus {
    SameRoot,
    ForeignRoot,
    AmbiguousRoot,
    ExpectedUnavailable,
}

fn backend_root_status(
    health: &BackendHealth,
    expected_studio_root_id: Option<&str>,
) -> BackendRootStatus {
    let Some(expected) = expected_studio_root_id else {
        return BackendRootStatus::ExpectedUnavailable;
    };

    match health.studio_root_id.as_deref() {
        Some(actual) if actual == expected => BackendRootStatus::SameRoot,
        Some(actual) if crate::desktop_backend_owner::is_valid_studio_root_id(actual) => {
            BackendRootStatus::ForeignRoot
        }
        _ => BackendRootStatus::AmbiguousRoot,
    }
}

fn backend_desktop_owner_match(
    health: &BackendHealth,
) -> crate::desktop_backend_owner::HealthOwnerMatch {
    let Some(owner) = health.desktop_owner.as_ref() else {
        return crate::desktop_backend_owner::HealthOwnerMatch::None;
    };
    crate::desktop_backend_owner::classify_health_desktop_owner(
        health.studio_root_id.as_deref(),
        owner.kind.as_deref(),
        owner.token_sha256.as_deref(),
    )
}

fn backend_capability_stale_reason(health: &BackendHealth) -> Option<String> {
    if health.desktop_protocol_version != Some(DESKTOP_PROTOCOL_VERSION) {
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
    if health.desktop_manageability_version.unwrap_or(0) < DESKTOP_BACKEND_MANAGEABILITY_VERSION {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if health.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    // Unauthenticated /api/health gates `version` behind a bearer, and the bits
    // above predate the floor, so no version means "auth-gated", not "old".
    match health.version.as_deref() {
        Some(version) if !version.is_empty() => backend_version_stale_reason(Some(version)),
        _ => None,
    }
}

/// Deliberately NOT part of `backend_capability_stale_reason`.
///
/// Only the desktop spawn sets the lease secret (`process.rs`), so this does not
/// mean "too old to talk to", it means "this app did not start it" -- permanently
/// true of a terminal-started backend. Folded in with the protocol bits it would
/// make every one of those an ExternalConflict and the app would refuse to start,
/// to fix a picker `use-linked-folders.ts` already greys out. Apply it only where
/// a restart is ours to perform.
fn backend_native_lease_stale_reason(health: &BackendHealth) -> Option<String> {
    if health.native_path_leases_supported != Some(true) {
        return Some("native_path_leases_unsupported".to_string());
    }
    None
}

#[derive(Serialize)]
struct DesktopLoginProbe<'a> {
    secret: &'a str,
}

pub(super) async fn probe_ownerless_spawned_backend(port: u16) -> BackendProbe {
    let client = match crate::loopback_http::client(Duration::from_secs(2)) {
        Ok(client) => client,
        Err(_) => return BackendProbe::Missing,
    };
    let Some(health) = backend_health(&client, port).await else {
        return BackendProbe::Missing;
    };
    if let Some(reason) = backend_capability_stale_reason(&health) {
        return BackendProbe::Old { port, reason };
    }
    // One WE spawned: a missing secret is a defect in our own start, so restart.
    if let Some(reason) = backend_native_lease_stale_reason(&health) {
        return BackendProbe::Old { port, reason };
    }

    let response = client
        .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
        .json(&DesktopLoginProbe {
            secret: "desktop-preflight-invalid-secret",
        })
        .send()
        .await;
    match response.map(|response| response.status()) {
        Ok(reqwest::StatusCode::UNAUTHORIZED) => BackendProbe::Ready { port },
        Ok(reqwest::StatusCode::NOT_FOUND) => BackendProbe::Old {
            port,
            reason: "desktop_login_not_found".to_string(),
        },
        _ => BackendProbe::Old {
            port,
            reason: "desktop_login_probe_failed".to_string(),
        },
    }
}

pub(super) async fn backend_desktop_auth_status(
    client: &reqwest::Client,
    port: u16,
    health: &BackendHealth,
    expected_studio_root_id: Option<&str>,
) -> BackendProbe {
    let root_status = backend_root_status(health, expected_studio_root_id);
    let owner_match = backend_desktop_owner_match(health);
    let verified_owner = owner_match == crate::desktop_backend_owner::HealthOwnerMatch::CurrentApp;
    let same_root_external = root_status == BackendRootStatus::SameRoot && !verified_owner;

    match root_status {
        BackendRootStatus::AmbiguousRoot | BackendRootStatus::ExpectedUnavailable => {
            // Not adoptable, and not proof that our own install is serving: an
            // id-less backend is just as likely a remote Unsloth reached through
            // a port forward. Starting is fine, the port simply is not free.
            return BackendProbe::Unrelated {
                port,
                reason: "ambiguous_root_external_backend_active".to_string(),
            };
        }
        BackendRootStatus::ForeignRoot => {
            return BackendProbe::Old {
                port,
                reason: "studio_root_id_mismatch".to_string(),
            };
        }
        BackendRootStatus::SameRoot => {}
    }

    if same_root_external
        && matches!(
            owner_match,
            crate::desktop_backend_owner::HealthOwnerMatch::PreviousApp
                | crate::desktop_backend_owner::HealthOwnerMatch::OtherDesktopOwner
        )
    {
        return BackendProbe::ExternalConflict {
            port,
            reason: "desktop_owned_backend_active".to_string(),
        };
    }

    if let Some(reason) = backend_capability_stale_reason(health) {
        return if same_root_external {
            BackendProbe::ExternalConflict { port, reason }
        } else {
            BackendProbe::Old { port, reason }
        };
    }
    // Owned only: ours to restart, and the restart re-injects the secret. An
    // ownerless same-root backend is terminal-started, so blocking on it would
    // make the app unstartable. See backend_native_lease_stale_reason.
    if !same_root_external {
        if let Some(reason) = backend_native_lease_stale_reason(health) {
            return BackendProbe::Old { port, reason };
        }
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
        let reason = backend_capability_stale_reason(health)
            .unwrap_or_else(|| "desktop_login_probe_failed".to_string());
        return if same_root_external {
            BackendProbe::ExternalConflict { port, reason }
        } else {
            BackendProbe::Old { port, reason }
        };
    };

    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => BackendProbe::Ready { port },
        reqwest::StatusCode::NOT_FOUND => {
            let reason = "desktop_login_not_found".to_string();
            if same_root_external {
                BackendProbe::ExternalConflict { port, reason }
            } else {
                BackendProbe::Old { port, reason }
            }
        }
        _ => {
            let reason = backend_capability_stale_reason(health)
                .unwrap_or_else(|| "desktop_login_probe_failed".to_string());
            if same_root_external {
                BackendProbe::ExternalConflict { port, reason }
            } else {
                BackendProbe::Old { port, reason }
            }
        }
    }
}

/// Classify every candidate port. Callers pick from the result, because launch
/// and mutation weigh the outcomes differently: launch may step over a backend
/// it cannot adopt, a venv rewrite may not.
pub(super) async fn probe_backend_ports(ignored_ports: &[u16]) -> Vec<BackendProbe> {
    let client = match crate::loopback_http::client(Duration::from_secs(2)) {
        Ok(client) => client,
        Err(_) => return Vec::new(),
    };

    // Fan out health probes concurrently. The desktop-auth probe is still
    // sequential per candidate because it has auth-log side effects.
    let ports: Vec<u16> = crate::desktop_backend_owner::desktop_candidate_ports().collect();
    let mut health_futs = Vec::with_capacity(ports.len());
    for port in ports {
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

    let expected_studio_root_id = crate::desktop_backend_owner::read_expected_studio_root_id();
    let mut probes = Vec::new();
    for (port, health) in candidates {
        if ignored_ports.contains(&port) {
            continue;
        }
        probes.push(
            backend_desktop_auth_status(&client, port, &health, expected_studio_root_id.as_deref())
                .await,
        );
    }
    probes
}

/// Pick the outcome that decides a launch. An `Unrelated` backend is not a
/// verdict at all: the port is taken, the backend picks the next one.
fn select_launch_probe(probes: Vec<BackendProbe>) -> BackendProbe {
    let mut first_conflict = None;
    let mut first_ready = None;
    let mut first_old = None;
    for probe in probes {
        match probe {
            conflict @ BackendProbe::ExternalConflict { .. } if first_conflict.is_none() => {
                first_conflict = Some(conflict)
            }
            ready @ BackendProbe::Ready { .. } if first_ready.is_none() => {
                first_ready = Some(ready)
            }
            old @ BackendProbe::Old { .. } if first_old.is_none() => first_old = Some(old),
            BackendProbe::Unrelated { port, reason } => {
                info!("Desktop preflight: skipping port {port} ({reason})");
            }
            _ => {}
        }
    }

    first_conflict
        .or(first_ready)
        .or(first_old)
        .unwrap_or(BackendProbe::Missing)
}

pub(super) async fn probe_existing_backends(ignored_ports: &[u16]) -> BackendProbe {
    select_launch_probe(probe_backend_ports(ignored_ports).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn health(studio_root_id: Option<&str>) -> BackendHealth {
        BackendHealth {
            desktop_protocol_version: None,
            desktop_manageability_version: None,
            supports_desktop_auth: None,
            supports_desktop_backend_ownership: None,
            native_path_leases_supported: None,
            studio_root_id: studio_root_id.map(str::to_string),
            desktop_owner: None,
            version: None,
            stale_reason: None,
        }
    }

    fn unrelated(port: u16) -> BackendProbe {
        BackendProbe::Unrelated {
            port,
            reason: "ambiguous_root_external_backend_active".to_string(),
        }
    }

    /// A tunnelled or id-less Unsloth anywhere in 8888..=8908 used to poison the
    /// whole launch, even with every other port free.
    #[test]
    fn unrelated_backends_never_decide_a_launch() {
        assert_eq!(
            select_launch_probe(vec![unrelated(8888), unrelated(8899)]),
            BackendProbe::Missing
        );
        assert_eq!(
            select_launch_probe(vec![unrelated(8888), BackendProbe::Ready { port: 8890 }]),
            BackendProbe::Ready { port: 8890 }
        );
    }

    #[test]
    fn a_genuine_conflict_still_decides_a_launch() {
        let conflict = BackendProbe::ExternalConflict {
            port: 8890,
            reason: "desktop_owned_backend_active".to_string(),
        };
        assert_eq!(
            select_launch_probe(vec![
                unrelated(8888),
                BackendProbe::Ready { port: 8889 },
                conflict.clone(),
            ]),
            conflict
        );
    }

    /// Minting an id at startup must not reclassify a backend that predates
    /// ownership: it reports no id, or an empty one, and stays unadoptable.
    #[test]
    fn ownerless_backends_stay_ambiguous_once_a_local_id_exists() {
        for reported in [None, Some(""), Some("not-hex"), Some("AAAA")] {
            for expected in [None, Some(LOCAL)] {
                assert_eq!(
                    backend_root_status(&health(reported), expected),
                    if expected.is_none() {
                        BackendRootStatus::ExpectedUnavailable
                    } else {
                        BackendRootStatus::AmbiguousRoot
                    },
                    "reported={reported:?} expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn a_valid_different_id_is_a_foreign_install_not_an_ambiguous_one() {
        assert_eq!(
            backend_root_status(&health(Some(OTHER)), Some(LOCAL)),
            BackendRootStatus::ForeignRoot
        );
        assert_eq!(
            backend_root_status(&health(Some(LOCAL)), Some(LOCAL)),
            BackendRootStatus::SameRoot
        );
    }
}
