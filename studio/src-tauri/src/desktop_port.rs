// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

use std::ffi::OsStr;

pub(crate) const PORT_ENV: &str = "UNSLOTH_DESKTOP_PORT";
// Fetch forbids these HTTP ports: https://fetch.spec.whatwg.org/#port-blocking
const BROWSER_BLOCKED_PORTS: &[u16] = &[
    1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669,
    6679, 6697, 10080,
];

pub(crate) fn configured_port(value: Option<&OsStr>) -> Result<Option<u16>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let invalid = || {
        format!("{PORT_ENV} must be between 1024 and 65515; the chosen port and its next 20 ports must be allowed by browsers.")
    };
    let value = value.to_str().ok_or_else(invalid)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<u16>()
        .ok()
        .filter(|&port| {
            (1024..=65515).contains(&port)
                && fallback_ports(port).all(|candidate| !BROWSER_BLOCKED_PORTS.contains(&candidate))
        })
        .map(Some)
        .ok_or_else(invalid)
}

pub(crate) fn resolve_port(requested: u16, value: Option<&OsStr>) -> Result<u16, String> {
    Ok(configured_port(value)?.unwrap_or(requested))
}

pub(crate) fn fallback_ports(start: u16) -> std::ops::RangeInclusive<u16> {
    start..=start.saturating_add(20)
}

pub(crate) fn discovery_ports(configured: Option<u16>) -> Vec<u16> {
    let mut ports: Vec<u16> = configured.into_iter().flat_map(fallback_ports).collect();
    for port in fallback_ports(8888) {
        if !ports.contains(&port) {
            ports.push(port);
        }
    }
    ports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_or_empty_keeps_the_requested_port() {
        for requested in [8888, 9010] {
            for value in [None, Some(OsStr::new("")), Some(OsStr::new("  "))] {
                assert_eq!(resolve_port(requested, value), Ok(requested));
            }
        }
    }

    #[test]
    fn custom_port_is_stable_across_start_requests() {
        for requested in [8888, 9010] {
            for _ in 0..3 {
                assert_eq!(
                    resolve_port(requested, Some(OsStr::new(" 9123 "))),
                    Ok(9123)
                );
            }
        }
    }

    #[test]
    fn invalid_ports_are_rejected() {
        for value in [
            "0",
            "80",
            "65516",
            "65535",
            "65536",
            "-1",
            "localhost:9000",
            "abc",
        ] {
            assert!(resolve_port(8888, Some(OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn every_accepted_fallback_is_browser_reachable_and_in_range() {
        for port in 1..=u16::MAX {
            if let Ok(Some(start)) = configured_port(Some(OsStr::new(&port.to_string()))) {
                assert_eq!(start, port);
                let candidates: Vec<_> = fallback_ports(start).collect();
                assert_eq!(candidates.len(), 21);
                assert!(candidates
                    .iter()
                    .all(|p| !BROWSER_BLOCKED_PORTS.contains(p)));
            }
        }
        for port in [1024, 3000, 8080, 9000, 65515] {
            assert_eq!(
                configured_port(Some(OsStr::new(&port.to_string()))),
                Ok(Some(port))
            );
        }
        for port in BROWSER_BLOCKED_PORTS {
            for offset in 0..=20 {
                assert!(configured_port(Some(OsStr::new(&(port - offset).to_string()))).is_err());
            }
        }
    }

    #[test]
    fn default_discovery_is_unchanged() {
        assert_eq!(discovery_ports(None), (8888..=8908).collect::<Vec<_>>());
        assert_eq!(discovery_ports(Some(8888)), discovery_ports(None));
    }

    #[test]
    fn custom_discovery_keeps_fallback_and_existing_backends() {
        let ports = discovery_ports(Some(19000));
        assert_eq!(&ports[..21], (19000..=19020).collect::<Vec<_>>());
        assert_eq!(&ports[21..], (8888..=8908).collect::<Vec<_>>());
        assert_eq!(discovery_ports(Some(8890)).len(), 23);
        assert_eq!(fallback_ports(65535).collect::<Vec<_>>(), vec![65535]);
    }
}
