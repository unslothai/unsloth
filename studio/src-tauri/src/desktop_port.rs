// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

use std::ffi::OsStr;

pub(crate) const PORT_ENV: &str = "UNSLOTH_DESKTOP_PORT";

pub(crate) fn configured_port(value: Option<&OsStr>) -> Result<Option<u16>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let invalid = || format!("{PORT_ENV} must be a port from 1 to 65535.");
    let value = value.to_str().ok_or_else(invalid)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
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
        for value in ["0", "65536", "-1", "localhost:9000", "abc"] {
            assert_eq!(
                resolve_port(8888, Some(OsStr::new(value))),
                Err("UNSLOTH_DESKTOP_PORT must be a port from 1 to 65535.".to_string()),
            );
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
