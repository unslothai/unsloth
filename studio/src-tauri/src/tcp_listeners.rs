// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! A connect to a closed 127.0.0.1 port on Windows runs until the 2 s client timeout; skip ports
//! no 127.0.0.1 connect can reach. One such port in a concurrent scan costs the whole timeout,
//! so a listener on `::1` or a LAN address alone must not count.

use std::collections::HashSet;

/// `None` (not Windows, or any table read failed) means probe every port (fail open).
pub(crate) fn loopback_listening_ports() -> Option<HashSet<u16>> {
    #[cfg(windows)]
    {
        windows_tables::loopback_listening_ports()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

pub(crate) fn retain_listening(ports: Vec<u16>, listening: Option<&HashSet<u16>>) -> Vec<u16> {
    match listening {
        Some(listening) => ports
            .into_iter()
            .filter(|port| listening.contains(port))
            .collect(),
        None => ports,
    }
}

/// dwLocalPort is in network byte order in the low 16 bits.
#[cfg_attr(not(windows), allow(dead_code))]
fn port_from_table_dword(value: u32) -> u16 {
    u16::from_be(value as u16)
}

#[cfg_attr(not(windows), allow(dead_code))]
fn v4_reaches_loopback(addr: [u8; 4]) -> bool {
    addr == [127, 0, 0, 1] || addr == [0, 0, 0, 0]
}

/// `::` may be dual-stack (IPV6_V6ONLY is not in the table), so it is kept.
#[cfg_attr(not(windows), allow(dead_code))]
fn v6_reaches_loopback(addr: [u8; 16]) -> bool {
    let ip = std::net::Ipv6Addr::from(addr);
    ip.is_unspecified()
        || ip
            .to_ipv4_mapped()
            .is_some_and(|v4| v4_reaches_loopback(v4.octets()))
}

#[cfg(windows)]
mod windows_tables {
    use super::{port_from_table_dword, v4_reaches_loopback, v6_reaches_loopback};
    use std::collections::HashSet;
    use windows_sys::Win32::Foundation::{
        ERROR_INSUFFICIENT_BUFFER, ERROR_NOT_SUPPORTED, NO_ERROR,
    };
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCP6ROW_OWNER_PID, MIB_TCP6TABLE_OWNER_PID, MIB_TCPROW_OWNER_PID,
        MIB_TCPTABLE_OWNER_PID, TCP_TABLE_OWNER_PID_LISTENER,
    };

    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 23;
    const MAX_ATTEMPTS: usize = 4;

    pub(super) fn loopback_listening_ports() -> Option<HashSet<u16>> {
        let mut ports = HashSet::new();
        let v4 = read_table(AF_INET)??;
        collect::<MIB_TCPROW_OWNER_PID>(
            &v4,
            std::mem::offset_of!(MIB_TCPTABLE_OWNER_PID, table),
            |row| {
                v4_reaches_loopback(row.dwLocalAddr.to_ne_bytes())
                    .then(|| port_from_table_dword(row.dwLocalPort))
            },
            &mut ports,
        )?;
        // Unsupported IPv6 family counts as no IPv6 listeners.
        if let Some(v6) = read_table(AF_INET6)? {
            collect::<MIB_TCP6ROW_OWNER_PID>(
                &v6,
                std::mem::offset_of!(MIB_TCP6TABLE_OWNER_PID, table),
                |row| {
                    v6_reaches_loopback(row.ucLocalAddr)
                        .then(|| port_from_table_dword(row.dwLocalPort))
                },
                &mut ports,
            )?;
        }
        Some(ports)
    }

    /// `None` on failure, `Some(None)` when the address family is not supported.
    fn read_table(family: u32) -> Option<Option<Vec<u32>>> {
        let mut size = 0u32;
        let mut buffer: Vec<u32> = Vec::new();
        for _ in 0..MAX_ATTEMPTS {
            let rc = unsafe {
                GetExtendedTcpTable(
                    if buffer.is_empty() {
                        std::ptr::null_mut()
                    } else {
                        buffer.as_mut_ptr().cast()
                    },
                    &mut size,
                    0,
                    family,
                    TCP_TABLE_OWNER_PID_LISTENER,
                    0,
                )
            };
            match rc {
                NO_ERROR if !buffer.is_empty() => return Some(Some(buffer)),
                NO_ERROR | ERROR_INSUFFICIENT_BUFFER => {
                    let bytes = (size as usize).max(4) + 1024;
                    buffer = vec![0u32; bytes.div_ceil(4)];
                    size = (buffer.len() * 4) as u32;
                }
                ERROR_NOT_SUPPORTED if family == AF_INET6 => return Some(None),
                _ => return None,
            }
        }
        None
    }

    fn collect<Row>(
        buffer: &[u32],
        rows_offset: usize,
        reachable_port: impl Fn(&Row) -> Option<u16>,
        ports: &mut HashSet<u16>,
    ) -> Option<()> {
        let count = *buffer.first()? as usize;
        let needed = count
            .checked_mul(std::mem::size_of::<Row>())?
            .checked_add(rows_offset)?;
        if needed > buffer.len() * 4 {
            return None;
        }
        let rows = unsafe {
            std::slice::from_raw_parts(
                buffer.as_ptr().cast::<u8>().add(rows_offset).cast::<Row>(),
                count,
            )
        };
        ports.extend(rows.iter().filter_map(reachable_port));
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_listeners_probe_every_port() {
        let ports: Vec<u16> = (8888..=8908).collect();
        assert_eq!(retain_listening(ports.clone(), None), ports);
    }

    #[test]
    fn known_listeners_keep_only_listening_ports_in_order() {
        let ports: Vec<u16> = vec![9010, 8888, 8890, 8908];
        let listening: HashSet<u16> = [8908, 8888, 22, 9010].into_iter().collect();
        assert_eq!(
            retain_listening(ports, Some(&listening)),
            vec![9010, 8888, 8908]
        );
    }

    #[test]
    fn an_empty_listener_table_probes_nothing() {
        let ports: Vec<u16> = (8888..=8908).collect();
        assert!(retain_listening(ports, Some(&HashSet::new())).is_empty());
    }

    #[test]
    fn table_ports_are_network_byte_order() {
        assert_eq!(
            port_from_table_dword(u32::from_le_bytes([0x22, 0xB8, 0, 0])),
            8888
        );
        assert_eq!(
            port_from_table_dword(u32::from_le_bytes([0x00, 0x50, 0xAB, 0xCD])),
            80
        );
    }

    #[test]
    fn only_addresses_a_127_0_0_1_connect_reaches_count() {
        assert!(v4_reaches_loopback([127, 0, 0, 1]));
        assert!(v4_reaches_loopback([0, 0, 0, 0]));
        assert!(!v4_reaches_loopback([127, 0, 0, 2]));
        assert!(!v4_reaches_loopback([192, 168, 1, 5]));

        let v6 = |s: &str| s.parse::<std::net::Ipv6Addr>().unwrap().octets();
        assert!(v6_reaches_loopback(v6("::")));
        assert!(v6_reaches_loopback(v6("::ffff:127.0.0.1")));
        assert!(v6_reaches_loopback(v6("::ffff:0.0.0.0")));
        assert!(!v6_reaches_loopback(v6("::1")));
        assert!(!v6_reaches_loopback(v6("fe80::1")));
        assert!(!v6_reaches_loopback(v6("::ffff:10.0.0.1")));
    }

    #[cfg(not(windows))]
    #[test]
    fn other_platforms_report_unknown() {
        assert!(loopback_listening_ports().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn windows_tables_list_only_loopback_reachable_listeners() {
        let v4 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let v4_port = v4.local_addr().unwrap().port();
        let any = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
        let any_port = any.local_addr().unwrap().port();
        let v6_only = std::net::TcpListener::bind("[::1]:0").ok();
        let ports = loopback_listening_ports().expect("listener tables readable");
        assert!(
            ports.contains(&v4_port),
            "missing 127.0.0.1 listener {v4_port}"
        );
        assert!(
            ports.contains(&any_port),
            "missing 0.0.0.0 listener {any_port}"
        );
        if let Some(v6) = v6_only {
            let v6_port = v6.local_addr().unwrap().port();
            assert!(
                !ports.contains(&v6_port),
                "::1-only listener {v6_port} listed"
            );
        }

        let closed = {
            let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            probe.local_addr().unwrap().port()
        };
        let ports = loopback_listening_ports().expect("listener tables readable");
        assert!(
            !ports.contains(&closed),
            "closed port {closed} still listed"
        );
    }
}
