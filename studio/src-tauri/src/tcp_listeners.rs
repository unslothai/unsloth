// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! Windows retransmits a SYN that loopback answers with a reset, so a connect to a closed
//! 127.0.0.1 port takes one to two seconds there instead of well under a millisecond. Reading the
//! listener tables once lets a port scan skip ports nothing listens on, which a connect would
//! only have seen refused.

use std::collections::HashSet;

/// Ports with a TCP listener on any local address, IPv4 or IPv6. `None` means unknown (not
/// Windows, or the table read failed): callers probe every port as before.
pub(crate) fn listening_ports() -> Option<HashSet<u16>> {
    #[cfg(windows)]
    {
        windows_tables::listening_ports()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Keeps order. A port with a listener on any address is kept, so the result is a superset of
/// the ports a 127.0.0.1 connect could reach.
pub(crate) fn retain_listening(ports: Vec<u16>, listening: Option<&HashSet<u16>>) -> Vec<u16> {
    match listening {
        Some(listening) => ports
            .into_iter()
            .filter(|port| listening.contains(port))
            .collect(),
        None => ports,
    }
}

/// The tables store the port in network byte order in the low 16 bits of a DWORD.
#[cfg_attr(not(windows), allow(dead_code))]
fn port_from_table_dword(value: u32) -> u16 {
    u16::from_be(value as u16)
}

#[cfg(windows)]
mod windows_tables {
    use super::port_from_table_dword;
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

    pub(super) fn listening_ports() -> Option<HashSet<u16>> {
        let mut ports = HashSet::new();
        let v4 = read_table(AF_INET)??;
        collect::<MIB_TCPROW_OWNER_PID>(
            &v4,
            std::mem::offset_of!(MIB_TCPTABLE_OWNER_PID, table),
            |row| row.dwLocalPort,
            &mut ports,
        )?;
        // No IPv6 stack means no dual-stack socket that could take a 127.0.0.1 connect.
        if let Some(v6) = read_table(AF_INET6)? {
            collect::<MIB_TCP6ROW_OWNER_PID>(
                &v6,
                std::mem::offset_of!(MIB_TCP6TABLE_OWNER_PID, table),
                |row| row.dwLocalPort,
                &mut ports,
            )?;
        }
        Some(ports)
    }

    /// `None` on failure, `Some(None)` when the address family is not supported.
    fn read_table(family: u32) -> Option<Option<Vec<u32>>> {
        let mut size = 0u32;
        // u32 storage keeps the rows (all DWORD-aligned) aligned.
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
                NO_ERROR if !buffer.is_empty() => {
                    buffer.truncate((size as usize).div_ceil(4).min(buffer.len()));
                    return Some(Some(buffer));
                }
                NO_ERROR | ERROR_INSUFFICIENT_BUFFER => {
                    // Listeners can appear between calls: leave room for a few more rows.
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

    /// Both table layouts are a DWORD count followed by the rows.
    fn collect<Row>(
        buffer: &[u32],
        rows_offset: usize,
        local_port: impl Fn(&Row) -> u32,
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
        ports.extend(
            rows.iter()
                .map(|row| port_from_table_dword(local_port(row))),
        );
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
        // 8888 = 0x22B8, stored as bytes 22 B8, which a little-endian DWORD reads as 0xB822.
        assert_eq!(
            port_from_table_dword(u32::from_le_bytes([0x22, 0xB8, 0, 0])),
            8888
        );
        assert_eq!(
            port_from_table_dword(u32::from_le_bytes([0x00, 0x50, 0xAB, 0xCD])),
            80
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn other_platforms_report_unknown() {
        assert!(listening_ports().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn windows_tables_list_live_listeners_on_both_families() {
        let v4 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let v4_port = v4.local_addr().unwrap().port();
        let v6 = std::net::TcpListener::bind("[::1]:0").ok();
        let ports = listening_ports().expect("listener tables readable");
        assert!(ports.contains(&v4_port), "missing IPv4 listener {v4_port}");
        if let Some(v6) = v6 {
            let v6_port = v6.local_addr().unwrap().port();
            assert!(ports.contains(&v6_port), "missing IPv6 listener {v6_port}");
        }

        let closed = {
            let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            probe.local_addr().unwrap().port()
        };
        let ports = listening_ports().expect("listener tables readable");
        assert!(
            !ports.contains(&closed),
            "closed port {closed} still listed"
        );
    }
}
