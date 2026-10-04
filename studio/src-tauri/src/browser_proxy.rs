// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Loopback proxy for browser-panel pages: resolves hosts itself, refuses private answers and
//! connects to the checked address, defeating DNS rebinding.

use crate::browser_webview::{host_is_private, ip_is_private};
use std::net::{IpAddr, SocketAddr};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use url::{Host, Position, Url};

const MAX_HEAD: usize = 16 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);
// One address's share of it, so a dead first answer leaves time for the next.
const ADDRESS_TIMEOUT: Duration = Duration::from_secs(10);

static ADDRESS: OnceLock<Result<SocketAddr, String>> = OnceLock::new();

pub fn address() -> Result<SocketAddr, String> {
    ADDRESS
        .get_or_init(|| {
            let listener =
                std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
            listener.set_nonblocking(true).map_err(|e| e.to_string())?;
            let address = listener.local_addr().map_err(|e| e.to_string())?;
            tauri::async_runtime::spawn(async move {
                let Ok(listener) = TcpListener::from_std(listener) else {
                    return;
                };
                loop {
                    match listener.accept().await {
                        Ok((client, _)) => {
                            tauri::async_runtime::spawn(serve(client));
                        }
                        // Out of descriptors, say: don't spin.
                        Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
                    }
                }
            });
            Ok(address)
        })
        .clone()
}

#[derive(Debug, PartialEq)]
struct Request {
    host: String,
    port: u16,
    forward: Option<String>,
}

#[derive(Debug, PartialEq)]
enum Refusal {
    Private,
    Unreachable,
}

fn parse(head: &str) -> Option<Request> {
    let mut lines = head.split("\r\n");
    let mut parts = lines.next()?.split_ascii_whitespace();
    let (method, target, version) = (parts.next()?, parts.next()?, parts.next()?);
    if method.eq_ignore_ascii_case("CONNECT") {
        let url = Url::parse(&format!("http://{target}/")).ok()?;
        return Some(Request {
            host: url.host_str()?.to_string(),
            port: url.port_or_known_default()?,
            forward: None,
        });
    }
    let url = Url::parse(target).ok()?;
    if url.scheme() != "http" {
        return None;
    }
    // One request per connection, without the proxy's own headers.
    let mut forward = format!("{method} {} {version}\r\n", &url[Position::BeforePath..]);
    for line in lines.filter(|line| !line.is_empty()) {
        let name = line
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if !matches!(
            name.as_str(),
            "connection" | "keep-alive" | "proxy-connection" | "proxy-authorization"
        ) {
            forward.push_str(line);
            forward.push_str("\r\n");
        }
    }
    forward.push_str("Connection: close\r\n\r\n");
    Some(Request {
        host: url.host_str()?.to_string(),
        port: url.port_or_known_default()?,
        forward: Some(forward),
    })
}

/// Connect to a public address of `host`, refusing it if any answer is private. One deadline covers
/// the lookup and every address, so a host with many dead answers can't hold a task for long.
async fn connect(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    tokio::time::timeout(TIMEOUT, connect_within(host, port))
        .await
        .unwrap_or(Err(Refusal::Unreachable))
}

async fn connect_within(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    let parsed = Host::parse(host).map_err(|_| Refusal::Unreachable)?;
    let borrowed = match &parsed {
        Host::Domain(name) => Host::Domain(name.as_str()),
        Host::Ipv4(ip) => Host::Ipv4(*ip),
        Host::Ipv6(ip) => Host::Ipv6(*ip),
    };
    if host_is_private(&borrowed) {
        return Err(Refusal::Private);
    }
    let addresses: Vec<SocketAddr> = match parsed {
        Host::Ipv4(ip) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Host::Ipv6(ip) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        Host::Domain(name) => tokio::net::lookup_host((name.as_str(), port))
            .await
            .map_err(|_| Refusal::Unreachable)?
            .collect(),
    };
    if addresses.iter().any(|address| ip_is_private(address.ip())) {
        return Err(Refusal::Private);
    }
    for address in addresses {
        if let Ok(Ok(stream)) =
            tokio::time::timeout(ADDRESS_TIMEOUT, TcpStream::connect(address)).await
        {
            return Ok(stream);
        }
    }
    Err(Refusal::Unreachable)
}

async fn read_head(client: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = client.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let rest = buffer.split_off(end + 4);
            return Some((String::from_utf8(buffer).ok()?, rest));
        }
        if buffer.len() > MAX_HEAD {
            return None;
        }
    }
}

async fn serve(mut client: TcpStream) {
    let Ok(Some((head, rest))) = tokio::time::timeout(TIMEOUT, read_head(&mut client)).await else {
        return;
    };
    let Some(request) = parse(&head) else {
        let _ = reply(&mut client, "400 Bad Request").await;
        return;
    };
    let mut upstream = match connect(&request.host, request.port).await {
        Ok(stream) => stream,
        Err(Refusal::Private) => {
            log::info!("browser proxy refused private host {}", request.host);
            let _ = reply(&mut client, "403 Forbidden").await;
            return;
        }
        Err(Refusal::Unreachable) => {
            let _ = reply(&mut client, "502 Bad Gateway").await;
            return;
        }
    };
    let opened = match &request.forward {
        Some(forward) => upstream.write_all(forward.as_bytes()).await,
        None => {
            client
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
        }
    };
    if opened.is_ok() && upstream.write_all(&rest).await.is_ok() {
        let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
    }
}

async fn reply(client: &mut TcpStream, status: &str) -> std::io::Result<()> {
    let response = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    client.write_all(response.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tunnels_and_plain_requests() {
        assert_eq!(
            parse("CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n"),
            Some(Request {
                host: "example.com".into(),
                port: 443,
                forward: None
            })
        );
        assert_eq!(
            parse("CONNECT [::1]:8888 HTTP/1.1\r\n\r\n").unwrap().host,
            "[::1]"
        );
        let plain = parse(
            "GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\nProxy-Connection: keep-alive\r\nConnection: keep-alive\r\nAccept: */*\r\n\r\n",
        )
        .unwrap();
        assert_eq!((plain.host.as_str(), plain.port), ("example.com", 80));
        assert_eq!(
            plain.forward.unwrap(),
            "GET /a?b=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nConnection: close\r\n\r\n"
        );
        assert_eq!(parse("GET /relative HTTP/1.1\r\n\r\n"), None);
        assert_eq!(parse("GET https://example.com/ HTTP/1.1\r\n\r\n"), None);
    }

    #[tokio::test]
    async fn private_hosts_are_refused_however_written() {
        for host in [
            "localhost",
            "127.0.0.1",
            "127.1",
            "[::1]",
            "[::ffff:7f00:1]",
            "10.0.0.1",
            "printer.local",
        ] {
            assert_eq!(
                connect(host, 80).await.err(),
                Some(Refusal::Private),
                "{host}"
            );
        }
    }

    #[tokio::test]
    async fn the_proxy_answers_a_private_tunnel_with_403() {
        let address = tokio::task::spawn_blocking(address).await.unwrap().unwrap();
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"CONNECT 127.0.0.1:22 HTTP/1.1\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    }
}
