// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Loopback proxy for browser-panel pages: resolves hosts itself, refuses private answers and
//! connects to the checked address, defeating DNS rebinding. Hosts the system proxy covers go to it
//! by name instead, after the same check, so a poisoned public answer or none at all doesn't matter.

use crate::browser_webview::{host_is_private, ip_is_private};
use hyper_util::client::proxy::matcher::Matcher;
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
                            // Read per connection, so a proxy switched on or off applies at once.
                            tauri::async_runtime::spawn(async move {
                                serve(client, system_proxies()).await
                            });
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
    forward: Option<Forward>,
}

/// A plain-HTTP request, without the proxy's own headers.
#[derive(Debug, PartialEq)]
struct Forward {
    method: String,
    url: Url,
    version: String,
    headers: String,
}

impl Forward {
    /// One request per connection: origin-form to the site, absolute-form to the system proxy.
    fn head(&self, upstream: Option<&Upstream>) -> String {
        let target = match upstream {
            Some(_) => self.url.as_str(),
            None => &self.url[Position::BeforePath..],
        };
        let mut head = format!(
            "{} {target} {}\r\n{}",
            self.method, self.version, self.headers
        );
        if let Some(auth) = upstream.and_then(|upstream| upstream.auth.as_deref()) {
            head.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
        }
        head.push_str("Connection: close\r\n\r\n");
        head
    }
}

/// The system proxy a request goes through.
#[derive(Debug, PartialEq)]
struct Upstream {
    address: String,
    auth: Option<String>,
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
    let mut headers = String::new();
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
            headers.push_str(line);
            headers.push_str("\r\n");
        }
    }
    Some(Request {
        host: url.host_str()?.to_string(),
        port: url.port_or_known_default()?,
        forward: Some(Forward {
            method: method.to_string(),
            url,
            version: version.to_string(),
            headers,
        }),
    })
}

/// The system's proxy settings: proxy variables, else macOS or Windows settings.
fn system_proxies() -> Matcher {
    #[cfg(windows)]
    if let Some(proxies) = windows_per_protocol() {
        return proxies;
    }
    Matcher::from_system()
}

/// hyper-util reads Windows' `ProxyServer` as one address, so its per-protocol form
/// (`http=a:80;https=b:443`) is split here.
#[cfg(windows)]
fn windows_per_protocol() -> Option<Matcher> {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_QUERY_VALUE};
    use winreg::RegKey;

    let settings = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags(
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            KEY_QUERY_VALUE,
        )
        .ok()?;
    if settings.get_value::<u32, _>("ProxyEnable").unwrap_or(0) == 0 {
        return None;
    }
    let server = settings.get_value::<String, _>("ProxyServer").ok()?;
    let proxy_override = settings
        .get_value::<String, _>("ProxyOverride")
        .unwrap_or_default();
    per_protocol(&server, &proxy_override, |name| std::env::var(name).ok())
}

/// Proxies for a per-protocol `ProxyServer` value (`None` for a single address). As in hyper-util,
/// each proxy variable overrides its own setting.
#[cfg(any(windows, test))]
fn per_protocol(
    server: &str,
    proxy_override: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Option<Matcher> {
    if !server.contains('=') {
        return None;
    }
    let setting = |wanted: &str| {
        server
            .split(';')
            .filter_map(|entry| entry.split_once('='))
            .find(|(scheme, _)| scheme.trim().eq_ignore_ascii_case(wanted))
            .map(|(_, address)| address.trim().to_string())
            .unwrap_or_default()
    };
    let variable = |names: [&str; 2]| names.into_iter().find_map(&env);
    // ProxyOverride as hyper-util reads it: `;`-separated, `*.` wildcards as suffixes.
    let bypass = || {
        proxy_override
            .split(';')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(",")
            .replace("*.", "")
    };
    Some(
        Matcher::builder()
            .all(variable(["ALL_PROXY", "all_proxy"]).unwrap_or_default())
            .http(variable(["HTTP_PROXY", "http_proxy"]).unwrap_or_else(|| setting("http")))
            .https(variable(["HTTPS_PROXY", "https_proxy"]).unwrap_or_else(|| setting("https")))
            .no(variable(["NO_PROXY", "no_proxy"]).unwrap_or_else(bypass))
            .build(),
    )
}

/// The system proxy for `request`, if one covers it. Only HTTP proxies are chained.
fn system_proxy(proxies: &Matcher, request: &Request) -> Option<Upstream> {
    let destination: http::Uri = match &request.forward {
        Some(forward) => forward.url.as_str().parse().ok()?,
        // Tunnels carry TLS, so they take the HTTPS proxy.
        None => format!("https://{}:{}/", request.host, request.port)
            .parse()
            .ok()?,
    };
    let intercept = proxies.intercept(&destination)?;
    let uri = intercept.uri();
    if uri.scheme_str() != Some("http") {
        log::info!(
            "browser proxy can't chain to a {:?} proxy",
            uri.scheme_str()
        );
        return None;
    }
    Some(Upstream {
        address: format!("{}:{}", uri.host()?, uri.port_u16().unwrap_or(80)),
        auth: intercept
            .basic_auth()
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
    })
}

/// Parse `host`, refusing it if it names a private address.
fn public_host(host: &str) -> Result<Host<String>, Refusal> {
    let parsed = Host::parse(host).map_err(|_| Refusal::Unreachable)?;
    let borrowed = match &parsed {
        Host::Domain(name) => Host::Domain(name.as_str()),
        Host::Ipv4(ip) => Host::Ipv4(*ip),
        Host::Ipv6(ip) => Host::Ipv6(*ip),
    };
    if host_is_private(&borrowed) {
        return Err(Refusal::Private);
    }
    Ok(parsed)
}

/// Connect to a public address of `host`, refusing it if any answer is private. One deadline covers
/// the lookup and every address, so a host with many dead answers can't hold a task for long.
async fn connect(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    tokio::time::timeout(TIMEOUT, connect_within(host, port))
        .await
        .unwrap_or(Err(Refusal::Unreachable))
}

async fn connect_within(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    let addresses: Vec<SocketAddr> = match public_host(host)? {
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
    dial(addresses).await
}

async fn dial(addresses: impl IntoIterator<Item = SocketAddr>) -> Result<TcpStream, Refusal> {
    for address in addresses {
        if let Ok(Ok(stream)) =
            tokio::time::timeout(ADDRESS_TIMEOUT, TcpStream::connect(address)).await
        {
            return Ok(stream);
        }
    }
    Err(Refusal::Unreachable)
}

/// Open `request` through the system proxy, which gets the name. Never falls back to a direct
/// connection. Returns what the proxy sent past its answer.
async fn connect_via(
    upstream: &Upstream,
    request: &Request,
) -> Result<(TcpStream, Vec<u8>), Refusal> {
    let host = public_host(&request.host)?;
    tokio::time::timeout(TIMEOUT, async {
        refuse_private_answers(&host, request.port).await?;
        let addresses = tokio::net::lookup_host(upstream.address.as_str())
            .await
            .map_err(|_| Refusal::Unreachable)?;
        let mut stream = dial(addresses).await?;
        if request.forward.is_some() {
            return Ok((stream, Vec::new()));
        }
        let target = format!("{}:{}", request.host, request.port);
        let mut head = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
        if let Some(auth) = &upstream.auth {
            head.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
        }
        head.push_str("\r\n");
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|_| Refusal::Unreachable)?;
        let (answer, early) = read_head(&mut stream).await.ok_or(Refusal::Unreachable)?;
        let status = answer.split_ascii_whitespace().nth(1).unwrap_or("");
        if !status.starts_with('2') {
            log::info!("system proxy answered {status} for {target}");
            return Err(Refusal::Unreachable);
        }
        Ok((stream, early))
    })
    .await
    .unwrap_or(Err(Refusal::Unreachable))
}

/// Refuse a name local DNS answers privately, as the direct path does. One it can't answer in time
/// (a proxy-only network) is left to the proxy.
async fn refuse_private_answers(host: &Host<String>, port: u16) -> Result<(), Refusal> {
    let Host::Domain(name) = host else {
        return Ok(());
    };
    // Its own share of the deadline, so a resolver dropping queries leaves time for the proxy.
    let answers = tokio::time::timeout(
        ADDRESS_TIMEOUT,
        tokio::net::lookup_host((name.as_str(), port)),
    );
    let private = answers.await.is_ok_and(|answers| {
        answers.is_ok_and(|mut answers| answers.any(|answer| ip_is_private(answer.ip())))
    });
    if private {
        return Err(Refusal::Private);
    }
    Ok(())
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

async fn serve(mut client: TcpStream, proxies: Matcher) {
    let Ok(Some((head, rest))) = tokio::time::timeout(TIMEOUT, read_head(&mut client)).await else {
        return;
    };
    let Some(request) = parse(&head) else {
        let _ = reply(&mut client, "400 Bad Request").await;
        return;
    };
    let via = system_proxy(&proxies, &request);
    let opened = match &via {
        Some(via) => connect_via(via, &request).await,
        None => connect(&request.host, request.port)
            .await
            .map(|stream| (stream, Vec::new())),
    };
    let (mut upstream, early) = match opened {
        Ok(opened) => opened,
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
        Some(forward) => {
            upstream
                .write_all(forward.head(via.as_ref()).as_bytes())
                .await
        }
        None => {
            client
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
        }
    };
    if opened.is_ok()
        && client.write_all(&early).await.is_ok()
        && upstream.write_all(&rest).await.is_ok()
    {
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
    use tokio::task::JoinHandle;

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
        let forward = plain.forward.unwrap();
        assert_eq!(
            forward.head(None),
            "GET /a?b=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nConnection: close\r\n\r\n"
        );
        let proxy = Upstream {
            address: "127.0.0.1:9".into(),
            auth: Some("Basic dTpw".into()),
        };
        assert_eq!(
            forward.head(Some(&proxy)),
            "GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nProxy-Authorization: Basic dTpw\r\nConnection: close\r\n\r\n"
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

    /// The port of the proxy `per_protocol` picks for `url`, given proxy variables `env`.
    fn port_for(server: &str, bypass: &str, env: &[(&str, &str)], url: &str) -> Option<u16> {
        let env = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, v)| v.to_string())
        };
        let intercept = per_protocol(server, bypass, env)?.intercept(&url.parse().unwrap())?;
        intercept.uri().port_u16()
    }

    #[test]
    fn windows_per_protocol_proxies_are_split() {
        let both = "http=127.0.0.1:1; HTTPS=127.0.0.1:2;socks=127.0.0.1:3";
        let corp = "*.corp.example;<local>";
        let (http, https) = ("http://a.example/", "https://a.example/");
        assert!(per_protocol("127.0.0.1:7890", "", |_| None).is_none());
        let (wiki, http_wiki) = ("https://wiki.corp.example/", "http://wiki.corp.example/");
        for (url, port) in [(http, Some(1)), (https, Some(2)), (wiki, None)] {
            assert_eq!(port_for(both, corp, &[], url), port, "{url}");
        }
        // Each variable overrides only its own setting.
        let http_var = [("HTTP_PROXY", "http://127.0.0.1:9")];
        assert_eq!(port_for(both, "", &http_var, http), Some(9));
        assert_eq!(port_for(both, "", &http_var, https), Some(2));
        let https_var = [
            ("HTTPS_PROXY", "http://127.0.0.1:8"),
            ("NO_PROXY", "a.example"),
        ];
        for (url, port) in [(https, None), (wiki, Some(8)), (http_wiki, Some(1))] {
            assert_eq!(port_for(both, corp, &https_var, url), port, "{url}");
        }
        let all_var = [("ALL_PROXY", "http://127.0.0.1:7")];
        assert_eq!(port_for("http=127.0.0.1:1", "", &all_var, https), Some(7));
    }

    // A name local DNS can't answer, so only a request sent on by name gets through.
    const UNRESOLVABLE: &str = "studio-proxy-test.invalid";

    fn tunnel(target: &str) -> String {
        format!("CONNECT {target} HTTP/1.1\r\n\r\n")
    }

    /// A system proxy that records the request head it gets and sends `answer`.
    async fn system_proxy_answering(answer: &'static str) -> (Matcher, JoinHandle<Option<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://u:p@{}", listener.local_addr().unwrap());
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.ok()?;
            let (head, _) = read_head(&mut stream).await?;
            stream.write_all(answer.as_bytes()).await.ok()?;
            Some(head)
        });
        let proxies = Matcher::builder().http(&address).https(&address).build();
        (proxies, seen)
    }

    async fn exchange(proxies: Matcher, request: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        tokio::spawn(serve(listener.accept().await.unwrap().0, proxies));
        client.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn requests_go_through_the_system_proxy_by_name() {
        let target = format!("{UNRESOLVABLE}:443");
        let auth = "Proxy-Authorization: Basic dTpw";
        let (proxies, seen) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await;
        let response = exchange(proxies, &tunnel(&target)).await;
        assert_eq!(response, "HTTP/1.1 200 Connection Established\r\n\r\nhello");
        let expected = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{auth}\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
        let (proxies, seen) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await;
        let url = format!("http://{UNRESOLVABLE}/a");
        let response = exchange(proxies, &format!("GET {url} HTTP/1.1\r\n\r\n")).await;
        assert_eq!(response, "HTTP/1.1 200 OK\r\n\r\nhello");
        let expected = format!("GET {url} HTTP/1.1\r\n{auth}\r\nConnection: close\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
    }

    #[tokio::test]
    async fn private_hosts_and_refused_tunnels_fail_through_the_system_proxy() {
        let (proxies, _) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\n").await;
        let response = exchange(proxies, "CONNECT printer.local:80 HTTP/1.1\r\n\r\n").await;
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
        let (proxies, _) = system_proxy_answering("HTTP/1.1 407 Proxy Auth\r\n\r\n").await;
        let response = exchange(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
        assert!(response.starts_with("HTTP/1.1 502"), "{response}");
    }
}
