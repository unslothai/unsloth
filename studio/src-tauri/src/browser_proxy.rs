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
    /// One request per connection: absolute-form to an HTTP proxy, origin-form otherwise.
    fn head(&self, upstream: Option<&Upstream>) -> String {
        let (target, auth) = match upstream {
            Some(Upstream::Http { auth, .. }) => (self.url.as_str(), auth.as_deref()),
            _ => (&self.url[Position::BeforePath..], None),
        };
        let mut head = format!(
            "{} {target} {}\r\n{}",
            self.method, self.version, self.headers
        );
        if let Some(auth) = auth {
            head.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
        }
        head.push_str("Connection: close\r\n\r\n");
        head
    }
}

/// The system proxy a request goes through.
#[derive(Debug, PartialEq)]
enum Upstream {
    Http {
        address: String,
        auth: Option<String>,
    },
    Socks {
        address: String,
        auth: Option<(String, String)>,
    },
}

#[derive(Debug, PartialEq)]
enum Refusal {
    Private,
    Unreachable,
    /// The proxy itself can't be reached, so the next route may be tried.
    ProxyDown,
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

/// How the system routes pages.
#[derive(Default)]
struct Proxies {
    /// A PAC script, which decides before `fixed` unless it can't be fetched or run.
    #[cfg(target_os = "macos")]
    pac: Option<pac::Pac>,
    /// Fixed settings, in order: proxy variables, else macOS or Windows settings.
    fixed: Vec<Matcher>,
}

fn system_proxies() -> Proxies {
    #[cfg(windows)]
    if let Some(proxies) = windows_per_protocol() {
        return Proxies {
            fixed: vec![proxies],
        };
    }
    #[cfg(target_os = "macos")]
    let proxies = macos_proxies(|name| std::env::var(name).ok());
    #[cfg(not(target_os = "macos"))]
    let proxies = Proxies {
        fixed: vec![Matcher::from_system()],
    };
    proxies
}

/// Proxy settings by protocol, as macOS and Windows store them.
#[cfg(any(target_os = "macos", windows, test))]
#[derive(Default)]
struct Settings {
    http: String,
    https: String,
    bypass: String,
}

/// Proxies for `settings`. As in hyper-util, each proxy variable that is set and not empty
/// overrides its own setting.
#[cfg(any(target_os = "macos", windows, test))]
fn with_variables(settings: Settings, env: &impl Fn(&str) -> Option<String>) -> Matcher {
    let variable = |names| variable(env, names);
    Matcher::builder()
        .all(variable(["ALL_PROXY", "all_proxy"]).unwrap_or_default())
        .http(variable(["HTTP_PROXY", "http_proxy"]).unwrap_or(settings.http))
        .https(variable(["HTTPS_PROXY", "https_proxy"]).unwrap_or(settings.https))
        .no(variable(["NO_PROXY", "no_proxy"]).unwrap_or(settings.bypass))
        .build()
}

#[cfg(any(target_os = "macos", windows, test))]
fn variable(env: &impl Fn(&str) -> Option<String>, names: [&str; 2]) -> Option<String> {
    names
        .into_iter()
        .filter_map(env)
        .find(|value| !value.is_empty())
}

/// macOS' proxy settings, read here rather than by hyper-util, which skips PAC, the SOCKS proxy and
/// the exceptions and can't parse an IPv6 host. The SOCKS proxy serves what HTTP(S) ones don't cover.
#[cfg(target_os = "macos")]
fn macos_proxies(env: impl Fn(&str) -> Option<String>) -> Proxies {
    use system_configuration::core_foundation::array::CFArray;
    use system_configuration::core_foundation::base::{CFType, TCFType};
    use system_configuration::core_foundation::number::CFNumber;
    use system_configuration::core_foundation::string::{CFString, CFStringRef};
    use system_configuration::dynamic_store::SCDynamicStoreBuilder;
    use system_configuration::sys::schema_definitions::*;

    let Some(proxies) = SCDynamicStoreBuilder::new("unsloth-browser")
        .build()
        .and_then(|store| store.get_proxies())
    else {
        return Proxies {
            pac: None,
            fixed: vec![with_variables(Settings::default(), &env)],
        };
    };
    let number = |key| {
        proxies
            .find(key)
            .and_then(|value| value.downcast::<CFNumber>())
            .and_then(|value| value.to_i32())
    };
    let text = |key| {
        proxies
            .find(key)
            .and_then(|value| value.downcast::<CFString>())
            .map(|value| value.to_string())
    };
    let proxy = |enable: CFStringRef, host: CFStringRef, port: CFStringRef| {
        if number(enable) != Some(1) {
            return None;
        }
        Some(authority(&text(host)?, number(port)))
    };
    let pac = unsafe {
        if number(kSCPropNetProxiesProxyAutoConfigEnable) != Some(1) {
            None
        } else {
            text(kSCPropNetProxiesProxyAutoConfigJavaScript)
                .map(|script| pac::Source::Script(script.into()))
                .or_else(|| text(kSCPropNetProxiesProxyAutoConfigURLString).map(pac::Source::Url))
        }
    };
    let pac = pac.map(|source| pac::Pac::new(source, &env));
    let exceptions = proxies
        .find(unsafe { kSCPropNetProxiesExceptionsList })
        .and_then(|value| value.downcast::<CFArray>())
        .map(|list| {
            list.iter()
                .filter_map(|item| {
                    unsafe { CFType::wrap_under_get_rule(*item as _) }.downcast::<CFString>()
                })
                .map(|entry| entry.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let settings = unsafe {
        Settings {
            http: proxy(
                kSCPropNetProxiesHTTPEnable,
                kSCPropNetProxiesHTTPProxy,
                kSCPropNetProxiesHTTPPort,
            )
            .unwrap_or_default(),
            https: proxy(
                kSCPropNetProxiesHTTPSEnable,
                kSCPropNetProxiesHTTPSProxy,
                kSCPropNetProxiesHTTPSPort,
            )
            .unwrap_or_default(),
            bypass: bypass_list(exceptions.iter().map(String::as_str)),
        }
    };
    let socks = unsafe {
        proxy(
            kSCPropNetProxiesSOCKSEnable,
            kSCPropNetProxiesSOCKSProxy,
            kSCPropNetProxiesSOCKSPort,
        )
    };
    let no = variable(&env, ["NO_PROXY", "no_proxy"]).unwrap_or_else(|| settings.bypass.clone());
    let mut fixed = vec![with_variables(settings, &env)];
    fixed.extend(socks.map(|address| socks_fallback(&address, &no)));
    Proxies { pac, fixed }
}

/// `host[:port]`, with an IPv6 host bracketed as a URI needs.
#[cfg(any(target_os = "macos", test))]
fn authority(host: &str, port: Option<i32>) -> String {
    let host = match host.parse::<std::net::Ipv6Addr>() {
        Ok(_) => format!("[{host}]"),
        Err(_) => host.to_string(),
    };
    match port {
        Some(port) => format!("{host}:{port}"),
        None => host,
    }
}

/// A NO_PROXY list from bypass entries, `*.` wildcards as suffixes as hyper-util reads them.
#[cfg(any(target_os = "macos", windows, test))]
fn bypass_list<'a>(entries: impl Iterator<Item = &'a str>) -> String {
    entries
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(",")
        .replace("*.", "")
}

/// Every destination through the SOCKS proxy at `address`, minus the `no` list.
#[cfg(any(target_os = "macos", test))]
fn socks_fallback(address: &str, no: &str) -> Matcher {
    Matcher::builder()
        .all(format!("socks5h://{address}"))
        .no(no)
        .build()
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

/// Proxies for a per-protocol `ProxyServer` value (`None` for a single address).
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
    let settings = Settings {
        http: setting("http"),
        https: setting("https"),
        bypass: bypass_list(proxy_override.split(';')),
    };
    Some(with_variables(settings, &env))
}

/// Where `request` may go, in order: through a system proxy, or direct (`None`).
async fn routes(proxies: &Proxies, request: &Request) -> Vec<Option<Upstream>> {
    let destination: Option<http::Uri> = match &request.forward {
        Some(forward) => forward.url.as_str().parse().ok(),
        // Tunnels carry TLS, so they take the HTTPS proxy.
        None => format!("https://{}:{}/", request.host, request.port)
            .parse()
            .ok(),
    };
    let Some(destination) = destination else {
        return vec![None];
    };
    #[cfg(target_os = "macos")]
    if let Some(pac) = proxies.pac.as_ref().filter(|pac| pac.decides(&destination)) {
        // As Safari asks, without a default port (CFNetwork drops the path and query itself).
        let url = Url::parse(&destination.to_string())
            .map_or_else(|_| destination.to_string(), String::from);
        if let Some(routes) = pac::routes(pac, &url).await {
            return routes;
        }
    }
    vec![system_proxy(&proxies.fixed, &destination)]
}

/// The first fixed setting that covers `destination`. HTTP and SOCKS5 proxies are chained.
fn system_proxy(proxies: &[Matcher], destination: &http::Uri) -> Option<Upstream> {
    let intercept = proxies
        .iter()
        .find_map(|proxies| proxies.intercept(destination))?;
    let uri = intercept.uri();
    let address = |default| {
        format!(
            "{}:{}",
            uri.host().unwrap_or(""),
            uri.port_u16().unwrap_or(default)
        )
    };
    match uri.scheme_str() {
        Some("http") => Some(Upstream::Http {
            address: address(80),
            auth: intercept
                .basic_auth()
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        }),
        // Either way the proxy gets the name, as with HTTP proxies.
        Some("socks5" | "socks5h") => Some(Upstream::Socks {
            address: address(1080),
            auth: intercept
                .raw_auth()
                .map(|(user, pass)| (user.to_owned(), pass.to_owned())),
        }),
        scheme => {
            log::info!("browser proxy can't chain to a {scheme:?} proxy");
            None
        }
    }
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
    let (address, http_auth) = match upstream {
        Upstream::Http { address, auth } => (address, auth),
        Upstream::Socks { address, .. } => (address, &None),
    };
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    // Running out of time before the proxy answers the dial means it's down: the next route may go.
    let mut stream = tokio::time::timeout_at(deadline, async {
        refuse_private_answers(&host, request.port).await?;
        let addresses = tokio::net::lookup_host(address.as_str())
            .await
            .map_err(|_| Refusal::ProxyDown)?;
        dial(addresses).await.map_err(|_| Refusal::ProxyDown)
    })
    .await
    .unwrap_or(Err(Refusal::ProxyDown))?;
    tokio::time::timeout_at(deadline, async {
        if let Upstream::Socks { auth, .. } = upstream {
            socks_connect(&mut stream, &host, request.port, auth.as_ref())
                .await
                .ok_or(Refusal::Unreachable)?;
            return Ok((stream, Vec::new()));
        }
        if request.forward.is_some() {
            return Ok((stream, Vec::new()));
        }
        let target = format!("{}:{}", request.host, request.port);
        let mut head = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
        if let Some(auth) = http_auth {
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

/// A SOCKS5 CONNECT (RFC 1928, password auth RFC 1929) that names `host`, so the proxy resolves it.
async fn socks_connect(
    stream: &mut TcpStream,
    host: &Host<String>,
    port: u16,
    auth: Option<&(String, String)>,
) -> Option<()> {
    let methods: &[u8] = if auth.is_some() { &[0, 2] } else { &[0] };
    let mut greeting = vec![5, methods.len() as u8];
    greeting.extend_from_slice(methods);
    stream.write_all(&greeting).await.ok()?;
    let mut chosen = [0u8; 2];
    stream.read_exact(&mut chosen).await.ok()?;
    match (chosen, auth) {
        ([5, 0], _) => {}
        ([5, 2], Some((user, pass))) => {
            let (user, pass) = (user.as_bytes(), pass.as_bytes());
            let mut login = vec![1, u8::try_from(user.len()).ok()?];
            login.extend_from_slice(user);
            login.push(u8::try_from(pass.len()).ok()?);
            login.extend_from_slice(pass);
            stream.write_all(&login).await.ok()?;
            let mut status = [0u8; 2];
            stream.read_exact(&mut status).await.ok()?;
            if status[1] != 0 {
                return None;
            }
        }
        _ => return None,
    }
    let mut connect = vec![5, 1, 0];
    match host {
        Host::Domain(name) => {
            connect.push(3);
            connect.push(u8::try_from(name.len()).ok()?);
            connect.extend_from_slice(name.as_bytes());
        }
        Host::Ipv4(ip) => {
            connect.push(1);
            connect.extend_from_slice(&ip.octets());
        }
        Host::Ipv6(ip) => {
            connect.push(4);
            connect.extend_from_slice(&ip.octets());
        }
    }
    connect.extend_from_slice(&port.to_be_bytes());
    stream.write_all(&connect).await.ok()?;
    let mut reply = [0u8; 4];
    stream.read_exact(&mut reply).await.ok()?;
    if reply[..2] != [5, 0] {
        log::info!("SOCKS proxy answered {} for {host}:{port}", reply[1]);
        return None;
    }
    // Skip the bound address the reply carries.
    let bound = match reply[3] {
        1 => 4,
        4 => 16,
        3 => {
            let mut length = [0u8; 1];
            stream.read_exact(&mut length).await.ok()?;
            usize::from(length[0])
        }
        _ => return None,
    };
    let mut rest = vec![0u8; bound + 2];
    stream.read_exact(&mut rest).await.ok()?;
    Some(())
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

async fn serve(mut client: TcpStream, proxies: Proxies) {
    let Ok(Some((head, rest))) = tokio::time::timeout(TIMEOUT, read_head(&mut client)).await else {
        return;
    };
    let Some(request) = parse(&head) else {
        let _ = reply(&mut client, "400 Bad Request").await;
        return;
    };
    let (mut via, mut opened) = (None, Err(Refusal::Unreachable));
    for route in routes(&proxies, &request).await {
        opened = match &route {
            Some(route) => connect_via(route, &request).await,
            None => connect(&request.host, request.port)
                .await
                .map(|stream| (stream, Vec::new())),
        };
        via = route;
        if !matches!(opened, Err(Refusal::ProxyDown)) {
            break;
        }
    }
    let (mut upstream, early) = match opened {
        Ok(opened) => opened,
        Err(Refusal::Private) => {
            log::info!("browser proxy refused private host {}", request.host);
            let _ = reply(&mut client, "403 Forbidden").await;
            return;
        }
        Err(Refusal::Unreachable | Refusal::ProxyDown) => {
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

/// PAC scripts, fetched directly (they're often served by a local proxy app or the intranet) and
/// run by CFNetwork, as Safari runs them.
#[cfg(target_os = "macos")]
mod pac {
    use super::{authority, variable, Upstream, ADDRESS_TIMEOUT};
    use hyper_util::client::proxy::matcher::Matcher;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use system_configuration::core_foundation::array::{CFArray, CFArrayRef};
    use system_configuration::core_foundation::base::{CFType, TCFType};
    use system_configuration::core_foundation::dictionary::CFDictionary;
    use system_configuration::core_foundation::error::{CFError, CFErrorRef};
    use system_configuration::core_foundation::number::CFNumber;
    use system_configuration::core_foundation::string::{CFString, CFStringRef};
    use system_configuration::core_foundation::url::{CFURLCreateWithString, CFURLRef, CFURL};
    use tokio::sync::Mutex;

    const SCRIPT_TTL: Duration = Duration::from_secs(60);

    #[link(name = "CFNetwork", kind = "framework")]
    extern "C" {
        fn CFNetworkCopyProxiesForAutoConfigurationScript(
            script: CFStringRef,
            url: CFURLRef,
            error: *mut CFErrorRef,
        ) -> CFArrayRef;
        static kCFProxyTypeKey: CFStringRef;
        static kCFProxyHostNameKey: CFStringRef;
        static kCFProxyPortNumberKey: CFStringRef;
        static kCFProxyTypeNone: CFStringRef;
        static kCFProxyTypeHTTP: CFStringRef;
        static kCFProxyTypeHTTPS: CFStringRef;
        static kCFProxyTypeSOCKS: CFStringRef;
    }

    pub(super) enum Source {
        Url(String),
        Script(Arc<str>),
    }

    pub(super) struct Pac {
        source: Source,
        /// Proxy variables, which override the script for their protocols.
        variables: Matcher,
        /// NO_PROXY, which keeps its hosts from the script: a stand-in proxy minus the list.
        bypass: Matcher,
    }

    impl Pac {
        pub(super) fn new(source: Source, env: &impl Fn(&str) -> Option<String>) -> Pac {
            let variable = |names| variable(env, names).unwrap_or_default();
            Pac {
                source,
                variables: Matcher::builder()
                    .all(variable(["ALL_PROXY", "all_proxy"]))
                    .http(variable(["HTTP_PROXY", "http_proxy"]))
                    .https(variable(["HTTPS_PROXY", "https_proxy"]))
                    .build(),
                bypass: Matcher::builder()
                    .all("http://pac.invalid")
                    .no(variable(["NO_PROXY", "no_proxy"]))
                    .build(),
            }
        }

        /// Whether the script routes `destination`, which the fixed settings route otherwise.
        pub(super) fn decides(&self, destination: &http::Uri) -> bool {
            self.variables.intercept(destination).is_none()
                && self.bypass.intercept(destination).is_some()
        }
    }

    /// The script's routes for `url`, `None` meaning direct; nothing if it can't be had or run.
    pub(super) async fn routes(pac: &Pac, url: &str) -> Option<Vec<Option<Upstream>>> {
        let script = match &pac.source {
            Source::Url(location) => script(location).await?,
            Source::Script(script) => script.clone(),
        };
        let url = url.to_string();
        let run = tokio::task::spawn_blocking(move || evaluate(&script, &url));
        tokio::time::timeout(ADDRESS_TIMEOUT, run)
            .await
            .ok()?
            .ok()?
    }

    struct Fetched {
        location: String,
        at: Instant,
        script: Option<Arc<str>>,
    }

    async fn script(location: &str) -> Option<Arc<str>> {
        // Held across the fetch, so a page's many connections fetch the script once.
        static CACHE: Mutex<Option<Fetched>> = Mutex::const_new(None);
        let mut cache = CACHE.lock().await;
        if let Some(fetched) = &*cache {
            if fetched.location == location && fetched.at.elapsed() < SCRIPT_TTL {
                return fetched.script.clone();
            }
        }
        let script = fetch(location).await.map(Arc::from);
        if script.is_none() {
            log::info!("browser proxy couldn't fetch the PAC script at {location}");
        }
        *cache = Some(Fetched {
            location: location.to_string(),
            at: Instant::now(),
            script: script.clone(),
        });
        script
    }

    async fn fetch(location: &str) -> Option<String> {
        let client = crate::loopback_http::client(ADDRESS_TIMEOUT).ok()?;
        let response = client.get(location).send().await.ok()?;
        response.error_for_status().ok()?.text().await.ok()
    }

    /// The entries that are DIRECT or a proxy this one can chain to, in order.
    fn evaluate(script: &str, url: &str) -> Option<Vec<Option<Upstream>>> {
        let (script, url) = (CFString::new(script), CFString::new(url));
        unsafe {
            let url = CFURLCreateWithString(
                std::ptr::null(),
                url.as_concrete_TypeRef(),
                std::ptr::null(),
            );
            if url.is_null() {
                return None;
            }
            let url = CFURL::wrap_under_create_rule(url);
            let mut error: CFErrorRef = std::ptr::null_mut();
            let proxies = CFNetworkCopyProxiesForAutoConfigurationScript(
                script.as_concrete_TypeRef(),
                url.as_concrete_TypeRef(),
                &mut error,
            );
            if proxies.is_null() {
                if !error.is_null() {
                    let error = CFError::wrap_under_create_rule(error);
                    log::info!("browser proxy couldn't run the PAC script: {error}");
                }
                return None;
            }
            let proxies: CFArray<CFDictionary<CFString, CFType>> =
                CFArray::wrap_under_create_rule(proxies);
            let mut routes = Vec::new();
            for proxy in proxies.iter() {
                let text = |key| proxy.find(key).and_then(|v| v.downcast::<CFString>());
                let Some(kind) = text(kCFProxyTypeKey) else {
                    continue;
                };
                let is = |other| kind == CFString::wrap_under_get_rule(other);
                if is(kCFProxyTypeNone) {
                    routes.push(None);
                    continue;
                }
                let (socks, default_port) = if is(kCFProxyTypeSOCKS) {
                    (true, 1080)
                } else if is(kCFProxyTypeHTTP) || is(kCFProxyTypeHTTPS) {
                    (false, 80)
                } else {
                    continue;
                };
                let Some(host) = text(kCFProxyHostNameKey) else {
                    continue;
                };
                let port = proxy
                    .find(kCFProxyPortNumberKey)
                    .and_then(|v| v.downcast::<CFNumber>())
                    .and_then(|v| v.to_i32());
                let address = authority(&host.to_string(), Some(port.unwrap_or(default_port)));
                routes.push(Some(match socks {
                    true => Upstream::Socks {
                        address,
                        auth: None,
                    },
                    false => Upstream::Http {
                        address,
                        auth: None,
                    },
                }));
            }
            // An empty answer means direct, as browsers read one.
            if routes.is_empty() {
                routes.push(None);
            }
            Some(routes)
        }
    }
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
        let proxy = Upstream::Http {
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
        // Each variable overrides only its own setting, and only when not empty.
        let http_var = [("HTTP_PROXY", "http://127.0.0.1:9"), ("HTTPS_PROXY", "")];
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
        exchange_all(vec![proxies], request).await
    }

    async fn exchange_all(fixed: Vec<Matcher>, request: &str) -> String {
        let proxies = Proxies {
            fixed,
            ..Default::default()
        };
        exchange_via(proxies, request).await
    }

    async fn exchange_via(proxies: Proxies, request: &str) -> String {
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

    /// A SOCKS5 proxy that wants `u`/`p`, records the CONNECT it gets, answers with a domain-typed
    /// bound address, sends "hello", then (with `request`) records the request head sent next.
    async fn socks_proxy(request: bool) -> (String, JoinHandle<(String, u16, String)>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut greeting = [0u8; 4];
            stream.read_exact(&mut greeting).await.unwrap();
            assert_eq!(greeting, [5, 2, 0, 2]);
            stream.write_all(&[5, 2]).await.unwrap();
            let mut login = [0u8; 5];
            stream.read_exact(&mut login).await.unwrap();
            assert_eq!(login, [1, 1, b'u', 1, b'p']);
            stream.write_all(&[1, 0]).await.unwrap();
            let mut connect = [0u8; 5];
            stream.read_exact(&mut connect).await.unwrap();
            assert_eq!(connect[..4], [5, 1, 0, 3]);
            let mut name = vec![0u8; usize::from(connect[4])];
            stream.read_exact(&mut name).await.unwrap();
            let mut port = [0u8; 2];
            stream.read_exact(&mut port).await.unwrap();
            stream
                .write_all(&[5, 0, 0, 3, 4, b'b', b'o', b'u', b'n', 0, 80])
                .await
                .unwrap();
            stream.write_all(b"hello").await.unwrap();
            let sent = if request {
                tokio::time::timeout(Duration::from_secs(5), read_head(&mut stream))
                    .await
                    .unwrap()
                    .unwrap()
                    .0
            } else {
                String::new()
            };
            (
                String::from_utf8(name).unwrap(),
                u16::from_be_bytes(port),
                sent,
            )
        });
        (address, seen)
    }

    #[test]
    fn proxy_settings_bracket_ipv6_hosts() {
        assert_eq!(authority("::1", Some(1080)), "[::1]:1080");
        assert_eq!(authority("::1", None), "[::1]");
        assert_eq!(authority("proxy.example", Some(1080)), "proxy.example:1080");
        let proxies = socks_fallback(&authority("::1", Some(1080)), "");
        assert!(proxies
            .intercept(&"https://example.com/".parse().unwrap())
            .is_some());
    }

    #[tokio::test]
    async fn tunnels_go_through_a_socks_proxy_by_name() {
        let (address, seen) = socks_proxy(false).await;
        let proxies = Matcher::builder()
            .all(format!("socks5://u:p@{address}"))
            .build();
        let response = exchange(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
        assert_eq!(response, "HTTP/1.1 200 Connection Established\r\n\r\nhello");
        let (name, port, _) = seen.await.unwrap();
        assert_eq!((name.as_str(), port), (UNRESOLVABLE, 443));
    }

    #[tokio::test]
    async fn plain_requests_through_socks_stay_origin_form() {
        let (address, seen) = socks_proxy(true).await;
        // The SOCKS fallback covers what the first settings leave out.
        let proxies = vec![
            Matcher::builder().build(),
            socks_fallback(&format!("u:p@{address}"), ""),
        ];
        let response = exchange_all(
            proxies,
            &format!("GET http://{UNRESOLVABLE}/a HTTP/1.1\r\nHost: {UNRESOLVABLE}\r\n\r\n"),
        )
        .await;
        assert_eq!(response, "hello");
        let (name, port, sent) = seen.await.unwrap();
        assert_eq!((name.as_str(), port), (UNRESOLVABLE, 80));
        assert!(sent.starts_with("GET /a HTTP/1.1\r\n"), "{sent}");
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn pac_scripts_decide_before_fixed_settings() {
        let script = r#"function FindProxyForURL(url, host) {
            if (host == "h.example") return "PROXY 127.0.0.1:1; DIRECT";
            if (host == "s.example") return "SOCKS 127.0.0.1:2";
            return "DIRECT";
        }"#;
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let location = format!("http://{}/proxy.pac", server.local_addr().unwrap());
        // Answers once, so only a cached script serves the later pages.
        tokio::spawn(async move {
            let (mut stream, _) = server.accept().await.unwrap();
            read_head(&mut stream).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                script.len()
            );
            stream.write_all((head + script).as_bytes()).await.unwrap();
        });
        let env = |name: &str| match name {
            "HTTP_PROXY" => Some("http://127.0.0.1:8".to_string()),
            "NO_PROXY" => Some("n.example".to_string()),
            _ => None,
        };
        let pac = |source| Some(pac::Pac::new(source, &env));
        let fixed = || vec![Matcher::builder().all("http://127.0.0.1:9").build()];
        let proxies = Proxies {
            pac: pac(pac::Source::Url(location)),
            fixed: fixed(),
        };
        let to = |host| parse(&tunnel(&format!("{host}:443"))).unwrap();
        let http = |address: &str| {
            Some(Upstream::Http {
                address: address.into(),
                auth: None,
            })
        };
        assert_eq!(
            routes(&proxies, &to("h.example")).await,
            [http("127.0.0.1:1"), None]
        );
        let socks = Upstream::Socks {
            address: "127.0.0.1:2".into(),
            auth: None,
        };
        assert_eq!(routes(&proxies, &to("s.example")).await, [Some(socks)]);
        assert_eq!(routes(&proxies, &to("d.example")).await, [None]);
        // Proxy variables and NO_PROXY take precedence, as over the other settings.
        let plain = parse("GET http://h.example/ HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(routes(&proxies, &plain).await, [http("127.0.0.1:9")]);
        assert_eq!(
            routes(&proxies, &to("n.example")).await,
            [http("127.0.0.1:9")]
        );
        let broken = Proxies {
            pac: pac(pac::Source::Script("not a script(".into())),
            fixed: fixed(),
        };
        assert_eq!(
            routes(&broken, &to("h.example")).await,
            [http("127.0.0.1:9")]
        );
        // The script sees the URL Safari would give it.
        let by_url = "function FindProxyForURL(url) { return url == 'https://h.example/' ? 'PROXY 127.0.0.1:3' : 'DIRECT'; }";
        let no_env = |_: &str| None;
        let proxies = Proxies {
            pac: Some(pac::Pac::new(pac::Source::Script(by_url.into()), &no_env)),
            fixed: vec![],
        };
        assert_eq!(
            routes(&proxies, &to("h.example")).await,
            [http("127.0.0.1:3")]
        );
        // A proxy that can't be reached gives way to the next route; one that refuses doesn't.
        let address = |(proxies, _): (Matcher, JoinHandle<_>)| {
            let proxy = proxies
                .intercept(&"https://a.example/".parse().unwrap())
                .unwrap();
            format!(
                "{}:{}",
                proxy.uri().host().unwrap(),
                proxy.uri().port_u16().unwrap()
            )
        };
        let working = address(system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await);
        let refusing = address(system_proxy_answering("HTTP/1.1 407 Proxy Auth\r\n\r\n").await);
        for (first, expected) in [
            (refusing.as_str(), "HTTP/1.1 502"),
            ("127.0.0.1:1", "HTTP/1.1 200"),
        ] {
            let script = format!(
                "function FindProxyForURL() {{ return 'PROXY {first}; PROXY {working}'; }}"
            );
            let proxies = Proxies {
                pac: pac(pac::Source::Script(script.into())),
                fixed: vec![],
            };
            let response = exchange_via(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
            assert!(response.starts_with(expected), "{first}: {response}");
        }
    }
}
