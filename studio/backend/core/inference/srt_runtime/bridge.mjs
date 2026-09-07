import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
export const VERSION = '0.0.75';
export const MAX_REQUEST = 262144;
const fail = (message) => { throw new Error(message); };
const plain = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const string = (value) => typeof value === 'string' && !value.includes('\0');
export const quote = (value) => "'" + value.replaceAll("'", "'\\''") + "'";

export function payloadCommand(request) {
  // Upstream uses localhost, which needs host resolution absent from our root.
  // Set numeric-loopback proxies after bwrap's generated environment overrides.
  const prefix = request.network ? ['/usr/bin/env',
    ...['HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy'].map((key) => `${key}=http://127.0.0.1:3128`),
    'NO_PROXY=localhost,127.0.0.1,::1', 'no_proxy=localhost,127.0.0.1,::1'] : [];
  return [...prefix, request.executable, ...request.argv].map(quote).join(' ');
}

export function validateRequest(value) {
  if (!plain(value) || value.v !== 1 || !['probe', 'run'].includes(value.operation)) fail('Invalid SRT protocol version or operation');
  const keys = new Set(['v', 'operation', 'executable', 'argv', 'cwd', 'env', 'readRoots', 'writeRoots', 'denyReadRoots', 'network', 'timeoutMs', 'controlFd', 'privateUnixSockets']);
  if (Object.keys(value).some((key) => !keys.has(key))) fail('Unknown SRT request field');
  for (const key of ['executable', 'cwd']) if (!string(value[key]) || !path.isAbsolute(value[key])) fail(`${key} must be an absolute path`);
  if (!Array.isArray(value.argv) || value.argv.length > 1024 || !value.argv.every(string)) fail('argv must contain bounded strings');
  if (!plain(value.env) || Object.entries(value.env).some(([k, v]) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(k) || !string(v))) fail('Invalid environment');
  if (Object.keys(value.env).some((key) => /^(LD_|DYLD_|NODE_OPTIONS$|NODE_PATH$|BASH_ENV$|ENV$|SHELLOPTS$|BASHOPTS$|PYTHONSTARTUP$)/i.test(key))) fail('Unsafe loader environment');
  for (const key of ['readRoots', 'writeRoots']) {
    if (!Array.isArray(value[key]) || value[key].length > 128 || !value[key].every((p) => string(p) && path.isAbsolute(p) && p !== '/' && !/[*?\[\]{}]/.test(p))) fail(`${key} must contain explicit absolute paths`);
  }
  if (value.denyReadRoots !== undefined && (!Array.isArray(value.denyReadRoots) || value.denyReadRoots.length > 128 || !value.denyReadRoots.every((p) => string(p) && path.isAbsolute(p) && p !== '/' && !/[*?\[\]{}]/.test(p)))) fail('denyReadRoots must contain explicit absolute paths');
  if (value.network !== undefined) {
    if (!plain(value.network) || Object.keys(value.network).some((key) => !['httpSocketPath','socksSocketPath','socatPath'].includes(key))) fail('Invalid network transport');
    for (const key of ['httpSocketPath','socksSocketPath']) {
      const p=value.network[key];
      // SRT places these paths in its socat command; admit no shell syntax.
      if (!string(p) || !/^\/[A-Za-z0-9_./-]+$/.test(p) || path.normalize(p)!==p) fail('Invalid proxy socket path');
    }
    if (value.network.socatPath !== undefined && (!string(value.network.socatPath) || !path.isAbsolute(value.network.socatPath))) fail('Invalid socat executable');
  }
  if (value.timeoutMs !== null && (!Number.isInteger(value.timeoutMs) || value.timeoutMs < 1 || value.timeoutMs > 86400000)) fail('timeoutMs outside supported range');
  if (value.controlFd !== undefined && (!Number.isInteger(value.controlFd) || value.controlFd < 3 || value.controlFd > 1024)) fail('Invalid control descriptor');
  if (value.privateUnixSockets !== undefined && typeof value.privateUnixSockets !== 'boolean') fail('Invalid private Unix socket setting');
  return value;
}

export function verifyInstallation(root = here) {
  const bytes = fs.readFileSync(path.join(root, 'integrity.json'));
  const manifest = JSON.parse(bytes.toString('utf8'));
  if (createHash('sha256').update(JSON.stringify(manifest)).digest('hex') !== 'f73892a45faac28d621ca1d400e0ad5a2244269bb8a7dbd23e010e5cd39b89e6') fail('SRT integrity manifest has changed');
  if (manifest.version !== VERSION || !plain(manifest.files)) fail('Invalid SRT integrity manifest');
  for (const [relative, digest] of Object.entries(manifest.files)) {
    if (path.isAbsolute(relative) || relative.split('/').includes('..') || !/^[a-f0-9]{64}$/.test(digest)) fail('Invalid integrity entry');
    const filename = path.join(root, 'node_modules', relative);
    const actual = createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
    if (actual !== digest) fail(`SRT dependency integrity mismatch: ${relative}`);
  }
  return path.join(root, 'node_modules', '@anthropic-ai', 'sandbox-runtime');
}

function checkPaths(request) {
  const cwd = fs.realpathSync(request.cwd);
  if (!fs.statSync(cwd).isDirectory()) fail('cwd is not a directory');
  const within = (p, root) => p === root || p.startsWith(root + path.sep);
  const writes = request.writeRoots.map((p) => fs.realpathSync(p));
  if ([cwd, ...writes, ...request.readRoots.map((p) => fs.realpathSync(p))].includes('/tmp')) fail('Host /tmp cannot replace the private temporary root');
  if (!writes.length || writes.some((p) => !within(p, cwd))) fail('Writes must stay within the private workdir');
  if (request.readRoots.some((p) => fs.realpathSync(p) === '/')) fail('Root read grant is forbidden');
  const denied = ['/sys', ...(request.denyReadRoots ?? [])].map((p) => fs.realpathSync(p));
  for (const allowed of [...request.readRoots, ...request.writeRoots]) {
    const resolved = fs.realpathSync(allowed);
    if (denied.some((p) => within(resolved, p))) fail('An allow root would reopen a denied subtree');
  }
  fs.accessSync(request.executable, fs.constants.X_OK);
  return cwd;
}

function checkNetwork(request,cwd) {
  if (!request.network) return {readRoots:[]};
  const {httpSocketPath,socksSocketPath}=request.network;
  if(httpSocketPath===socksSocketPath) fail('HTTP and SOCKS refusal sockets must be distinct');
  const roots=[httpSocketPath,socksSocketPath];
  for(const socketPath of roots) {
    if(fs.realpathSync(socketPath)!==socketPath || socketPath===cwd || socketPath.startsWith(cwd+path.sep)) fail('Proxy socket must be outside the workdir without symlinks');
    const info=fs.lstatSync(socketPath);
    const parent=fs.statSync(path.dirname(socketPath));
    if(!info.isSocket() || info.uid!==process.getuid() || parent.uid!==process.getuid() || (parent.mode&0o777)!==0o700) fail('Proxy socket must belong to this user in a private directory');
    if((request.denyReadRoots??[]).some((p)=>socketPath===p || socketPath.startsWith(p+path.sep))) fail('Proxy socket lies in a denied subtree');
  }
  const socatPath=request.network.socatPath??'/usr/bin/socat';
  try {fs.accessSync(socatPath,fs.constants.X_OK);} catch {fail('HTTPS sandbox transport requires an installed socat executable');}
  const canonical=fs.realpathSync(socatPath);
  if(canonical===cwd || canonical.startsWith(cwd+path.sep) || !fs.statSync(canonical).isFile()) fail('socat must be a trusted executable outside the workdir');
  return {readRoots:[...roots,socatPath,canonical],httpSocketPath,socksSocketPath,socatPath};
}

export async function execute(request, emit) {
  validateRequest(request);
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 20 || (major === 20 && minor < 11)) fail('SRT requires Node.js >=20.11.0');
  if (process.platform !== 'linux') fail('SRT Required is unavailable: this platform has not qualified strict DNS and read isolation');
  if (!['x64', 'arm64'].includes(process.arch)) fail('SRT Unix socket filtering is unavailable for this architecture');
  // The trusted Python launcher installs the exact inherited host-IPC filter.
  // This check is an additional prerequisite, not proof of the filter policy.
  if (request.privateUnixSockets && !/^Seccomp:\s+2\s*$/m.test(fs.readFileSync('/proc/self/status', 'utf8'))) fail('Private Unix sockets require an inherited seccomp filter');
  const cwd = checkPaths(request);
  const network = checkNetwork(request,cwd);
  const packageRoot = verifyInstallation();
  const helper = path.join(packageRoot, 'vendor', 'seccomp', process.arch, 'apply-seccomp');
  fs.accessSync(helper, fs.constants.X_OK);
  const { wrapCommandWithSandboxLinux, cleanupBwrapMountPoints } = await import(path.join(packageRoot, 'dist/sandbox/linux-sandbox-utils.js'));
  // Empty-root SRT supplies a fresh /tmp tmpfs. A short private path also keeps
  // multiprocessing resource_sharer addresses below AF_UNIX's pathname limit.
  const env = { ...request.env, HOME: cwd, TMPDIR: '/tmp', TMP: '/tmp', TEMP: '/tmp' };
  const command = payloadCommand(request);
  const wrapped = await wrapCommandWithSandboxLinux({
    command,
    needsNetworkRestriction: true,
    readConfig: { denyOnly: ['/', '/sys', ...(request.denyReadRoots ?? [])], allowWithinDeny: [...request.readRoots, ...(request.privateUnixSockets ? [] : [helper]), ...network.readRoots] },
    writeConfig: { allowOnly: request.writeRoots, denyWithinAllow: [] },
    binShell: '/bin/bash',
    bwrapPath: '/usr/bin/bwrap',
    seccompConfig: request.privateUnixSockets ? undefined : { applyPath: helper },
    allowAllUnixSockets: request.privateUnixSockets === true,
    enableWeakerNestedSandbox: false,
    httpSocketPath:network.httpSocketPath,
    socksSocketPath:network.socksSocketPath,
    httpProxyPort:network.httpSocketPath?3128:undefined,
    socksProxyPort:network.socksSocketPath?1080:undefined,
    socatPath:network.socatPath,
  });
  emit({ event: 'ready', version: VERSION, backend: 'srt-linux', network: request.network?'https-allowlist':'none', unixSockets: request.privateUnixSockets === true, qualificationComplete: false });
  let child;
  let timer;
  let hardTimer;
  let parentTimer;
  let reason;
  const stop = (why) => {
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    reason ??= why;
    child.kill('SIGTERM');
    hardTimer ??= setTimeout(() => child.kill('SIGKILL'), 1000);
  };
  const onTerm = () => stop('cancelled');
  process.on('SIGTERM', onTerm);
  process.on('SIGINT', onTerm);
  try {
    const result = await new Promise((resolve, reject) => {
      // exec makes bwrap the direct child: its die-with-parent contract covers
      // an uncatchable bridge death, and the workload receives only stdio 0–2.
      child = spawn('/bin/bash', ['-c', `exec ${wrapped}`], { cwd, env, shell: false, stdio: ['ignore', 'inherit', 'inherit'] });
      child.once('error', reject);
      child.once('spawn', () => emit({ event: 'spawned', pid: child.pid, stage: 'sandbox-launcher' }));
      child.once('exit', (code, signal) => resolve({ code, signal }));
      if (request.timeoutMs !== null) timer = setTimeout(() => stop('timeout'), request.timeoutMs);
      const parent = process.ppid;
      parentTimer = setInterval(() => { if (process.ppid !== parent) stop('parent-exit'); }, 250);
    });
    emit({ event: 'exit', code: result.code, signal: result.signal, reason: reason ?? 'completed' });
    return reason ? 124 : (result.code ?? 1);
  } finally {
    clearTimeout(timer); clearTimeout(hardTimer); clearInterval(parentTimer);
    process.removeListener('SIGTERM', onTerm); process.removeListener('SIGINT', onTerm);
    cleanupBwrapMountPoints();
  }
}

async function main() {
  let fd = Number(process.argv[2] ?? 3);
  if (!Number.isInteger(fd) || fd < 3 || fd > 1024) return 2;
  const emit = (record) => fs.writeSync(fd, JSON.stringify({ v: 1, ...record }) + '\n');
  try {
    let size = 0;
    const parts = [];
    for await (const chunk of process.stdin) {
      size += chunk.length;
      if (size > MAX_REQUEST) fail('SRT request exceeds size limit');
      parts.push(chunk);
    }
    const request = validateRequest(JSON.parse(Buffer.concat(parts).toString('utf8')));
    if (request.controlFd !== undefined && request.controlFd !== fd) fail('Control descriptor mismatch');
    return await execute(request, emit);
  } catch (error) {
    try { emit({ event: 'error', message: String(error.message).slice(0, 2048) }); } catch { /* The controller has exited. */ }
    return 125;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await main();
}
