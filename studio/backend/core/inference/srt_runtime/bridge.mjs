import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import net from 'node:net';

const here = path.dirname(fileURLToPath(import.meta.url));
export const VERSION = '0.0.75';
export const MAX_REQUEST = 262144;
// Linux enumerates individual shared libraries to avoid granting their parent trees.
const MAX_READ_ROOTS = 1024;
const fail = (message) => { throw new Error(message); };
const diagnosticError = (code, stage, dependency) => Object.assign(new Error(code), {code, stage, dependency});

export function errorDiagnostic(error) {
  const codes = ['runtime_missing', 'runtime_invalid', 'dependency_missing', 'policy_invalid', 'policy_oversized', 'operation_unsupported', 'probe_timeout', 'enforcement_failed'];
  const counts = ['readRoots','writeRoots','denyReadRoots','denyWriteRoots'].includes(error?.field) && Number.isInteger(error?.count) && Number.isInteger(error?.limit) ? {field:error.field,count:error.count,limit:error.limit} : {};
  return {code: codes.includes(error?.code) ? error.code : 'probe_failed', stage: error?.stage ?? 'probe', ...(error?.dependency ? {dependency:error.dependency} : {}), ...counts};
}
const plain = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const string = (value) => typeof value === 'string' && !value.includes('\0');
export function validateRequest(value) {
  try { return validatePolicy(value); }
  catch (error) { error.code ??= 'policy_invalid'; error.stage = 'policy'; throw error; }
}

function validatePolicy(value) {
  if (!plain(value) || value.v !== 1 || !['probe', 'run'].includes(value.operation)) fail('Invalid SRT protocol version or operation');
  const keys = new Set(['v', 'operation', 'executable', 'argv', 'cwd', 'env', 'readRoots', 'writeRoots', 'denyReadRoots', 'denyWriteRoots', 'nativeAllowedDomains', 'windowsProxyPortRange', 'timeoutMs', 'controlFd', 'controlSocket', 'privateUnixSockets']);
  if (Object.keys(value).some((key) => !keys.has(key))) fail('Unknown SRT request field');
  for (const key of ['executable', 'cwd']) if (!string(value[key]) || !path.isAbsolute(value[key])) fail(`${key} must be an absolute path`);
  if (!Array.isArray(value.argv) || value.argv.length > 1024 || !value.argv.every(string)) fail('argv must contain bounded strings');
  if (!plain(value.env) || Object.entries(value.env).some(([k, v]) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(k) || !string(v))) fail('Invalid environment');
  if (Object.keys(value.env).some((key) => /^(LD_|DYLD_|NODE_OPTIONS$|NODE_PATH$|BASH_ENV$|ENV$|SHELLOPTS$|BASHOPTS$|PYTHONSTARTUP$)/i.test(key))) fail('Unsafe loader environment');
  for (const key of ['readRoots', 'writeRoots', 'denyReadRoots', 'denyWriteRoots']) {
    if (key.startsWith('deny') && value[key] === undefined) continue;
    if (!Array.isArray(value[key])) fail(`${key} must contain explicit absolute paths`);
    const limit = key === 'readRoots' ? MAX_READ_ROOTS : 128;
    if (value[key].length > limit) throw Object.assign(new Error(`${key} exceeds ${limit} entries`), {code:'policy_oversized',field:key,count:value[key].length,limit});
    if (!value[key].every((p) => string(p) && path.isAbsolute(p) && p !== '/' && !/[*?\[\]{}]/.test(p))) fail(`${key} must contain explicit absolute paths`);
  }
  if (value.nativeAllowedDomains !== undefined && (!Array.isArray(value.nativeAllowedDomains) || value.nativeAllowedDomains.length > 256 || !value.nativeAllowedDomains.every((p) => string(p) && p.length > 0 && p.length <= 253))) fail('Invalid native allowed domains');
  if (value.windowsProxyPortRange !== undefined) {
    const ports = value.windowsProxyPortRange;
    if (!Array.isArray(ports) || ports.length !== 2 || !ports.every((p) => Number.isInteger(p) && p >= 1024 && p <= 65535) || ports[1] - ports[0] < 1 || ports[1] - ports[0] > 99) fail('Invalid Windows proxy port range');
  }
  if (value.timeoutMs !== null && (!Number.isInteger(value.timeoutMs) || value.timeoutMs < 1 || value.timeoutMs > 86400000)) fail('timeoutMs outside supported range');
  if (value.controlFd !== undefined && (!Number.isInteger(value.controlFd) || value.controlFd < 3 || value.controlFd > 1024)) fail('Invalid control descriptor');
  if (value.controlSocket !== undefined) {
    const control = value.controlSocket;
    if (!plain(control) || Object.keys(control).some((key) => !['port','token'].includes(key)) || !Number.isInteger(control.port) || control.port < 1 || control.port > 65535 || !/^[a-f0-9]{64}$/.test(control.token)) fail('Invalid authenticated control socket');
    if (value.controlFd !== undefined) fail('Only one control transport is permitted');
  }
  if (value.privateUnixSockets !== undefined && typeof value.privateUnixSockets !== 'boolean') fail('Invalid private Unix socket setting');
  return value;
}

export function verifyInstallation(root = here) {
  try { return verifyInstalledFiles(root); }
  catch (error) { error.code = error.code === 'ENOENT' ? 'runtime_missing' : 'runtime_invalid'; error.stage = 'installation'; throw error; }
}

function verifyInstalledFiles(root) {
  const bytes = fs.readFileSync(path.join(root, 'integrity.json'));
  const manifest = JSON.parse(bytes.toString('utf8'));
  if (createHash('sha256').update(JSON.stringify(manifest)).digest('hex') !== '6b98645137b11cc1dc3e64faf79edb45b2bc6ff185a2787221b6811a5b1a3b83') fail('SRT integrity manifest has changed');
  if (manifest.version !== VERSION || !plain(manifest.files)) fail('Invalid SRT integrity manifest');
  for (const [relative, digest] of Object.entries(manifest.files)) {
    if (path.isAbsolute(relative) || relative.split('/').includes('..') || !/^[a-f0-9]{64}$/.test(digest)) fail('Invalid integrity entry');
    const filename = path.join(root, 'node_modules', relative);
    const actual = createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
    if (actual !== digest) fail(`SRT dependency integrity mismatch: ${relative}`);
  }
  return path.join(root, 'node_modules', '@anthropic-ai', 'sandbox-runtime');
}

export function supportedConfig(request, windowsHelper) {
  const config = {
    network: { allowedDomains: request.nativeAllowedDomains ?? [], deniedDomains: [] },
    filesystem: { allowRead: request.readRoots, denyRead: request.denyReadRoots ?? [], allowWrite: request.writeRoots, denyWrite: request.denyWriteRoots ?? [] },
  };
  if (windowsHelper) config.windows = { srtWin: { path: windowsHelper }, ...(request.windowsProxyPortRange ? {proxyPortRange:request.windowsProxyPortRange} : {}) };
  return config;
}

export async function supportedArgv(manager, request, platform, signal) {
  if (platform !== 'win32') {
    const command = [request.executable, ...request.argv].map(quote).join(' ');
    return manager.wrapWithSandboxArgv(command, '/bin/bash', undefined, signal, request.cwd);
  }
  // The public object form accepts a native executable and argument vector.
  // Its command parameter becomes the last native argument, not shell source.
  const final = request.argv.at(-1) ?? '';
  const result = await manager.wrapWithSandboxArgv(final, {exe:request.executable,args:request.argv.slice(0,-1)}, undefined, signal, request.cwd);
  if (!request.argv.length) {
    if (result.argv.at(-1) !== '') fail('Unexpected SRT empty-argument wrapper');
    result.argv = result.argv.slice(0,-1);
  }
  const boundary = result.argv.indexOf('--');
  if (boundary < 1) fail('Unexpected Windows SRT command wrapper');
  // Documented srt-win exec --env overlays preserve the supplied safe runtime
  // settings. Its own proxy/CA settings and isolated Windows profile win.
  const reserved = /^(?:.*_PROXY|NO_PROXY|USERPROFILE|HOMEDRIVE|HOMEPATH|TEMP|TMP|TMPDIR|SSL_CERT_FILE|SSL_CERT_DIR|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|GIT_SSL_CAINFO|NODE_EXTRA_CA_CERTS|CARGO_HTTP_CAINFO|GIT_CONFIG_.*)$/i;
  const overlay = Object.entries(request.env).filter(([key]) => !reserved.test(key)).flatMap(([key,value]) => ['--env',`${key}=${value}`]);
  return {...result,argv:[...result.argv.slice(0,boundary),...overlay,...result.argv.slice(boundary)]};
}

export async function executeSupported(request, emit, options = {}) {
  const platform = options.platform ?? process.platform;
  const loadManager = options.loadManager ?? (async () => {
    const root = verifyInstallation();
    return import(pathToFileURL(path.join(root,'dist/index.js')).href);
  });
  const spawnProcess = options.spawnProcess ?? spawn;
  if (platform !== 'win32') fail('Unsupported SRT platform');
  if (request.network) fail('Native platforms use SRT domain policy, not Linux proxy sockets');
  const cwd = fs.realpathSync(request.cwd);
  const within = (p,root) => p === root || p.startsWith(root + path.sep);
  if (!fs.statSync(cwd).isDirectory() || !request.writeRoots.length || request.writeRoots.some((p) => !within(fs.realpathSync(p),cwd))) fail('Writes must stay within the tool workdir');
  fs.accessSync(request.executable,fs.constants.X_OK);
  let manager, child, timer, hardTimer, parentTimer, reason;
  const abort = new AbortController();
  const stop = (why) => {
    reason ??= why; abort.abort();
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    const kill = (signal) => {
      try { if (platform === 'darwin') process.kill(-child.pid,signal); else child.kill(signal); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    };
    kill('SIGTERM'); hardTimer ??= setTimeout(() => kill('SIGKILL'),1000);
  };
  const onTerm = () => stop('cancelled');
  const onControlClose = () => stop('controller-exit');
  process.on('SIGTERM',onTerm); process.on('SIGINT',onTerm);
  options.signal?.addEventListener('abort',onControlClose,{once:true});
  // Workload descriptors are inherited at OS level; suppress only host-side
  // SRT JavaScript logging so it cannot masquerade as workload output.
  const originals = [process.stdout.write,process.stderr.write];
  const discard = (_chunk,encoding,callback) => { (typeof encoding === 'function' ? encoding : callback)?.(); return true; };
  process.stdout.write = discard; process.stderr.write = discard;
  let result;
  try {
    if (options.signal?.aborted) fail('Controller closed before sandbox setup');
    const api = await loadManager(); manager = api.SandboxManager;
    const config = supportedConfig(request,platform === 'win32' ? api.VENDORED_SRT_WIN_EXE : undefined);
    if (options.readLease) {
      if (platform !== 'win32' || options.readLease.socket.destroyed) fail('Read lease is unavailable');
      const held = new Set(options.readLease.readRoots.map(root=>fs.realpathSync(root).toLowerCase()));
      config.filesystem.allowRead = config.filesystem.allowRead.filter(root=>!held.has(fs.realpathSync(root).toLowerCase()));
    }
    await manager.initialize(config,undefined,false);
    if (abort.signal.aborted) fail('Sandbox setup was cancelled');
    const wrapped = await supportedArgv(manager,request,platform,abort.signal);
    if (abort.signal.aborted) fail('Sandbox launch was cancelled');
    emit({event:'ready',version:VERSION,backend:`srt-${platform}`,network:request.nativeAllowedDomains?.length?'native-allowlist':'none',unixSockets:false,qualificationComplete:false});
    result = await new Promise((resolve,reject) => {
      child = spawnProcess(wrapped.argv[0],wrapped.argv.slice(1),{
        cwd,env:platform === 'win32' ? wrapped.env : request.env,
        shell:false,windowsHide:true,detached:platform === 'darwin',stdio:['ignore','inherit','inherit'],
      });
      child.once('error',reject);
      child.once('spawn',() => emit({event:'spawned',pid:child.pid,stage:'sandbox-launcher'}));
      child.once('exit',(code,signal) => resolve({code,signal}));
      if (request.timeoutMs !== null) timer = setTimeout(() => stop('timeout'),request.timeoutMs);
      const parent = process.ppid;
      parentTimer = setInterval(() => { if (process.ppid !== parent) stop('parent-exit'); },250);
    });
  } finally {
    clearTimeout(timer);clearTimeout(hardTimer);clearInterval(parentTimer);
    process.removeListener('SIGTERM',onTerm);process.removeListener('SIGINT',onTerm);
    options.signal?.removeEventListener('abort',onControlClose);
    try {
      try { if (child && platform === 'darwin') process.kill(-child.pid,'SIGKILL'); }
      catch (error) { if (error.code !== 'ESRCH') throw error; }
      finally {
        if (manager) {
          try { manager.cleanupAfterCommand(); }
          finally { await manager.reset(); }
        }
      }
    }
    finally { process.stdout.write = originals[0];process.stderr.write = originals[1]; }
  }
  emit({event:'exit',code:result.code,signal:result.signal,reason:reason ?? 'completed'});
  return reason ? 124 : (result.code ?? 1);
}

export async function execute(request, emit, options = {}) {
  validateRequest(request);
  const [major,minor] = process.versions.node.split('.').map(Number);
  if (major < 20 || (major === 20 && minor < 11)) throw diagnosticError('dependency_missing', 'dependency', 'node');
  if (process.platform !== 'win32') fail('Studio SRT is Windows-only');
  return executeSupported(request,emit,options);
}

export async function connectControl(control) {
  const connection = net.createConnection({host:'127.0.0.1',port:control.port});
  await new Promise((resolve,reject) => {
    connection.once('connect',resolve);connection.once('error',reject);
    connection.setTimeout(5000,() => connection.destroy(new Error('Control connection timed out')));
  });
  connection.setTimeout(0);
  connection.write(JSON.stringify({v:1,event:'hello',token:control.token})+'\n');
  return connection;
}

async function main() {
  const socketMode = process.argv[2] === '--control-socket';
  const fd = socketMode ? undefined : Number(process.argv[2] ?? 3);
  if (!socketMode && (!Number.isInteger(fd) || fd < 3 || fd > 1024)) return 2;
  let connection, readLease;
  const abort = new AbortController();
  const emit = (record) => {
    const data = JSON.stringify({v:1,...record})+'\n';
    if (connection) connection.write(data); else if (!socketMode) fs.writeSync(fd,data);
  };
  try {
    let size = 0;
    const parts = [];
    for await (const chunk of process.stdin) {
      size += chunk.length;
      if (size > MAX_REQUEST) throw diagnosticError('policy_oversized', 'policy');
      parts.push(chunk);
    }
    let value;
    try {value=JSON.parse(Buffer.concat(parts).toString('utf8'));}
    catch {throw diagnosticError('policy_invalid','policy');}
    const readLeaseTransport = value?.readLeaseTransport;
    if (value && typeof value === 'object') delete value.readLeaseTransport;
    const request = validateRequest(value);
    if (socketMode) {
      if (!request.controlSocket) fail('Authenticated control socket is required');
      connection = await connectControl(request.controlSocket);
      connection.on('error',() => abort.abort());connection.on('close',() => abort.abort());
    } else if (request.controlSocket || (request.controlFd !== undefined && request.controlFd !== fd)) fail('Control descriptor mismatch');
    if (readLeaseTransport) {
      const {connectReadLease} = await import('./windows-read-lease.mjs');
      readLease = await connectReadLease(readLeaseTransport,request,abort);
    }
    return await execute(request, emit, {signal:abort.signal,readLease});
  } catch (error) {
    try { emit({ event: 'error', ...errorDiagnostic(error), message: 'SRT setup failed' }); } catch { /* The controller has exited. */ }
    return 125;
  } finally {
    readLease?.socket.destroy();
    if (connection && !connection.destroyed) await new Promise((resolve) => connection.end(() => { connection.destroy(); resolve(); }));
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await main();
}
