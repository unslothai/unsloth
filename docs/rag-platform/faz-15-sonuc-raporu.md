# Faz 15 Sonuç Raporu

## Faz durumu

**BLOCKED**

Faz 15'in uygulama, statik kalite ve yerel runtime kapsamı tamamlandı. Ayrı
etiketli `rag-platform-backend:0.26.4-phase15-local-smoke` imajı sağlıklı;
same-origin güvenlik başlıkları ve Faz 7–15 tam runtime smoke paketi geçiyor.
Global Definition of Done yalnız iki dış release koşulu yüzünden tamamlanmış
sayılmaz:

1. Backend çalışma ağacı kullanıcıya ait commitlenmemiş değişiklikler içeriyor.
   Production build kapısı temiz ve protected/tagged backend commit'i zorunlu
   tuttuğu için release imajını bilinçli olarak reddediyor.
2. Güvenilir TLS edge arkasında gerçek bir HTTPS canary URL sağlanmadı. Yerel
   HTTP doğrulaması release TLS kapısının yerine geçirilemez.

Bu iki koşulu aşmak için commit/push yapılmadı ve güvenlik kapıları gevşetilmedi.

## Yapılan değişiklikler

- Route inventory modeli düzeltildi: Python/MCP tabanı `v0.26.4`, Go API/admin
  ise yerel backend otoritesinden derlenen owned runtime olarak ayrı provenance
  ile taranıyor. Güncel Go route'ları artık yanlış biçimde source-only/404
  gösterilmiyor.
- Faz 14 baseline'ına karşı gözden geçirilmiş Faz 15 drift artifact'i üretildi;
  stale veya onaysız drift CI'ı durduruyor.
- 801 endpoint/alternate kaydı release validator'ından geçirildi. Eksik,
  mükerrer, sınıflandırılmamış, yarım ve erişilebilir `unsupported` kayıt
  sayıları sıfırlandı; sahte eksik route negatif testi eklendi.
- Eksik pipeline GET ve güncel Go document/structure/stop route kanıtları;
  root, health, live, language ve not-implemented kullanıcı meta route
  sınıflandırmaları matrise eklendi.
- Settings > Data altında güvenli legacy Project/Thread migration ve export UI'ı
  uygulandı: dry-run, ilerleme, abort, resume, cleanup ve taşınamayan alan
  raporu. Kaynak veri silinmiyor; token/secret ledger veya export'a yazılmıyor.
- Legacy metadata alias'ları typed `api-only` adapter ve exact contract
  testleriyle kapsandı.
- Owned frontend bundle, dört servisli readiness/healthcheck, CSP/HSTS ve diğer
  güvenlik başlıkları, SSE/upload proxy sınırları, wildcard CORS engeli ve
  production release/rollback otomasyonu tamamlandı.
- Nginx'in upstream `proxy.conf` ile çakışan directive'leri ayrıştırıldı;
  security header include'ları API ve SPA location'larında kalıcı hale getirildi.
- Production build temiz protected kaynak zorunluluğunu koruyor. Yalnız açıkça
  istenen `RAG_PLATFORM_LOCAL_SMOKE=true` yolu dirty değişiklikleri ayrı,
  non-release etiketli imaja overlay ederek yerel doğrulama sağlıyor.
- Faz 10, 11 ve 12 smoke sözleşmeleri gerçek güncel Go router/callback
  contract'larına düzeltildi; Faz 15 smoke'a `/healthz`, `/live` ve
  `/api/v1/language` eklendi.
- CI; kalite, coverage, 24 zorunlu senaryo kanıtı, dependency/secret/license,
  branding/performance, image vulnerability, SBOM ve provenance işlerini
  içeriyor. Taranan imaj commit etiketiyle GHCR'a push ediliyor; HTTPS canary
  yalnız aynı imajın immutable digest'ini pull edip çalıştırıyor.

## Değiştirilen dosyalar

### Frontend ve migration

- `studio/frontend/src/features/migration/*`
- `studio/frontend/src/features/settings/tabs/data-tab.tsx`
- `studio/frontend/src/features/chat/api/chat-api.ts`
- `studio/frontend/src/integrations/platform-backend/advanced-dataset-api.ts`
- `studio/frontend/src/integrations/platform-backend/__tests__/advanced-dataset-api.test.ts`
- `studio/frontend/src/integrations/platform-backend/__tests__/phase-15-performance.test.ts`
- `studio/frontend/package.json`
- `studio/frontend/vite.config.ts`

### Runtime, inventory ve release

- `infra/rag-platform/Dockerfile.backend-with-go`
- `infra/rag-platform/build-backend-image.sh`
- `infra/rag-platform/docker-compose.rag-platform.yml`
- `infra/rag-platform/rag-platform.hybrid.conf`
- `infra/rag-platform/proxy.conf`
- `infra/rag-platform/security-headers.conf`
- `infra/rag-platform/runtime-readiness.sh`
- `scripts/rag-platform/{proxy-config,route-inventory,route-inventory-diff}.mjs`
- `scripts/rag-platform/{coverage-matrix,coverage-release-validator,coverage-negative-test}.mjs`
- `scripts/rag-platform/{backend-coverage-report,e2e-evidence,performance-gate}.mjs`
- `scripts/rag-platform/{release-security-gate,secret-scan,phase-15-runtime-smoke}.mjs`
- `scripts/rag-platform/phase-{10,11,12}-runtime-smoke.mjs`
- `.github/workflows/phase-15-release.yml`
- `.github/workflows/backend-image-security.yml`

### Kanıt ve yönetişim

- `docs/adr/0018-phase-15-security-migration-and-release-boundary.md`
- `docs/rag-platform/{route-inventory,endpoint-coverage-matrix,runtime-disabled}.*`
- `docs/rag-platform/route-inventory-phase{14-baseline,15-diff}.json`
- `docs/rag-platform/backend-coverage-report.md`
- `docs/rag-platform/phase-15-e2e-evidence.json`
- `docs/rag-platform/{production-release-runbook,version-compatibility}.md`
- `docs/rag-platform/license-baseline.json`
- `docs/maintenance/upstream-sync.md`
- `THIRD_PARTY_NOTICES.md`

Backend repository'sinde Faz 15 tarafından kaynak dosya değiştirilmedi; mevcut
dirty durum kullanıcıya ait haliyle korundu.

## Eklenen frontend ekranları ve aksiyonları

- Settings > Data > **Migration ve güvenli export**
  - Legacy kaynağı dry-run tarama
  - Yerel/custom veriyi indirme
  - Desteklenen Project → Chat ve Thread → Session migration'ını başlatma
  - Kısmi hata sonrası devam/resume
  - Çalışan işlemi abort etme ve cleanup
  - Mesaj ve taşınamayan archive/fork/local sandbox/container alanlarını
    export'ta koruyup açıkça raporlama

## Kullanılan backend endpoint'leri

- `GET /api/v1/chats`
- `POST /api/v1/chats`
- `POST /api/v1/chats/{chat_id}/sessions`
- `POST /api/v1/document/metadata/summary`
- `POST /api/v1/document/set_meta`
- `GET /api/v1/system/ping`, `GET /api/v1/admin/ping`, `GET /health`
- `GET /healthz`, `GET /live`, `GET /api/v1/language`

Opsiyonel `/api/chat/export` yalnız legacy custom frontend kaynağıdır; Rag
Platform backend endpoint'i değildir. Backend'de geçmiş Session mesajı ekleme
contract'ı bulunmadığı için alan veya endpoint tahmin edilmedi.

## Route coverage sonuçları

- Top-level inventory: **692**
- Coverage kayıtları (alternates dahil): **801**
- Faz 14 → Faz 15 diff: **28 added / 117 removed / 285 changed**
- Runtime-enabled: **475**
- Top-level runtime-disabled: **212**; erişilebilir eşdeğeri olmayan: **11**
- Coverage runtime-disabled: **320**; not-proxied: **6**
- `frontend-screen`: **66**
- `frontend-action`: **318**
- `api-only`: **81**
- `external-callback`: **10**
- `internal`: **9**
- `unsupported`: **317**; erişilebilir `unsupported`: **0**
- `planned` / `in-progress` / `unclassified`: **0**
- Coverage, contract, report drift ve negatif test kapıları: **PASS**

## Çalıştırılan komutlar ve sonuçları

| Komut / kapı | Sonuç |
| --- | --- |
| `npm run typecheck` | PASS |
| `npm run lint:all` | PASS — 0 error, mevcut 78 warning |
| `npm run i18n:check:strict` | PASS |
| `npm run test:phase15 -- --maxWorkers=2 --testTimeout=30000` | PASS — 3 dosya, 7 test |
| `npm test -- --maxWorkers=2 --testTimeout=30000` | PASS — 77 dosya, 265 test |
| `npm run build` | PASS |
| `npm audit --audit-level=high` | PASS — 0 vulnerability |
| Branding source/build taraması | PASS |
| Performance gate | PASS — 527 JS chunk, 5,278,309 gzip byte |
| Proxy, route, coverage, contract ve report drift kapıları | PASS |
| Coverage negatif testi | PASS — 3 beklenen ihlal yakalandı |
| 24 senaryolu E2E evidence manifest | PASS |
| Secret, license ve statik release security | PASS |
| Workflow YAML, shell syntax ve Compose config/profile | PASS |
| Local-smoke image build | PASS — Go test/build dahil |
| Container healthcheck ve `nginx -t` | PASS |
| Faz 7–15 `--full` runtime smoke | PASS |
| CSP, HSTS, nosniff, frame/referrer/permissions ve CORS kontrolleri | PASS |

## Başarısız testler

Yerel/statik test başarısızlığı yoktur. Aşağıdaki iki production gate beklenen
şekilde kapalıdır ve fazın `BLOCKED` kalma nedenidir:

- `release-security-gate.mjs --runtime`: HTTPS canary yerine HTTP URL verildiğinde
  exit 1 — `RAG_PLATFORM_PUBLIC_URL must be an https:// URL for release`.
- `build-backend-image.sh`: dirty backend ile exit 1 — release images require a
  clean protected commit.

## Bilinen sınırlamalar

- Local tenant'ta gerçek provider credential bulunmadığından Faz 8
  provider-success chat/voice zinciri koşturulamadı; auth, stream ve hata
  contract'ları geçti.
- GitHub-hosted Trivy/SBOM/provenance ve self-hosted HTTPS canary işleri workflow
  olarak tanımlı, fakat commit/push ve dış canary yetkisi olmadan yerelde icra
  edilemez.
- Session create contract'ında metadata/idempotency-key yoktur. Session resume
  mapping'i kullanıcıya özel browser ledger'ında tutulur; geçmiş mesajlar
  backend'e yazılmaz, export'ta korunur.

## Runtime-disabled kayıtlar ve kanıtları

- **11** eşdeğersiz kayıt: `CodeNotImplemented` döndüren 7 auth ve 2 users/me
  route'u ile pinned Python tabanında bulunmayan 2 navigation route'u.
- Diğer disabled kayıtlar aynı method+path'i owned Go/Python hedefinde sunan
  alternates veya kaynakla doğrulanmış işlevsel runtime sınırlarıdır.
- Global skill UI, yerel runtime'da eksik `skill_spaces` şeması nedeniyle açık
  runtime-disabled gerekçesi gösterir; kontroller sessizce sunulmaz.
- Flat document list/read güvenlik-contract sınırları ve Go upload/parse
  alternates için kaynak, proxy ve smoke kanıtları korunur.

Ayrıntılı source/proxy/smoke kanıtı `runtime-disabled.md`, typed service/UI/test
kanıtı `endpoint-coverage-matrix.md` içindedir.

## Sonraki faza geçiş

**Güvenli değildir.** Önce backend değişiklikleri kullanıcı tarafından review
edilip clean protected commit/tag'e alınmalı; production image bu ref'ten
üretilmeli; image scan/SBOM/provenance ve gerçek HTTPS canary üzerinde tam
runtime paketi geçmelidir. Faz 16 kapsamına başlanmadı.
