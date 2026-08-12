# Faz 0 Sonuç Raporu

> Durum: **COMPLETE — FAZ 0 KABUL KRİTERLERİ PASS.**
> Faz 1 başlatılmadı. Yerel teknik kapılar ve GitHub owner kontrolleri
> doğrulandı; commit veya push yapılmadı.

## Tamamlananlar

- Tek `Rag Platform` marka sözleşmesi, source/build branding denetimi ve gerekli
  white-label metinleri tamamlandı.
- Frontend/backend repository sınırları, origin/upstream düzeni, fetch-only
  upstream koruması, tam geçmiş ve provenance doğrulandı.
- Sahip olunan Compose katmanı; project/container/service/network/volume adları,
  CPU/GPU mutual-exclusion guard'ı ve `API_PROXY_SCHEME=hybrid` hedefi kuruldu.
- Upstream image'da eksik olan Go server, doğrulanmış `v0.26.4` arşivinden
  `CGO_ENABLED=0` profiliyle tekrar üretildi. Unicode tokenizer fallback'i build
  aşamasında unit test edilir; opsiyonel native PDF sayfa-sayısı fallback'i
  Python parse servisine bırakılır.
- Go'nun zorunlu NATS bağımlılığı profile eklendi. Python API/admin, Go
  API/admin ve Go ingestor aynı yığında kalıcı olarak çalışıyor.
- Static upstream PEM çifti backend çalışma ağacından ve image katmanından
  kaldırıldı. RSA-2048 çift runtime'da `rag-platform-key-material` volume'unda
  üretiliyor; izin, geçerlilik ve pair-match testleri geçiyor.
- Route inventory artık hareketli backend `main` yerine çalışan image ile aynı
  `v0.26.4` ref/commit'ini tarıyor. Method+path proxy haritası 14 Python
  specificity override'ıyla genel Go regex'lerinin daha özel Python route'larını
  gölgelemesini engelliyor.
- Route inventory, endpoint coverage, runtime-disabled ve contract çıktıları
  yeniden üretildi; `unclassified=0`, proxy kaynaklı capability loss 0.
- Frontend ortak lint hatalarının 38'i giderildi. Uyarılar kalite kapısını
  düşürmüyor; `npm run lint:all` çıkış kodu 0.
- Her iki GitHub fork'unda `main` koruması, PR/onay zorunluluğu, required CI
  checks, secret scanning/push protection ve Dependabot etkinleştirildi.
- Frontend ve backend required check workflow'ları çalışma ağacına eklendi:
  `.github/workflows/phase-0.yml` ve
  `.github/workflows/rag-platform-governance.yml`.

## Repository ve provenance

| Kontrol | Sonuç |
|---|---|
| Frontend branch | `feature/rag-platform-phase-0` |
| Frontend origin / upstream | `acrbaran/rag-frontend` / `unslothai/unsloth` |
| Backend branch | `main` |
| Backend HEAD / origin/main | `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2` / aynı commit |
| Backend origin / upstream | `acrbaran/rag-backend` / `infiniflow/ragflow` |
| Runtime source ref / commit | `v0.26.4` / `cb93883f3f8c975eecb2fed81210effeb3bdb06f` |
| Upstream image digest | `sha256:16d24d1968ab59e2715a85d2590f1569c9539e0362344a42f3a23e8be06a655b` |
| Derived image ID | `sha256:fe17fda6fb5a1e244fd9a081d44ae8b9e0af320403df15e71f2e55c509586f71` |
| Go build profile | pure Go, `CGO_ENABLED=0`, `linux/amd64` |

Her iki çalışma ağacı dirty'dir. Backend'deki değişiklikler `.gitignore`, static
PEM silmeleri ve bunların runtime volume karşılığıdır. Commit, stage veya push
yapılmadı; kullanıcı değişiklikleri geri alınmadı.

## Endpoint coverage

- Inventory: **700** top-level route.
- Servisler: Go admin 114, Python admin 34, Python API 304, Go API 243, MCP 5.
- Coverage: **810** kayıt (110 alternatif implementasyon dahil).
- Aktif proxy üzerinden erişilebilir: **516**.
- Runtime-disabled implementasyon: **179**; tamamında erişilebilir eşdeğer var.
- Proxy kaynaklı kaybedilen capability: **0**.
- Sınıflar: `frontend-screen` 64, `frontend-action` 245, `api-only` 41,
  `external-callback` 10, `unsupported` 450; **unclassified 0**.
- Durumlar: contract-verified 25, planned 491, runtime-disabled implementasyon
  288, not-proxied 6.

## Test kanıtı

| Komut/kontrol | Sonuç | Kanıt/not |
|---|---|---|
| `npm run lint:all` | PASS | 0 hata, 78 uyarı, exit 0. |
| `npm run typecheck` | PASS | TypeScript project build temiz. |
| `npm run i18n:check` | PASS | Locale parity temiz. |
| `npm run build` | PASS | Vite production build tamamlandı. |
| Branding source/build audit | PASS | 1105 TS/TSX; 6 gerekçeli allowlist kuralı. |
| Pure-Go analyzer unit test | PASS | `go test ./internal/binding`. |
| Derived image build / `--help` | PASS | Binary flag parser'a SIGSEGV olmadan ulaşıyor. |
| Image PEM absence | PASS | `/ragflow/conf` image katmanında PEM yok. |
| Runtime key | PASS | Volume, 0700/0600/0644, RSA check, pair match, symlink. |
| Proxy config `--check` | PASS | 357 Go route + 14 Python specificity override. |
| Route inventory `--check` | PASS | 700 route, pinned ref/commit. |
| Coverage matrix `--check` | PASS | 810 kayıt, `unclassified=0`, capability loss 0. |
| Contract matrix `--check` | PASS | 272 frontend/backend çifti. |
| Fixture JSON/secret kontrolü | PASS | 9 JSON fixture parse temiz; yüksek güvenli secret eşleşmesi yok. |
| Compose config/profile guard | PASS | CPU config çözülüyor; CPU+GPU birlikte reddediliyor. |
| Dört servis direct smoke | PASS | 9380 ping/version 200; 9381 admin ping 200; 9383 admin ping/version 200; 9384 health 200. |
| Hybrid proxy smoke | PASS | Python-only 200; Go-only direct/proxy 401 ve byte-identical; iki specificity route'u direct/proxy byte-identical. |
| Nginx config | PASS | `nginx -t`. |
| LICENSE/attribution | PASS | Frontend root/AGPL ve backend LICENSE upstream ile byte-identical. |
| `git diff --check` | PASS | Her iki repository. |
| GitHub governance API readback | PASS | İki fork'ta protected `main`, PR + 1 onay, strict required checks, secret scanning/push protection ve Dependabot doğrulandı. |
| Backend image release CI | PASS | Protected `v0.26.4` commit doğrulaması, SBOM, HIGH/CRITICAL scan ve provenance workflow'u mevcut. |
| `npm run test` | UYGULANMAZ | Paket test script'i/test runner'ı Faz 1 kapsamındadır. |

## Secret ve veri kapısı

- Backend `conf/private.pem` ve `conf/public.pem` teslim ağacından silindi ve
  ignore edildi; derived image da upstream kopyalarını kaldırıyor.
- Tracked-content yüksek güvenli taramasında yalnızca açıkça `fake` gövdeli SSH
  unit-test PEM'i ve frontend locale'deki sahte örnek API anahtarı kaldı; gerçek
  credential/token/provider secret bulunmadı.
- Fixture'larda, log/config bind alanlarında ve sahip olunan deployment
  dosyalarında gerçek secret yok. Runtime anahtarı yalnız named volume'da.
- Gitleaks/TruffleHog/detect-secrets kurulu değildi; desen taraması, JSON parse,
  image-layer ve runtime key kontrolleri ayrı ayrı uygulandı.

## Kabul kriterleri

- [x] Görünen ürün adı ve build artifact yalnız `Rag Platform`.
- [x] Repository/remote/provenance ve `main == origin/main` doğrulandı.
- [x] Git'e gönderilecek değişiklik ağacında gerçek secret/static private key yok.
- [x] Sahip olunan Docker adları, CPU/GPU guard ve image alias doğru.
- [x] Hybrid proxy ve 9380/9381/9383/9384 readiness PASS.
- [x] Runtime image ref'i ile route inventory ref'i aynı.
- [x] `unclassified=0`, capability loss 0; P0 fixture/ADR/contract kanıtları mevcut.
- [x] Project → Chat, Thread → Session ve refresh-token kararları yazılı.
- [x] Upstream lisans/telif/attribution korunuyor.
- [x] GitHub branch protection, PR zorunluluğu, secret scanning/push
  protection, Dependabot ve required CI kontrolleri owner erişimiyle uygulandı
  ve sunucu tarafında API readback ile kanıtlandı.

## Bilinen sınırlamalar

- Upstream image yalnız `linux/amd64`; Apple Silicon'da runtime emülasyonla
  çalışır. Native C++ analyzer bu ortamda process-start crash ürettiği için Go
  server pure-Go fallback kullanır. Python servisi tam native document parsing
  yüzeyini korur; Go fallback tokenizer gelişmiş sözlük/POS frekans semantiğini
  sağlamaz ve compressed-PDF native page-count fallback'i kullanmaz.
- MCP portu 9382 nginx yüzeyi değil, doğrudan ve opt-in protokol servisidir.
- Coverage'daki planned kayıtlar Faz 1-14'e aittir; Faz 0'da uygulanmaları kapsam
  ihlali olurdu.

## Rollback

1. `infra/rag-platform/platform-compose.sh --cpu stop` ile owned stack'i durdur.
2. Önceki image ID'sini tekrar tag'le veya alias'ı build script'iyle yeniden üret.
3. Proxy kararını yalnız ADR 0005 ile birlikte değiştir; inventory → proxy →
   inventory → coverage/contract sırasını yeniden çalıştır.
4. Volume silme veya backend repository reset yapma. Bu çalışma hiçbir veri
   volume'unu silmedi.

## GitHub governance kanıtı

Sunucu tarafı readback özeti
`docs/rag-platform/github-governance-evidence.json` içindedir. Her iki fork'ta
`main` için strict required checks, PR + bir onay, stale-review dismissal,
conversation resolution, admin enforcement, linear history, force-push ve
deletion yasağı doğrulandı. Secret scanning, push protection, Dependabot alerts
ve security updates etkindir. Required check workflow'ları çalışma ağacındadır;
talimat gereği commit veya push yapılmadı.
