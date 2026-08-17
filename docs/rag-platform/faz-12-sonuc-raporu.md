# Faz 12 Sonuç Raporu

- Tarih: 2026-08-17
- Durum: **COMPLETE**
- Kapsam: Connector/OAuth, dosya ve klasör yönetimi, file commit geçmişi;
  gerçek PDF yükleme/indeksleme ve chat regresyonlarının Faz 12 E2E doğrulaması
- Frontend branch: `feature/rag-platform-phase-12`
- Backend değişikliği: Var; retrieval sonuç kimliğini koruyan düzeltme ile file
  commit rename/move metadata algılama ve saklama düzeltmeleri. Kullanıcının
  mevcut backend değişiklikleri korunmuştur.

## Ön koşul doğrulaması

Faz 0–11 kodu, ADR'leri, route inventory, endpoint coverage matrix, test
kanıtları ve Faz Sonuç Raporları incelendi. Faz 12'ye başlamayı engelleyen önceki
faz eksiği bulunmadı. Her iki worktree başlangıçta ve sonuçta incelendi; kullanıcı
değişiklikleri silinmedi, geri alınmadı veya ezilmedi. Commit/push yapılmadı.

## Yapılan değişiklikler

- `/files` ürün yolu; dosya/folder CRUD, arama, breadcrumb, upload/download,
  move/rename, parent/ancestors, dataset linkleme ve sürüm geçmişi eklendi.
- Connector CRUD/test/rebuild/log ekranı ve backend'in desteklediği kaynak
  seçenekleri eklendi. Rebuild artık seçili connector'ı dataset'in canonical
  `connectors` sözleşmesiyle bağlar, mevcut bağları korur, sonra rebuild eder.
  Bu düzeltme worker ve log sorgusunun ihtiyaç duyduğu `connector2kb` kaydını
  oluşturur.
- Google Drive, Gmail ve Box OAuth start/callback/result akışı; state/flow
  korelasyonu, popup/full-page köprüsü ve secret redaction eklendi.
- Workspace/folder/dataset commit create/list/detail/files/diff/tree/content ve
  uncommitted changes UI'si eklendi. İçerik 1 MiB/metin güvenlik sınırlarıyla
  typed dosya adapter'ından okunur.
- Loading, empty, error, permission, timeout, abort, cleanup, pagination,
  virtualization ve destructive confirmation durumları uygulandı.
- Hybrid proxy'de OAuth start/result Go `9384`, connector test/rebuild ve file
  parent Python `9380` sınırları açıkça kaydedildi. Yenilenen hibrit config aktif
  Nginx kopyasına alındı ve reload edildi; unprefixed callback gerçek runtime'da
  `200` oldu.
- Elasticsearch bellek sınırı 2 GiB, JVM heap 1 GiB yapıldı; OOM/restart olmadan
  stabil çalıştı.
- Chat adapter'ı boş retrieval yanıtında genel selamlaşma fallback'ine izin
  verecek, dataset bağlamında kaynaklı cevap verecek ve eski sert no-match sistem
  prompt'unu güvenli ürün prompt'una taşıyacak biçimde düzeltildi.
- Retrieval prune aşamasında normalize edilmiş chunk'tan kaybolan `_id`/`_index`
  kimliği korunarak KNN aday eşlemesi düzeltildi; ADR 0014 ve Go regresyon testi
  eklendi.
- Commit `changes` kontratı artık canlı ad ve parent bilgisini commit ağacıyla
  karşılaştırarak rename/move üretir. Aynı dosyanın tek aksiyonda taşınıp yeniden
  adlandırılması, DB benzersizlik sözleşmesine uygun tek atomik `move` item'ında
  old/new name ve old/new parent metadata'sıyla saklanır.
- Pinned `v0.26.4` kaynağına doğrulanmış patch uygulayan, no-CGO binding ve
  retrieval regresyonunu build sırasında çalıştıran uyumlu backend image'ı
  üretildi ve gerçek runtime bu image ile yeniden oluşturuldu.
- Silinmiş belgeye ait Elasticsearch'teki 17 stale chunk silinmeden, geri
  döndürülebilir biçimde `available_int=0` yapıldı.
- Route inventory, runtime-disabled dökümü, endpoint coverage matrix, proxy ve
  contract matrix güncellendi.

## Değiştirilen dosya grupları

- UI/router: `studio/frontend/src/features/files/*`, `files.tsx`, OAuth callback
  route'u, router, root layout, sidebar, capabilities, preferences ve i18n.
- Typed servis/model: `connector-api.ts`, `connector-types.ts`,
  `connector-oauth-state.ts`, `file-api.ts`, `file-types.ts`, `dataset-types.ts`,
  platform client/config/index.
- Chat/retrieval: `platform-chat-adapter.ts` ve testi;
  backend `internal/service/nlp/retrieval.go` ve `retrieval_prune_test.go`.
- Commit metadata: frontend `file-types.ts`, `files-page.tsx` ve testleri;
  backend Python/Go file-commit servisleri, entity ve odaklı testleri.
- Runtime: `.env.rag-platform`, `rag-analyzer-nocgo.go` ve testi,
  `proxy-config.mjs`, generated `rag-platform.hybrid.conf`, Vite proxy,
  `Dockerfile.backend-with-go`, `phase12-backend-v0.26.4.patch`, odaklı build
  testi ve `phase-12-runtime-smoke.mjs`.
- Governance/test: ADR 0013/0014, Faz 12 fixture ve contract/UI testleri, route
  inventory, runtime-disabled, endpoint coverage matrix ve bu rapor.

## Eklenen frontend ekranları ve aksiyonları

| UI yolu | Aksiyonlar |
| --- | --- |
| Dosyalar → Dosya alanı | list/search/open/create/upload/download/select/delete/move/rename/parent/ancestors/link/versions |
| Dosyalar → Connector'lar | list/detail/create/update/delete/test/dataset-link/rebuild/log pagination |
| Dosyalar → Connector'lar → OAuth | Google Drive, Gmail, Box start/callback/result; popup/full-page dönüş |
| Dosyalar → Commit geçmişi | workspace/folder/dataset list/create/detail/files/diff/tree/content/changes |
| Sohbet | genel selamlaşma ve seçili dataset'ten kaynaklı cevap |

## Kullanılan backend endpoint aileleri

- `/api/v1/connectors`, `/:id`, `/:id/test`, `/:id/rebuild`, `/:id/logs`
- `/api/v1/connectors/{google,box}/oauth/web/{start,result}` ve
  `/connectors/{google-drive,gmail,box}/oauth/web/callback`
- `/api/v1/datasets/:id` connector bağlama kontratı
- `/api/v1/files`, `/files/move`, `/files/link-to-datasets`, `/files/:id`,
  `/parent`, `/ancestors`, `/versions`
- `/api/v1/{workspace,folders,datasets}/:id/commits` ve commit alt yolları
- Dataset belge upload/parse/list/chunk ve chat completion/retrieval yolları

Legacy `/api/v1/file/*` ve `/v1/{file,connector}/*` yolları API-only uyumluluk
kontratlarıdır; UI canonical endpoint'leri kullanır.

## Route coverage sonucu

Pinned runtime kaynağı ve aktif hybrid proxy'den üretilen inventory toplam **746 route**,
coverage matrix **856 kayıt** içerir ve `unclassified=0`'dır. Faz 12 için:

- 117 kayıt: 45 `implemented`, 24 `contract-verified`, 48
  `runtime-disabled`
- sınıf: 5 `frontend-screen`, 40 `frontend-action`, 6 `external-callback`, 18
  `api-only`, 48 runtime-disabled `unsupported` alternate
- runtime: 69 enabled, 48 disabled
- `capability_lost=0`; planned/in-progress/unclassified yok

Typed service, UI yolu, test kanıtı, kaynak satırı, proxy hedefi ve gerekçeler
generated matrix/inventory dosyalarında kayıtlıdır.

## Gerçek veri E2E sonucu

- Medyasoft PDF sağlık kontrolü: 651 KiB, 34 sayfa, şifresiz, metin katmanlı,
  JavaScript/form yok.
- Belge per-document `naive` + `Plain Text` parser ile 12.59 s, 20.21 s ve
  14.13 s'lik üç aralıkta tamamlandı: **Ready, 59 chunk**.
- `selam` sorgusu doğal Türkçe selamlamayla yanıtlandı.
- Belgenin amacı ve ISO referansları sorusunda ISO 27001:2013 ve ISO 27002:2013
  doğru döndü; UI Medyasoft PDF'yi Document Sources altında gösterdi.
- Retrieval kanıtı: 65 aday için KNN skoru üretildi; ilk skor yaklaşık 0.9034,
  eşik sonrası 6 sonuç döndü.
- Gerçek REST connector dataset'e bağlandı; UI logu görünür oldu ve worker
  `DONE` durumunda **1 belge / 1 chunk** taşıdı. Test için refresh cadence geçici
  olarak 0 yapıldı, ardından 30'a geri alındı.
- Gerçek dosya UI'sında rename ve move aynı dosya üzerinde birlikte yapıldı.
  Bekleyen değişikliklerde iki metadata farkı görüldü, commit tek atomik item ile
  oluşturuldu, commit ağacında yeni ad/parent doğrulandı ve pending sıfırlandı.
- Kimlik doğrulamalı Faz 12 smoke; connector/file/commit/callback, auth boundary,
  secret-log probe ve cleanup dahil PASS.

## Çalıştırılan komutlar ve sonuçları

| Komut / doğrulama | Sonuç |
| --- | --- |
| `pnpm typecheck` | PASS |
| `pnpm lint:all` | PASS; 0 error, 78 mevcut/non-blocking warning |
| Faz 12 targeted Vitest | PASS; 2 dosya, 8 test |
| `pnpm test` | PASS; 69 dosya, 233 test |
| `pnpm i18n:check:strict` | PASS; 0 missing key |
| `pnpm build` | PASS; 8080 module |
| Uyumlu image build (`build-backend-image.sh`) | PASS; patch check, binding testi, retrieval testi ve Go production binary |
| Python file-commit route unit testi (Linux image) | PASS; 17 test |
| Go metadata helper doğrudan unit testi | PASS; dependency-free command-line package |
| `CGO_ENABLED=0 go test -overlay=… ./internal/service/file` | PASS; güncel paket ve metadata testleri |
| route/proxy/coverage/contract `--check` | PASS; 746/856/0/264 |
| `node scripts/rag-platform/branding-scan.mjs --build` | PASS |
| `node scripts/rag-platform/phase-12-runtime-smoke.mjs http://127.0.0.1` | PASS |
| `docker exec rag-platform-backend nginx -t` | PASS |
| her iki repository `git diff --check` | PASS |

## Başarısız testler ve açık kabul kriterleri

Başarısız nihai test yok. İlk gerçek rename+move denemesi DB'nin `(commit_id, file_id)` benzersiz
kısıtını ortaya çıkardı; atomik metadata item düzeltmesinden sonra aynı UI
senaryosu ve tüm otomatik kapılar yeniden çalıştırılarak geçti. Tam Go file
paketini amd64 emülasyonda tekrar derleyen tanısal deneme geçici Docker build
cache'i diski doldurduğu için I/O hatasıyla kesildi; cache kaldırıldı, metadata
mantığı bağımlılıksız dosyaya ayrılıp doğrudan test edildi, güncel file paketi
native no-CGO overlay ile PASS oldu ve runtime yeniden başlatıldıktan sonra
health ile authenticated smoke tekrar PASS oldu.

## Runtime-disabled kayıtlar ve kanıtları

- OAuth start/result Python alternatifi credential-log riski nedeniyle Go
  `9384` tarafından shadow edilir; benzersiz marker log taraması PASS.
- Go connector test alternatifi Python `9380` tarafından shadow edilir; BigQuery
  dalı korunur.
- Go connector rebuild alternatifi aktif Python DB-polling worker ile uyumsuz
  task kanalına yayın yaptığı için Python `9380` tarafından shadow edilir.
- Go file-parent alternatifi canonical path parametresi yerine query okuduğu için
  Python `9380` tarafından shadow edilir.
- Faz 12'deki 48 disabled kaydın tamamı duplicate Python/Go veya legacy prefix
  alternateleridir; seçili eşdeğerleri enabled olduğundan capability kaybı yoktur.

## Bilinen sınırlamalar

- Gerçek Google/Gmail/Box provider consent başarı dönüşü, provider credential'ı
  olmadan çalıştırılmadı; state/PKCE/callback/result/redaction kontratları test
  edildi.
- Pinned v0.26.4 Python rebuild, güncel yerel kaynakta bulunan
  `run_immediately` davranışını taşımıyor; iş aktif worker tarafından connector
  refresh cadence'inde tüketiliyor.
- Native Go file paketi Darwin'de upstream'in yalnız Linux için sunduğu
  `pdf_oxide` arşiviyle varsayılan CGO profilinde derlenemez. Zorunlu file paketi
  testi production-compatible no-CGO adapter overlay'iyle çalıştırılmıştır; bu
  ürün/runtime eksikliği değildir.
- Production build büyük chunk ve ineffective dynamic import uyarıları üretir;
  build başarıyla tamamlanır.

## Sonraki faza geçiş

Belge yükleme, grounded chat, connector veri aktarımı, file commit rename/move
metadata akışı ve uyumlu backend runtime gerçek veriyle doğrulandı. Zorunlu
kalite kapılarında başarısızlık veya açık Faz 12 kabul kriteri yoktur. **Faz 12
COMPLETE**; Faz 13'e geçiş güvenlidir.
