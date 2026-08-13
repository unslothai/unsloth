# Faz 3 Sonuç Raporu

> Durum: **COMPLETE.** Uygulama, canlı runtime düzeltmesi ve duplicate logical
> model status regresyonu sahip DAO katmanındaki izole SQLite testiyle doğrulandı;
> Global Definition of Done test kapısı kapandı.
> Yalnız Faz 3 uygulandı; Faz 4 başlatılmadı.
> Her iki repository'de başlangıçta bulunan kullanıcı değişiklikleri silinmedi,
> geri alınmadı veya ezilmedi.

Faz 4 geçiş denetiminde `ConnectionsTab` entegrasyon testinin beklediği responsive
alt boşluğun component class'ında eksik olduğu ayrıca görüldü. `pb-8 sm:pb-10`
container'a geri eklendi ve hedef test yeniden çalıştırılarak PASS alındı; bu
geçiş eksikliği Faz 4 kapsamına taşınmadı.

## Yapılan değişiklikler

- Provider, provider instance, model, default model, connection, utility ve
  pipeline sözleşmeleri yerel backend kaynağından typed servis + domain adapter
  olarak uygulandı. Balance/task telemetry sözleşmeleri typed contract testiyle
  korunur ancak odaklı kurulum UI'ına dahil edilmez.
- Provider instance response `api_key` alanı domain sınırında atılır; yalnız
  `hasCredential` bilgisi tutulur. Create/edit/draft-test secret alanları masked,
  autocomplete kapalı, geçici component state'indedir ve işlem sonunda temizlenir.
- Settings → Connections'ın mevcut connection listesi ve görsel dili korunur.
  Üstteki `Add connection` yalnız ekleme yüzeyidir; Rag Platform provider
  kataloğunu doğrudan `Connection` seçicisinde açar. Provider seçildiğinde
  instance adı, API key ve Base URL aynı kutuda görünür; region UI'dan
  çıkarılmıştır. Kayıt sonrasında form kapanır ve yeni provider mevcut
  connection listesinde, `Add connection` satırının altında görünür. Satırdaki
  test/edit/delete aksiyonları inline çalışır; edit formu saklı credential'ı
  geri göstermeden boş API-key alanıyla mevcut anahtarı koruyabilir. Satır
  genişletildiğinde model/default, pipeline ve yetkili model araçları aynı
  tasarım sistemi içinde alt alta açılır. Balance ve task telemetry ikincil
  operasyon verisi olduğu için bu kurulum yüzeyinden kaldırılmıştır. Owner/admin
  dışındaki kullanıcılar salt okunur permission state'i görür; model araçları
  render edilmez.
- Önceki backend readiness/check kartı kullanıcı akışından kaldırılmıştır.
  System ping/version/health servisleri UI gerektirmeyen contract-verified
  entegrasyon sözleşmeleri olarak korunur.
- Capability listesi backend `model_type`/provider catalog yanıtlarından
  türetilir. Chat, embedding, rerank, speech-to-text, text-to-speech, OCR ve
  file-parse kontrolü uyumlu model yoksa neden göstererek kapalı kalır.
- Mixed-version hybrid runtime'da Python provider listesinde bulunan
  `OpenAI-API-Compatible`, pinned Go model kataloğunda bulunmayabilir. Provider
  model kataloğu bu nedenle yalnız enrichment olarak ele alınır; hata ekranın
  tamamına taşınmaz ve kayıtlı instance/model sözleşmeleri kullanılmaya devam
  eder. Yeni custom OpenAI-compatible bağlantılar aktif runtime'ın desteklediği
  `VLLM` sözleşmesine açıkça yönlendirilir; dropdown etiketi
  `OpenAI compatible / Custom (VLLM)` olur ve Base URL eksikse `/v1` ile
  normalize edilir. Uyumsuz seçenek dropdown'da gerekçesiyle disabled kalır.
  Draft `Test connection` başarılı ya da başarısız tamamlandığında secret form
  state'inde kalır; böylece kullanıcı aynı credential ile `Add connection`
  yapabilir. Secret yalnız başarılı create sonrasında temizlenir. VLLM create
  butonu instance adı, API key ve Base URL eksikken açık değildir.
  Bu davranış provider-specific contract/component testleriyle sabitlenmiştir.
- Kayıtlı instance model dropdown'ı artık backend'in canlı discovery
  sözleşmesini (`GET .../models?supported=true`) kullanır. Backend saklı
  credential ile provider `/models` çağrısını yaptığı için API key browser'a
  geri dönmez. Persist edilmiş instance modelleri ve statik provider kataloğu
  fallback/enrichment olarak birleştirilir. Bağlantı testi gerçek connection
  probe'unu ve discovery çağrısını birlikte çalıştırır; UI başarı halinde kaç
  model bulunduğunu, hata halinde backend mesajını instance satırında gösterir.
- Model ekleme/silme adapter'ı backend main'in tekil payload'ı ile deploy edilen
  `v0.26.4` Go runtime'ın `models[]` batch payload'ını birlikte taşır. Böylece
  her runtime kendi tanıdığı alanları okuyarak aynı UI aksiyonunu kabul eder;
  model eklemedeki HTTP 400 mixed-version hatası giderilmiştir.
- Ekli model listesi tenant genelindeki yalnız aktif `/models` cevabı yerine
  instance envanterini kullanır. Aynı model adı capability başına birden fazla
  satırla saklansa bile UI tek satırda capability'leri birleştirir; tüm kayıtlar
  inactive olduğunda model kaybolmaz, `Devre dışı` etiketiyle listede kalır ve
  yeniden etkinleştirilebilir. Reload sırasında runtime inactive modeli
  instance envanterinden çıkarsa, component son doğrulanmış kaydı korur ve
  status mutation sonucunu yerelde birleştirir; doğrulanmış silme ise bu kaydı
  birleştirmeden önce açıkça kaldırır. Silme onayı aynı logical ada ait kayıtların
  tamamını kaldırır. Pinned Go PATCH handler'ı duplicate satırlardan yalnız
  birini güncellediği için aktif hybrid map bu PATCH'i geçici olarak mevcut
  Python batch-status implementasyonuna yönlendirir. Backend main Go servisi de
  sonraki owned image için tüm logical satırları tek update ile değiştirecek
  şekilde güncellendi; karar `docs/adr/0008-model-status-and-instance-inventory.md`
  içinde kayıtlıdır.
- Yetkili utility workspace yedi runtime-enabled route'u kullanır. Metin/dosya
  limiti, MIME kontrolü, timeout, abort ve FileReader/Object URL cleanup eklendi.
  Tam embedding vektörü yerine dimension/token/ilk sekiz değer gösterilir.
- Default model mutation'ları optimistic değildir. Server cevabı sonrasında
  liste yeniden yüklenir. Embedding varsayılanı uygulamanın erişilebilir onay
  popup'ında mevcut → yeni model özeti ve indeks uyumluluğu uyarısı gösterilmeden
  değiştirilmez. Vazgeçme mutation göndermez; onay popup'ı sunucu kaydı başarıyla
  tamamlanana kadar loading durumunda açık kalır.
- Chat project create için chat; knowledge-base create için embedding readiness
  gate'i eklendi. Eksik default/capability kullanıcıyı Connections sekmesine
  yönlendirir; unmount/close sırasında readiness isteği abort edilir.
- Chat başlığındaki model seçici yalnız Connections üzerinden eklenmiş, etkin ve
  `chat` capability'sine sahip Rag Platform modellerini gösterir. Yerel model,
  Hub, Recommended ve On Device alanları bu yüzeyden kaldırılmıştır. Modeller
  provider/instance bazında ikonlu ve sade gruplar halinde listelenir; seçim
  optimistic değildir ve `PATCH /models/default` cevabı geldikten sonra chat
  varsayılanı olarak işaretlenir. Loading, empty, search, error, retry, abort ve
  Connections'a yönlendirme durumları uygulanmıştır. Faz 8 completion stream
  taşımasına dokunulmamıştır.
- Knowledge-base create/edit ekranına pipeline selector eklendi. Seçim exact Go
  dataset sözleşmesine `{pipeline_id, parse_type: 2}` map edilir ve detail DSL
  çağrısıyla doğrulanır. Aktif runtime route'u taşımadığından selector açık
  runtime-disabled nedeni gösterir ve sahte değer üretmez.
- Route inventory forward-source taraması implement edilmiş pipeline list/detail
  route'larını da kapsayacak şekilde genişletildi. Coverage matrix Phase 3
  canonical UI kanıtlarını ve gerekçeli compatibility alias'larını içerir.

## Eklenen frontend ekranları ve aksiyonları

| UI yolu                                          | Aksiyonlar                                                                                                                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Settings → Connections → Add connection          | provider seçimi ardından instance name/API key/Base URL, draft test ve create; başarılı kayıtta ana listeye dönüş                                                                                                  |
| Settings → Connections → configured provider row | provider/instance özeti, saved test + bulunan model sayısı, inline edit, inline delete confirmation, expand/collapse                                                                                               |
| Configured provider → Models ve varsayılanlar    | server-side live supported-model discovery, persisted/provider catalog fallback, loading/empty/error/retry, add, enable/disable, delete, chat/embedding/rerank ve backend-dönen diğer capability default seçimleri |
| Configured provider → Pipeline kataloğu          | loading/empty/error/runtime-disabled katalog durumu                                                                                                                                                                |
| Configured provider → Yetkili model araçları     | chat-to-model, embedding, rerank, transcription, speech, OCR, file parse; abort ve güvenli sonuç gösterimi                                                                                                         |
| Knowledge bases → Create/Edit                    | pipeline selector + exact dataset mapping; embedding readiness gate                                                                                                                                                |
| New project                                      | chat readiness gate ve Settings yönlendirmesi                                                                                                                                                                      |
| Chat → Select model                              | yalnız etkin Connections chat modellerini provider/instance bazında listeleme, arama ve server-confirmed chat varsayılanı seçme                                                                                    |

UI component'lerinden doğrudan network çağrısı yapılmaz. Akış
`component → typed service/domain adapter → platformRequest → backend`
biçimindedir.

## Kullanılan backend endpoint'leri

- Provider: `GET/PUT /api/v1/providers`, provider detail/delete, provider model
  list/detail, provider draft connection test.
- Instance: create/list/detail/update/delete, saved connection test, instance
  model list (`supported=true` live discovery dahil), add/patch/delete.
- API-only provider telemetry: balance ve task list/detail typed servisleri exact
  contract testleriyle korunur; Connections kurulum UI'ında gösterilmez.
- Model/default: `GET /api/v1/models`, `GET/PATCH /api/v1/models/default`,
  `GET /api/v1/users/me/models`.
- Utility: `POST /api/v1/chat/to_model`, `/embeddings`, `/rerank`,
  `/audio/transcriptions`, `/audio/speech`, `/file/ocr`, `/file/parse`.
- Pipeline typed contract: `GET /api/v1/pipelines` ve
  `GET /api/v1/pipelines/:id`; deploy runtime'ında 404 olduğu için UI çağrısı
  explicit runtime-disabled state'e düşer.

`PATCH /models`, `PATCH /users/me/models`, Python instance-model batch PUT,
Python model-path POST ve balance/task telemetry route'ları
`api-only / contract-verified` sınıflandırılmıştır.

## Route coverage sonucu

- Inventory: **711** top-level route.
- Coverage: **821** kayıt; erişilebilir **516**; `unclassified=0`.
- Runtime-disabled: **190**; bunların **9** tanesinde erişilebilir eşdeğer yoktur.
- Faz 3: **56** kayıt:
  - `implemented`: **28**
  - `contract-verified`: **7**
  - `runtime-disabled`: **21**
  - `frontend-action`: **22**
  - `frontend-screen`: **6**
  - `api-only`: **7**
  - `unsupported`: **21**
- Erişilebilir Faz 3 route'larında `planned` veya `in-progress`: **0**.
- Faz 3 `unsupported` kayıtlarında boş/gerekçesiz karar: **0**.
- 21 runtime-disabled Faz 3 kaydın **19** tanesi aktif Python/Go duplicate
  implementasyonuyla korunur; capability kaybı olan yalnız iki pipeline route'udur.

## Runtime-disabled kayıtlar ve kanıt

| Route grubu                                                     | Kaynak                                                                                                                                      | Proxy/smoke                                                                                                                                                                              | Karar                                                                                                                  |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 19 Python/Go duplicate provider/model route'u                   | pinned source içinde iki implementasyon                                                                                                     | hybrid map diğer servisi seçer                                                                                                                                                           | capability korunur; matrix alternates                                                                                  |
| `OpenAI-API-Compatible` provider model katalog zenginleştirmesi | local Python catalog provider'ı içerir; pinned Go model config'i içermez; pinned runtime `VLLM` generic OpenAI-compatible driver'ını içerir | authenticated Go yanıtı `provider 'OpenAI-API-Compatible' not found`; pinned Python image aynı yeni REST route'unda 404; external LiteLLM `/v1/models` auth katmanı 401 ile erişilebilir | uyumsuz seçenek disabled + gerekçeli; custom bağlantı `VLLM` sözleşmesine ve normalize `/v1` base URL'ye yönlendirilir |
| `GET /api/v1/pipelines`                                         | backend main `router.go:170`, `pipeline.go` implemented                                                                                     | canlı proxy HTTP 404                                                                                                                                                                     | runtime-disabled; UI açık neden + built-in parser fallback                                                             |
| `GET /api/v1/pipelines/:id`                                     | backend main `router.go:171`, `pipeline.go` implemented                                                                                     | canlı proxy HTTP 404                                                                                                                                                                     | runtime-disabled; sahte detail/selector yok                                                                            |

Canlı kontrol ayrıca auth'suz `GET /api/v1/providers` ve
`POST /api/v1/chat/to_model` için HTTP 401 döndürdü; bu route'ların active hybrid
proxy üzerinden doğru auth katmanına ulaştığını gösterir. Ayrıntı
`phase-3-model-contract.md`, `route-inventory.json` ve `runtime-disabled.md`
içindedir.

## Test ve doğrulama kanıtı

| Komut / kontrol                                       | Sonuç                                                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `npm run typecheck`                                   | PASS                                                                                                  |
| `npm run lint:all`                                    | PASS; 0 hata, önceden mevcut 78 warning                                                               |
| `npx vitest run --maxWorkers=1 --testTimeout=30000`   | PASS; 25 dosya, 87/87 test                                                                            |
| Connection UI targeted Vitest                         | PASS; 3 dosya, 12/12 test; ana liste yerleşimi, inline edit, canlı discovery ve model-add akışı dahil |
| `npm run build`                                       | PASS; TypeScript + Vite production build                                                              |
| `npm run i18n:check:strict`                           | PASS; missing key 0                                                                                   |
| `npm run catalog:check`                               | PASS                                                                                                  |
| `node scripts/rag-platform/branding-scan.mjs --build` | PASS; source + build artifact                                                                         |
| Route inventory `--check`                             | PASS; 711 route                                                                                       |
| Coverage matrix `--check`                             | PASS; 821 kayıt, `unclassified=0`                                                                     |
| Contract matrix `--check`                             | PASS; 272 scanned pair                                                                                |
| `git diff --check`                                    | PASS                                                                                                  |
| Hybrid live route smoke                               | PASS; provider/utility 401, pipeline list/detail 404                                                  |
| Codebase Memory Verify coverage                       | PASS, best-effort caveat; tüm kanıt yollarında recorded gap yok ve exact source ayrıca okundu         |
| Model disable/re-enable/delete component integration  | PASS; 9/9, inactive envanterden düşse de satır kalır, yeniden etkinleşir ve delete payload'ı doğrulanır |
| Model typed service + component targeted suite        | PASS; 17/17                                                                                           |
| Chat connected-model selector targeted suite          | PASS; 2/2; active chat filtreleme, empty state ve server-confirmed default mutation                   |
| Go duplicate-status regression test                   | PASS; `CGO_ENABLED=0 go test ./internal/dao -count=1`, duplicate logical model satırları ve scope izolasyonu |

Repository'de ayrı bir browser E2E scripti bulunmadığından UI akışı MSW
contract/component testiyle yürütülmüştür. Canlı kayıt üzerinde secret değeri
çıktıya alınmadan provider `/v1/models` çağrısı 200/12 model ve düşük maliyetli
chat probe'u 200 olarak doğrulanmıştır. In-app browser yerel uygulamanın parola
ekranında kaldığından kullanıcının oturumunu devralan görsel browser smoke'u
çalıştırılmamıştır.

### Başarısız veya tamamlanamayan testler

Frontend final durumda başarısız test veya kabul kriteri bırakmaz. Kalite kapıları sırasında
ilk birleşik i18n komutu repository root'tan çalıştırıldığı için `Missing script`
döndürdü; doğru `studio/frontend` çalışma dizininde yeniden çalıştırıldı ve PASS
oldu. Son readiness cleanup düzenlemesinin ilk typecheck'i React `useRef`
başlangıç değeri eksikliğini yakaladı; ref'ler açık `undefined` başlangıcıyla
düzeltildi, typecheck ve production build yeniden çalıştırılarak PASS oldu.
Başarısız unit/contract/integration testi yoktur.
Son doğrulama yüksek Docker/VM CPU yükü altında normalden uzun sürdü; 25 dosya
tek worker ve 30 saniyelik test sınırıyla çalıştırıldı ve 87/87 PASS oldu.
Assertion veya contract failure kalmadı.

İlk service-package doğrulaması repository dışında beklenen
`/Users/baran/ragflow-native-libs/office_oxide/lib/liboffice_oxide.a` bulunmadığı,
macOS derleyicisi Linux-only `Pdeathsig` alanını desteklemediği ve read-only
Linux container denemesi geçici alan sınırına ulaştığı için çalıştırılamadı.
Regresyonun sahipliği daha dar `TenantModelDAO` katmanına taşındı:
`UpdateStatusByNameAndScope` aynı provider/instance/model adına ait bütün
capability satırlarını tek scoped mutation ile günceller. İzole SQLite testi iki
duplicate satırın birlikte güncellendiğini ve başka instance satırının
değişmediğini doğrular. Hedef test ve tüm `internal/dao` paketi ayrı ayrı PASS
oldu; final durumda başarısız ilgili backend testi kalmadı.

## Değiştirilen dosyalar

- Typed service/domain:
  - `studio/frontend/src/integrations/platform-backend/{model-api,model-types,model-readiness}.ts`
  - `studio/frontend/src/integrations/platform-backend/{config,index}.ts`
- UI ve readiness:
  - `studio/frontend/src/features/settings/components/{platform-models-settings,platform-model-tools}.tsx`
  - `studio/frontend/src/features/settings/tabs/connections-tab.tsx`
  - `studio/frontend/src/features/chat/chat-providers-dialog.tsx`
  - `studio/frontend/src/features/rag/components/{platform-pipeline-select,knowledge-base-dialog}.tsx`
  - `studio/frontend/src/features/chat/components/new-project-dialog.tsx`
  - `studio/frontend/src/features/chat/components/platform-chat-model-selector.tsx`
  - `studio/frontend/src/features/chat/chat-page.tsx`
- Testler:
  - `studio/frontend/src/integrations/platform-backend/__tests__/{model-api,model-readiness,config}.test.ts`
  - `studio/frontend/src/features/settings/components/platform-model-tools.test.tsx`
  - `studio/frontend/src/features/settings/tabs/connections-tab.test.tsx`
  - `studio/frontend/src/features/chat/chat-providers-dialog.test.tsx`
  - `studio/frontend/src/features/chat/components/platform-chat-model-selector.test.tsx`
  - kaldırıldı: `studio/frontend/src/integrations/platform-backend/__tests__/backend-connection-status.test.tsx`
- Kaldırılan backend-check UI:
  - `studio/frontend/src/integrations/platform-backend/backend-connection-status.tsx`
- Rollout/config:
  - `studio/frontend/.env.example`
- Governance ve dokümantasyon:
  - `scripts/rag-platform/{proxy-config,route-inventory,coverage-matrix}.mjs`
  - `docs/rag-platform/{route-inventory,endpoint-coverage-matrix}.{md,json}`
  - `docs/rag-platform/{runtime-disabled,phase-3-model-contract,faz-3-sonuc-raporu}.md`
  - `docs/adr/0008-model-status-and-instance-inventory.md`
  - `infra/rag-platform/rag-platform.hybrid.conf`

Backend repository'de `internal/dao/tenant_model.go`,
`internal/dao/tenant_model_test.go`, `internal/service/model_service.go` ve
`internal/service/model_service_test.go` duplicate logical model status
güncellemesi için değiştirildi. Başlangıçtaki
`.gitignore` değişikliği, iki PEM silme kaydı ve untracked governance workflow
aynen korundu.
Frontend'de çalışma sırasında kullanıcı tarafından yapılan
`src/i18n/locales/tr.ts` (`Profili` → `Profil`) değişikliği bu fazın parçası
değildir; değiştirilmeden korunmuştur.

## Bilinen sınırlamalar

- Pipeline handler'ları normative backend `main` kaynağında implement edilmiş
  olsa da pinned `v0.26.4` runtime image bunları içermez. Selector açık neden
  gösterir ve built-in parser fallback'i korur; pipeline seçimi runtime image
  yükseltilene kadar kullanılamaz.
- Pinned Go runtime'ın statik model kataloğu Python provider kataloğundan daha
  dardır. `OpenAI-API-Compatible` için provider-level katalog zenginleştirmesi
  kullanılamaz; UI kayıtlı instance modellerini göstermeye devam eder ve yeni
  custom OpenAI-compatible endpoint'leri desteklenen `VLLM` provider'ıyla
  ekler. Statik provider model kataloğunun tamamı runtime image eşitlendiğinde
  `OpenAI-API-Compatible` adıyla da geri gelir.
- Canlı provider model discovery ve kısa chat probe'u başarıyla çalıştı; secret
  hiçbir komut çıktısına veya repository dosyasına yazılmadı. Görsel browser
  smoke'u yerel uygulamanın kullanıcı parolasıyla açılması gerektiğinden bu
  oturumda tamamlanamadı.
- Repository'de browser E2E runner/scripti yoktur; gerçek UI yolu bunun yerine
  contract + component entegrasyon testiyle korunur.
- Production build exit kodu 0'dır; repository'nin önceden mevcut büyük chunk
  ve ineffective dynamic-import uyarıları devam eder, Faz 3'e özgü build hatası
  yoktur.
- Pinned runtime Go image yeni batch-status değişikliğini henüz içermez; aktif
  proxy override PATCH'i Python 9380'e gönderir. Source image yeniden üretildiğinde
  override ADR 0008'e göre kaldırılmalıdır.

## Sonraki faz kapısı

Faz 3 davranış ve runtime kabul kriterleri uygulanmıştır; runtime-enabled
kullanıcıya anlamlı route'larda bitmemiş status yoktur. Duplicate logical model
status regresyon testi sahip DAO katmanında PASS olduğu için Global Definition
of Done tamamlandı. Faz 4'e geçiş **güvenlidir**.
