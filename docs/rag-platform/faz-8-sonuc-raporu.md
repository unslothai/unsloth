# Faz 8 Sonuç Raporu

## Faz durumu

**COMPLETE**

Faz 8 kapsamındaki native chat akışı, reasoning/referans/usage ayrıştırması, geri bildirim, zihin haritası, öneriler, konuşma sentezi/transkripsiyon adaptörleri, istemci tarafı iptal semantiği, UI erişim yolları, contract testleri ve runtime route smoke doğrulamaları uygulandı. Faz 8'e ait coverage kayıtlarında `planned` veya `in-progress` endpoint kalmadı.

Çalışma sırasında kaldırılmış olan Faz 3 `PlatformModelTools` bileşeni ve testi kullanıcı talimatıyla geri yüklendi; ayarlar ekranındaki owner/admin erişim yolu yeniden bağlandı. Böylece `/api/v1/audio/speech` ve `/api/v1/audio/transcriptions` için coverage matrisinde belirtilen gerçek UI yolu yeniden doğrulandı. Global Definition of Done ve Faz 8 kabul kriterleri sağlandı.

## Yapılan değişiklikler

- Go native `/api/v1/chat/completions` SSE sözleşmesi için `choices[]` varsayımı yapmayan ayrı typed service ve runtime adapter eklendi.
- Parçalı SSE frame, incremental/cumulative delta, reasoning sınırları, referanslar, usage, terminal `data:true`, duplicate frame, abort ve terminal öncesi bağlantı kopması işlendi.
- Go feedback ile Python mindmap/recommendation endpoint'leri typed servisler ve gerçek chat UI aksiyonlarıyla bağlandı.
- Öneriler yalnızca composer metnini doldurur; otomatik gönderim yapmaz.
- Mindmap erişilebilir ağaç/dialog ve JSON export akışıyla sunuldu.
- Mesaj geri bildirimi olumlu/olumsuz ve isteğe bağlı metinle bağlandı.
- Python konuşma sentezi ve transkripsiyon endpoint'leri Assistant UI speech/dictation adaptörlerine bağlandı.
- Ses MIME türü, 25 MB dosya sınırı, iki dakikalık kayıt sınırı, izin reddi, abort, MediaRecorder track cleanup ve object URL revoke durumları uygulandı.
- İptal davranışı sunucu-side cancel olarak sunulmadı; yalnızca tarayıcı bağlantısının kapatıldığı açıkça belirtildi.
- Referans metadata'sı belge/chunk kimliğiyle mevcut ortak kaynak önizlemesine aktarıldı.
- Native stream ve client-only cancellation kararı ADR-0009 ile kaydedildi.
- Faz 8 source-verified contract fixture ve runtime smoke betiği eklendi.
- Contract matrix, route inventory ve endpoint coverage matrix yeniden üretildi.
- Provider kataloğunda bulunmayan legacy `@provider` yönlendirme soneki, yalnızca canlı katalog soneksiz modeli doğruladığında güvenli biçimde normalize edildi; gerçek `@` içeren model kimlikleri korunur.
- Go `model_type` bitmask ve numeric-array cevapları tüm chat/embedding/ASR/vision/rerank/TTS/OCR yetenekleri için frontend domain modeline uyarlandı.
- Rag Platform chat aktifken opsiyonel yerel inference envanterinin 502 hatası platform modeli hatası gibi gösterilmez; açık yerel model işlemlerinin hata görünürlüğü korunur.
- Başarılı native platform stream'inden sonra composer pre-stream rezervasyonu `finally` ile her terminal yolda bırakılır; aynı sohbette ikinci ve sonraki mesajlar artık kuyrukta kilitli kalmaz.
- assistant-ui'nin yerel sohbet kimliği ile persistence sonrasında atanan backend
  Session kimliği aynı bağlı sohbetin iki alias'ı olarak ele alınır. Backend
  kimliği aktif hale geldiğinde gereksiz yeniden-switch yapılmaz ve composer
  kalıcı olarak disabled durumda bırakılmaz.
- Parented platform geçmişi zaman damgasına göre yeniden sıralanmak yerine kararlı topolojik sırada içe aktarılır. Backend'in milisaniye/saniye/eksik timestamp karışımı çocuk mesajı ebeveyninden önce getirse bile geçmiş artık tamamen boşalmıyor.
- Rezervasyon success/error/erken tüketici kapanışı ve geçmiş parent-order davranışları için altı yeni regresyon testi eklendi.
- Normal `General` sohbeti için eksik olan görünür dataset kapsamı kontrolü chat başlığına eklendi. Kaynak seçimi typed Chat adapter üzerinden backend `dataset_ids` alanına kalıcı yazılır; loading, empty, error, permission, timeout, abort ve cleanup durumları görünürdür.
- Canlı kullanıcı hesabındaki `General` Chat, UI üzerinden `baran` dataset'ine bağlandı. Backend kaydı ve doğrudan retrieval sonucu belge işleme yerine chat-scope eksikliğinin giderildiğini doğruladı.
- Hybrid Elasticsearch aramasında lexical `query_string` koşulunun KNN
  filtresine sızması düzeltildi. Genel belge soruları artık birebir kelime
  eşleşmesi olmadığında da dataset/izin/availability filtreleri altında vektör
  adaylarını değerlendirebilir.
- Native SSE hata event'leri boş başarılı yanıt gibi yutulmak yerine assistant
  runtime'a hata olarak iletilir. Genel chat sidebar sorgusundaki sahte
  `__no_project_selected__` Chat kimliği kaldırıldı.
- Backend'de üretilen belge referanslarının sohbet geçmişi yüklenirken yalnızca
  ham metadata olarak kalması giderildi. Referanslar ortak domain normalizatörüyle
  `platformCitations` alanına dönüştürülür; kaynak UI'sı eski ham
  `platformReference` kayıtlarını da geriye dönük olarak görünür belge kartına
  çevirir. Backend'in `docnm_kwd` ve `position_int` uyumluluk alanları da aynı
  typed normalizatörde desteklenir.
- Platform chat referansları açık `source: platform` ayrımıyla taşınır. Kaynak
  kartı artık kapalı yerel `/api/rag/documents/{id}/preview-target` servisine
  gitmez; typed `GET /api/v1/documents/{id}/preview` servisini kullanır. PDF için
  tek kullanımlık blob URL oluşturulur; panel kapanınca istek abort edilir ve URL
  revoke edilir. Eski sohbetlerde `source` işareti bulunmayan kalıcı
  `platformCitations` metadata'sı da geriye uyumlu olarak platform kaynağına
  yükseltilir. Böylece yeni ve geçmiş referans belge açılışındaki 502 giderildi.
- `See response details` artık native platform adapter'ının usage, başlangıç/bitiş
  zamanı, süre, Chat yapılandırmasındaki gerçek `llm_id`, Chat/Session/backend
  mesaj kimlikleri, stream durumu, reasoning, feedback ve referans sayılarını
  ortak typed metadata üzerinden gösterir. Yeni ve geçmiş platform mesajları
  desteklenir. Panel ayrıca eldeki response
  metadata'sının tamamını kaydırılabilir JSON olarak sunar; authorization,
  cookie, parola, provider/API key ve kimlik tokenları render öncesinde redakte
  edilir ve hiçbir yeni secret kalıcılaştırılmaz.
- Geçmiş oturum kimliksiz bir assistant karşılama mesajıyla başladığında bu
  mesaj artık backend `reference[]` dizisinden öğe tüketmez. Her cevap kendi
  belge referansıyla eşleşir; son cevap kaynaksız kalmaz.
- Rag Platform modunda sohbet ayarları, desteklenmeyen legacy
  `/api/chat/settings` uçları yerine sanitize edilmiş tarayıcı deposunda kalıcı
  tutulur. Secret/provider key alanları kabul edilmez.
- Connections ekranı platform oturumu açıkken legacy provider registry/sync
  uçlarını çağırmaz; platform model yönetimi tek backend otoritesi olarak kalır.
- Resources ekranı platform modunda legacy `/api/system`, model-memory ve
  Hugging Face cache uçlarına gitmez. Canlı platform health/version bilgisi ve
  desteklenmeyen telemetri için açık bir unavailable durumu gösterir; sahte
  sıfır değer ve 502 bildirimi üretmez.
- Frontend bağımlılık kilidi güvenli aynı-ana-sürüm yamalarıyla güncellendi;
  `npm audit` sonucu 18 bulgudan sıfıra indirildi.

## Değiştirilen dosyalar

### Uygulama ve testler

- `studio/frontend/src/integrations/platform-backend/client.ts`
- `studio/frontend/src/integrations/platform-backend/sse.ts`
- `studio/frontend/src/integrations/platform-backend/index.ts`
- `studio/frontend/src/integrations/platform-backend/chat-types.ts`
- `studio/frontend/src/integrations/platform-backend/chat-completion-types.ts`
- `studio/frontend/src/integrations/platform-backend/chat-completion-api.ts`
- `studio/frontend/src/integrations/platform-backend/__tests__/sse.test.ts`
- `studio/frontend/src/integrations/platform-backend/__tests__/chat-completion-api.test.ts`
- `studio/frontend/src/features/chat/api/platform-chat-runtime-adapter.ts`
- `studio/frontend/src/features/chat/api/platform-chat-runtime-adapter.test.ts`
- `studio/frontend/src/features/chat/api/platform-chat-adapter.ts`
- `studio/frontend/src/features/chat/runtime-provider.tsx`
- `studio/frontend/src/features/chat/utils/chat-history-order.ts`
- `studio/frontend/src/features/chat/utils/chat-history-order.test.ts`
- `studio/frontend/src/features/chat/utils/pre-stream-run-reservation.ts`
- `studio/frontend/src/features/chat/utils/pre-stream-run-reservation.test.ts`
- `studio/frontend/src/features/chat/utils/thread-identity.ts`
- `studio/frontend/src/features/chat/utils/thread-identity.test.ts`
- `studio/frontend/src/features/chat/components/platform-chat-enrichments.tsx`
- `studio/frontend/src/features/chat/components/platform-chat-enrichments.test.tsx`
- `studio/frontend/src/features/chat/components/platform-chat-sources-button.tsx`
- `studio/frontend/src/features/chat/components/platform-chat-sources-button.test.tsx`
- `studio/frontend/src/features/chat/adapters/platform-speech-synthesis-adapter.ts`
- `studio/frontend/src/features/chat/adapters/platform-dictation-adapter.ts`
- `studio/frontend/src/features/chat/adapters/platform-voice-adapters.test.ts`
- `studio/frontend/src/features/chat/adapters/studio-dictation-adapter.tsx`
- `studio/frontend/src/features/settings/components/platform-model-tools.tsx` (geri yüklendi)
- `studio/frontend/src/features/settings/components/platform-model-tools.test.tsx` (geri yüklendi)
- `studio/frontend/src/features/settings/components/platform-models-settings.tsx`
- `studio/frontend/src/components/assistant-ui/thread.tsx`
- `studio/frontend/src/components/assistant-ui/rag-sources.tsx`
- `studio/frontend/src/components/assistant-ui/rag-sources.test.tsx`
- `studio/frontend/src/components/assistant-ui/citation-utils.ts`
- `studio/frontend/src/components/assistant-ui/citation-utils.test.tsx`
- `studio/frontend/src/components/assistant-ui/tool-ui-knowledge-base.tsx`
- `studio/frontend/src/components/assistant-ui/message-response-details-sheet.tsx`
- `studio/frontend/src/components/assistant-ui/message-response-details-sheet.test.tsx`
- `studio/frontend/src/features/chat/chat-page.tsx`
- `studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts`
- `studio/frontend/src/features/chat/hooks/model-refresh-error.ts`
- `studio/frontend/src/features/chat/hooks/model-refresh-error.test.ts`
- `studio/frontend/src/features/chat/utils/chat-settings-storage.ts`
- `studio/frontend/src/features/chat/utils/chat-settings-storage.test.ts`
- `studio/frontend/src/features/chat/chat-providers-dialog.tsx`
- `studio/frontend/src/features/chat/chat-providers-dialog.test.tsx`
- `studio/frontend/src/features/settings/tabs/connections-tab.tsx`
- `studio/frontend/src/features/settings/tabs/connections-tab.test.tsx`
- `studio/frontend/src/features/settings/tabs/resources-tab.tsx`
- `studio/frontend/src/features/settings/tabs/resources-tab.test.tsx`
- `studio/frontend/package.json`
- `studio/frontend/package-lock.json`
- `studio/frontend/src/integrations/platform-backend/model-types.ts`
- `studio/frontend/src/integrations/platform-backend/__tests__/model-api.test.ts`
- `studio/frontend/src/features/rag/api/rag-api.ts`
- `studio/frontend/src/features/rag/api/document-preview-adapter.ts`
- `studio/frontend/src/features/rag/api/document-preview-adapter.test.ts`
- `studio/frontend/src/features/rag/components/document-preview-sheet.tsx`
- `studio/frontend/src/features/rag/components/preview-store.ts`
- `../rag-backend/internal/engine/elasticsearch/chunk.go`
- `../rag-backend/internal/engine/elasticsearch/chunk_helpers_test.go`

### Dokümantasyon ve yönetişim

- `docs/adr/0009-native-chat-stream-and-client-only-cancellation.md`
- `docs/rag-platform/fixtures/phase-8-chat-contract.json`
- `docs/rag-platform/fixtures/README.md`
- `docs/rag-platform/contract-matrix.md`
- `docs/rag-platform/route-inventory.json`
- `docs/rag-platform/route-inventory.md`
- `docs/rag-platform/endpoint-coverage-matrix.json`
- `docs/rag-platform/endpoint-coverage-matrix.md`
- `scripts/rag-platform/contract-matrix.mjs`
- `scripts/rag-platform/coverage-matrix.mjs`
- `scripts/rag-platform/phase-8-runtime-smoke.mjs`
- `docs/rag-platform/faz-8-sonuc-raporu.md`

### Backend model uyumluluğu

- `internal/dao/migration.go`
- `internal/service/model_service.go`
- `internal/service/model_name_normalization.go`
- `internal/service/model_name_normalization_test.go`

## Eklenen frontend ekranları ve aksiyonları

- Chat composer üzerinden native platform completion başlatma ve stream'i durdurma.
- Chat başlığından normal veya proje sohbetinin belge dataset'lerini seçme ve backend'e kaydetme.
- Assistant mesajında reasoning, referanslar ve usage metadata'sını koruma.
- Kaynak kartından belge/chunk önizleme açma.
- Canlı yanıt ve yeniden açılan sohbet geçmişinde referans belge kartını koruma.
- Backend'in ham `[ID:n]` chunk işaretlerini yeni ve geçmiş assistant
  mesajlarında kullanıcı metninden kaldırma; ayrı ve tıklanabilir belge kaynak
  kartlarını koruma.
- Platform kaynak kartından belgeyi platform preview route'u ile açma; kapanışta
  ağ isteğini ve geçici blob URL'yi temizleme.
- Assistant mesajında olumlu/olumsuz geri bildirim ve isteğe bağlı açıklama gönderme.
- Chat için zihin haritası açma, yeniden deneme ve JSON dışa aktarma.
- Chat önerilerini açma/kapatma ve seçilen öneriyi composer'a yerleştirme.
- Assistant mesajını platform TTS ile okutma/durdurma.
- Composer mikrofonundan kayıt başlatma, durdurma, iptal etme ve platform transkripsiyonunu metne aktarma.
- Loading, empty, error, permission, timeout, abort ve cleanup durumları.

## Kullanılan backend endpoint'leri ve Faz 8 route sınıflandırması

| Runtime | Method | Endpoint | Sınıf | Durum | Typed service / UI yolu |
|---|---|---|---|---|---|
| Go | POST | `/api/v1/chat/completions` | `frontend-action` | `implemented` | `streamChatCompletion`; Chat composer → gönder/durdur |
| Go | PUT | `/api/v1/chat/completions/{completion_id}` | `frontend-action` | `implemented` | `submitChatCompletionFeedback`; assistant mesajı → geri bildirim |
| Python | GET | `/api/v1/chats/{chat_id}/mindmap` | `frontend-action` | `implemented` | `getChatMindmap`; chat → zihin haritası |
| Python | GET | `/api/v1/chats/{chat_id}/recommendation` | `frontend-action` | `implemented` | `getChatRecommendations`; chat → öneriler |
| Python | POST | `/api/v1/chats/{chat_id}/speech` | `frontend-action` | `implemented` | `synthesizeChatSpeech`; mesaj → sesli oku |
| Python | POST | `/api/v1/chats/{chat_id}/transcription` | `frontend-action` | `implemented` | `transcribeChatAudio`; composer → mikrofon |
| Python | POST | `/api/v1/chats/{chat_id}/completions` | `api-only` | `contract-verified` | Deprecated backward-compatible alias; aktif UI Go route'unu kullanır |
| Python | POST | `/api/v1/chat/completions` | `unsupported` | `runtime-disabled` | Hybrid proxy Go eşdeğerine yönlendirir (`proxy-shadowed`) |
| Python | PUT | `/api/v1/chat/completions/{completion_id}` | `unsupported` | `runtime-disabled` | Hybrid proxy Go eşdeğerine yönlendirir (`proxy-shadowed`) |
| Go | GET | `/api/v1/chats/{chat_id}/mindmap` | `unsupported` | `runtime-disabled` | Hybrid proxy Python eşdeğerine yönlendirir (`proxy-shadowed`) |
| Go | GET | `/api/v1/chats/{chat_id}/recommendation` | `unsupported` | `runtime-disabled` | Hybrid proxy Python eşdeğerine yönlendirir (`proxy-shadowed`) |

## Route coverage sonuçları

- Route inventory: **711** route, validator PASS.
- Endpoint coverage matrix: **821** kayıt, `unclassified=0`, validator PASS.
- Faz 8: **6 implemented**, **1 contract-verified**, **4 runtime-disabled**.
- Faz 8'de `planned` veya `in-progress`: **0**.
- Contract matrix: **264** route çifti, validator PASS.
- Aktif proxy hedefleri:
  - completion/feedback → `127.0.0.1:9384` (Go)
  - speech/transcription/mindmap/recommendation → `127.0.0.1:9380` (Python)
- Codebase Memory Verify incelemesinde ilgili frontend/backend kaynak yollarında kayıtlı indeks boşluğu görülmedi. Bildirilen dar test/type aralıkları doğrudan kaynak üzerinden ayrıca okundu.
- Faz 3 model speech/transcription araçlarının ayarlar UI yolu geri yüklendi ve ilgili component/contract testleriyle yeniden doğrulandı; coverage kanıtında semantik drift kalmadı.
- Faz 7 `GET/PATCH /api/v1/chats/{chat_id}` kayıtlarının UI kanıtı, chat başlığındaki `Chat sources` yolu ve `platform-chat-sources-button.test.tsx` ile güncellendi; yeni backend route eklenmedi.

## Runtime kanıtları

Yerel `rag-platform-backend:0.26.4` hybrid runtime üzerinde authenticated smoke çalıştırıldı:

- login/chat/session oluşturma ve cleanup/logout: HTTP 200, application code 0.
- Go completion: SSE route ve content type doğrulandı; tenant'ta varsayılan chat modeli olmadığı için application code 500.
- Go feedback: sahte turn ile route erişimi HTTP 200, application code 0.
- Python mindmap/recommendation: route erişimi doğrulandı; model/dataset yokluğu application code 100.
- Python speech/transcription: route erişimi doğrulandı; tenant audio capability eksikliği application code 102.
- Soneksiz `BAAI/bge-small-en-v1.5` ile yükleme + parse smoke tamamlandı: belge `run=3`, `progress=1`, `chunk_num=1`; geçici belge API ile silindi ve veritabanında kalmadığı doğrulandı.
- Chat UI'da `claude-sonnet-5` ve `openai/claude-opus-5` VLLM modelleri gerçek completion ile başarıyla yanıt verdi; chat başlığında yerel servis kaynaklı 502 durum çipi/uyarısı oluşmadı.
- Canlı tarayıcı regresyonunda aynı yeni sohbet içinde `BIR` ve `IKI` istekleri ardışık olarak tamamlandı; ikinci istek queue durumuna düşmedi ve stream sonunda composer yeniden kullanılabilir kaldı.
- Aynı canlı sohbet başka bir sohbet açıldıktan sonra yeniden seçildi; iki kullanıcı mesajı ve iki assistant yanıtı backend Session geçmişinden eksiksiz hydrate edildi. Geçici doğrulama sohbeti sonrasında UI üzerinden silindi.
- Canlı chat başlığında `Chat sources` açıldı; `baran` seçilip kaydedildi ve `General.kb_ids` değerinin ilgili dataset kimliğini içerdiği veritabanından doğrulandı. Hassas veri içeren belgeyi modele yeniden iletmemek için yeni provider completion oluşturulmadı.
- Son belge sorularının doğru Chat/dataset kapsamıyla backend'e ulaştığı, fakat
  Elasticsearch hybrid aramasının `0` aday döndürdüğü loglardan doğrulandı.
  İndekste `BaranCV.pdf` için dört chunk bulunduğu ve yalnız dataset/availability
  filtresinin dört kaydı da gördüğü içerik okunmadan doğrulandı. Kök neden,
  lexical sorgunun KNN `filter` ağacına eklenerek semantik adayların da birebir
  kelime eşleşmesine zorlanmasıydı.
- Düzeltme `CGO_ENABLED=0` Go regresyon testiyle doğrulandı, sabit
  `rag-platform-backend:0.26.4` runtime imajına temiz release bağlamı üzerinden
  taşındı ve container yeniden oluşturuldu. Aktif Go API kimliksiz smoke'ta
  beklenen HTTP 401'i verdi; imaj `hybrid-knn-filter` hotfix etiketiyle çalışıyor.
- Son kayıtlı sohbet satırında belge/yanıt içeriği okunmadan referans yapısı
  doğrulandı: dört assistant referans grubunun her birinde dört chunk ve bir
  belge özeti mevcut. Bu, referansın backend'de üretildiğini ve kaybın frontend
  geçmiş normalizasyonu/render yolunda olduğunu kanıtladı.
- Kimliksiz ve içerik okumayan route smoke'ta eski yerel
  `/api/rag/documents/test/preview-target` çağrısı HTTP 502 döndürürken aktif
  platform `/api/v1/documents/test/preview` çağrısı beklenen HTTP 401'i döndürdü.
  UI artık platform referanslarını yalnızca ikinci route'a yönlendirir.
- Authenticated canlı chat regresyonunda `BaranCV.pdf` kaynaklı geçmiş yanıt
  yeniden açıldı. Ham `[ID:0]` işaretlerinin DOM'da bulunmadığı, buna karşılık
  `BaranCV.pdf · p.1` kaynak düğmesinin görünür kaldığı doğrulandı.
- Aynı authenticated sohbette belgeye ilişkin iki soru art arda gönderildi.
  Her iki yanıt da doğru dataset bağlamını ve görünür belge kaynağını korudu;
  her stream sonrasında composer yeniden kullanılabilir oldu. Yeni sohbet
  görünümüne geçilip geçmiş sohbet tekrar açıldığında iki kullanıcı mesajı,
  iki assistant yanıtı ve iki kaynak ilişkisi eksiksiz hydrate edildi.
- İkinci yanıtın kaynak kartı açıldı; `BaranCV.pdf · page 1` önizleme dialog'u
  gerçek platform preview route'u üzerinden yüklendi ve 502 oluşmadı.

Bu smoke provider çıktısının kalitesini değil, aktif hybrid proxy topolojisini ve route erişilebilirliğini kanıtlar. Parser/adapter davranışı source-verified fixture ve deterministic contract testleriyle doğrulandı.

## Çalıştırılan komutlar ve sonuçları

| Kontrol | Sonuç |
|---|---|
| Faz 6-7 kritik regresyon testleri | PASS — 7 dosya, 27 test |
| Faz 8 hedefli Vitest | PASS — 6 dosya, 23 test |
| Faz 3 geri yükleme + Faz 8 birleşik Vitest | PASS — 7 dosya, 32 test |
| Model adı/capability + 502 politika testleri | PASS — 2 dosya, 11 test |
| Chat rezervasyon/geçmiş regresyon testleri | PASS — 2 dosya, 6 test |
| Chat dataset kapsamı regresyon testleri | PASS — 2 dosya, 8 test |
| Hybrid KNN/lexical ayrımı Go regresyon testi | PASS — `CGO_ENABLED=0 go test ./internal/engine/elasticsearch` |
| Native SSE hata görünürlüğü hedefli Vitest | PASS — runtime + sources, 2 dosya, 4 test |
| Referans görünürlüğü hedefli Vitest | PASS — parser, runtime, geçmiş adapter ve kaynak UI; 4 dosya, 21 test |
| Referans belge önizleme 502 regresyonu | PASS — platform/local route ayrımı, eski sohbet uyumluluğu, abort ve blob cleanup dahil; 6 dosya, 25 test |
| Response details veri bütünlüğü | PASS — live/history typed metadata, platform alanları ve secret redaction; 3 dosya, 11 test |
| Citation işareti normalizasyonu | PASS — canlı/geçmiş adapter ve marker kuralları; 3 dosya, 12 test |
| Geçmiş referans/ayarlar/502 regresyonları | PASS — 5 dosya, 15 test |
| Yerel/uzak sohbet kimliği alias regresyonu | PASS — 1 dosya, 4 test |
| Tam frontend Vitest | PASS — 53 dosya, 185 test |
| Backend model adı normalizasyon testi | PASS |
| TypeScript typecheck | PASS |
| ESLint | PASS — 0 error; tam kapsamda mevcut 77 warning |
| Production build | PASS — 8048 module; yalnızca mevcut chunk/dynamic-import uyarıları |
| npm audit | PASS — 0 vulnerability |
| i18n strict check | PASS — 0 missing |
| catalog check | PASS |
| source branding scan | PASS — 1213 TypeScript dosyası |
| build branding scan | PASS |
| proxy check | PASS — 368 Go route, 14 Python override |
| route inventory check | PASS — 711 route |
| endpoint coverage check | PASS — 821 kayıt, 0 unclassified |
| contract matrix check | PASS — 264 pair |
| `nginx -t` | PASS |
| `git diff --check` | PASS |
| Faz 8 authenticated runtime smoke | PASS — route/topology doğrulaması |

İlk `npm run type-check` çağrısı package script adı `typecheck` olduğu için komut-adı hatası verdi; doğru `npm run typecheck` çağrısı hemen tekrarlandı ve PASS oldu. İlk branding çağrısında göreli yol bir üst dizini fazla geçti; doğru `../../scripts/rag-platform/branding-scan.mjs` yolu ile source ve build taramaları PASS oldu. Bunlar test başarısızlığı değildir.

## Başarısız testler

Frontend ve saf backend regresyon testlerinde başarısız test yoktur. İlk host
`go test` çağrısı eksik tokenizer static library nedeniyle link aşamasında
durdu; aynı paket `CGO_ENABLED=0` ile PASS oldu. Güncel backend çalışma ağacını
sabit v0.26.4 imajına bütünüyle almak, release no-CGO adapter'ının daha yeni
`SetLanguage` API'sini taşımaması nedeniyle derlenmedi; runtime sürüm drift'i
yaratmamak için temiz v0.26.4 bağlamına yalnızca retrieval hotfix'i uygulanarak
imaj başarıyla üretildi.

## Bilinen sınırlamalar

- Yerel runtime `rag-platform-backend:0.26.4` sürümüne pinlidir; güncel backend kaynak şemasıyla arasındaki tenant-model alan farkı için canlı veride geri uyumlu composite model referansları korunmuştur. Güncel kaynak image'a alındığında idempotent migration integer bitmask/string-ID şemasını devralır.
- Tenant audio provider capability yapılandırılmadığından speech/transcription için canlı provider-success cevabı üretilemedi; route ve hata sözleşmeleri doğrulandı.
- Backend kaynaklarında server-side cancellation endpoint'i yoktur. Durdurma yalnızca istemci stream bağlantısını kapatır; UI bunu açıkça belirtir.
- Python chat-scoped completion alias'ı deprecated ve `api-only` bırakılmıştır; ürün UI'sı aktif Go completion route'unu kullanır.

## Runtime-disabled kayıtlar ve kanıtları

- Python `POST /api/v1/chat/completions`: kaynakta mevcut, fakat hybrid proxy aynı public path'i Go 9384'e yönlendirdiği için `proxy-shadowed`.
- Python `PUT /api/v1/chat/completions/{completion_id}`: kaynakta mevcut, fakat hybrid proxy Go 9384 eşdeğerini aktif tuttuğu için `proxy-shadowed`.
- Go `GET /api/v1/chats/{chat_id}/mindmap`: kaynakta mevcut, fakat hybrid proxy Python 9380 eşdeğerini aktif tuttuğu için `proxy-shadowed`.
- Go `GET /api/v1/chats/{chat_id}/recommendation`: kaynakta mevcut, fakat hybrid proxy Python 9380 eşdeğerini aktif tuttuğu için `proxy-shadowed`.

Kanıtlar backend kaynak route kayıtları, nginx hybrid proxy konfigürasyonu, proxy validator ve authenticated runtime smoke birlikte kullanılarak oluşturuldu.

## Sonraki faza geçiş

**Güvenli.** Faz 8 kabul kriterleri, önceki faz coverage bütünlüğü ve Global Definition of Done doğrulandı. Faz 9'a geçişi engelleyen kritik eksik bulunmuyor.

Commit veya push yapılmadı.
