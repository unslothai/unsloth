# Faz 3 provider/model contract ve runtime kanıtı

Bu belge Faz 3 uygulamasının yerel backend kaynak sözleşmesini ve runtime
kararlarını kaydeder. Sözleşme otoritesi
`/Users/baran/Desktop/rag-backend` `main` commit
`a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`; deploy envanteri ise plan gereği
`v0.26.4` (`cb93883f3f8c975eecb2fed81210effeb3bdb06f`) kaynağına sabittir.

## Domain ve güvenlik sınırı

- Provider instance yanıtındaki `api_key` domain modele alınmaz. Adapter yalnız
  `hasCredential: boolean` üretir; secret hiçbir Zustand/localStorage/cache
  alanına, loga veya geri gösterilebilir UI state'ine yazılmaz.
- API key girişi `type=password` ve `autocomplete=new-password` kullanır. Secret
  browser storage'a yazılmaz; başarılı create/edit sonrasında temizlenir. Draft
  connection-test sonrasında aynı bağlantının kaydedilebilmesi için yalnız form
  belleğinde kalır.
- Model capability listesi backend `model_type`/`model_types` alanından dinamik
  okunur. UI yalnız backend'in döndürdüğü capability ile eşleşen utility
  kontrolünü etkinleştirir ve uyumsuzluk nedenini gösterir.
- Varsayılan model mutation'ı optimistic değildir. `PATCH /models/default`
  tamamlandıktan sonra modeller/default listesi sunucudan yeniden okunur.
  Embedding değişimi kullanıcı onayı olmadan gönderilmez.
- Utility girdileri 32.000 karakter ve 10 MB ile sınırlandırılır; dosya MIME'ı
  araca göre doğrulanır. Her çağrı AbortSignal ve typed client timeout'u kullanır.
  Ses sonucu object URL'i değiştirilirken ve unmount'ta revoke edilir. Embedding
  UI'ı tam vektör yerine boyut, token sayısı ve ilk sekiz değeri gösterir.

## Canonical UI yolları

| Route grubu                                |      Runtime hedef | Typed servis                                                 | UI yolu                                                          |
| ------------------------------------------ | -----------------: | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| `GET/PUT /providers`                       |        Python 9380 | provider list/add                                            | Settings → Connections → Add connection → provider selector/form |
| provider detail/delete                     |            Go 9384 | `getProvider`, `deleteProvider`                              | Settings → Connections → configured provider row                 |
| provider instance CRUD                     |            Go 9384 | create/list/detail/update/delete                             | Add connection + configured provider inline edit/delete          |
| provider connection test                   |            Go 9384 | draft/instance test                                          | Add form test + configured provider row test                     |
| provider balance/tasks                     |            Go 9384 | balance, task list/detail                                    | `api-only`; typed contract test, no setup UI                     |
| provider/instance models                   |            Go 9384 | saved catalog + `supported=true` live discovery + model CRUD | Configured provider expandable workspace → Models                |
| `GET /models`, `GET/PATCH /models/default` |        Python 9380 | model/default typed servisleri                               | Configured provider workspace → capability defaults/readiness    |
| `GET /users/me/models`                     |        Python 9380 | tenant model mapper                                          | role/permission ve readiness state                               |
| chat/embedding/rerank utilities            |            Go 9384 | typed utility servisleri                                     | Configured provider workspace → Yetkili model araçları           |
| transcription/speech/OCR/parse utilities   |            Go 9384 | typed utility servisleri                                     | Configured provider workspace → Yetkili model araçları           |
| pipeline list/detail                       | worktree Go source | typed pipeline servisleri                                    | Dataset create/edit selector + configured provider workspace     |

Provider/instance model route'ları Go 9384'te erişilebilirdir; ancak pinned Go
image'ın statik provider kataloğu local Python `llm_factories.json` kataloğundan
dardır. `OpenAI-API-Compatible` provider-level model listesi bu runtime'da
business error döndürdüğünden bu çağrı yalnız enrichment'tır: Settings yüklemesi
başarısız olmaz ve UI `GET /providers/:provider/instances/:instance/models`
sonucunu göstermeye devam eder. Aynı route'un `supported=true` biçimi saklı
instance credential'ını server tarafında kullanarak provider'ın canlı model
kataloğunu getirir; secret browser'a geri dönmez. Statik provider kataloğu ve
persist edilmiş instance listesi fallback/enrichment olarak kalır. Local Python worktree'deki eşdeğer yeni REST
route'u pinned Python image'da 404 olduğundan proxy yanlış hedefe çevrilmez.
Yeni custom OpenAI-compatible endpoint UI seçeneği pinned runtime'da mevcut
`VLLM` provider ID'sini kullanır. Typed adapter, Python `_normalize_provider_base_url`
sözleşmesini eşleyerek eksik `/v1` suffix'ini create/update/draft-test
payload'larına ekler; secret yalnız geçici component state → Authorization
işlemi akışındadır ve browser storage'a yazılmaz.

`PATCH /models`, `PATCH /users/me/models`, Python batch model PUT ve Python
model-path POST aynı kullanıcı yeteneğinin bulk/compatibility biçimleridir.
Canonical UI atomik `/models/default`, instance-model POST/PATCH ve
`/chat/to_model` sözleşmesini kullanır; alias'lar `api-only / contract-verified`
olarak test edilir ve ikinci bir state machine oluşturmaz.

Balance ve asynchronous task telemetry route'ları bağlantı/model kurulumunun
parçası değildir. Connections yüzeyini operasyon dashboard'una çevirmemek için
UI'da gösterilmez; typed servisleri ve exact response contract testleri korunur
ve matrix'te `api-only / contract-verified` olarak sınıflandırılır.

Model ekleme endpoint'inin payload sözleşmesi deploy runtime'ları arasında
değişmiştir: backend main tekil `model_name`/`model_type`, pinned `v0.26.4` Go
runtime ise `models[]` içinde `model_name`/`model_types` bekler. Typed adapter iki
şekli aynı istekte gönderir; her iki Go decoder da tanımadığı alanları yok sayar.
Model silmede de aynı nedenle `model_name[]` ve `models[]` birlikte gönderilir.
Contract testleri bu mixed-runtime uyumluluk gövdesini sabitler.

## Readiness ve dataset mapping

- Chat aksiyonu, backend default listesinde etkin `chat` kaydı ve aynı kimlikte
  `chat` capability'li tenant modeli bulunmadan çalışmaz; kullanıcı Connections
  sekmesine yönlendirilir.
- Dataset create aksiyonu aynı doğrulamayı `embedding` için yapar.
- Pipeline seçimi backend dataset create sözleşmesine tam olarak
  `{ pipeline_id: <id>, parse_type: 2 }` biçiminde map edilir. `parser_id`
  pipeline modunda gönderilmez. Mapping `internal/service/dataset_types.go` ve
  `internal/service/dataset/crud.go` kaynaklarından doğrulanmıştır.

## Runtime-disabled pipeline kaydı

Yerel backend `main`, `internal/router/router.go:170-171` içinde public
`GET /api/v1/pipelines` ve `GET /api/v1/pipelines/:id` route'larını kaydeder;
handler'lar `internal/handler/pipeline.go` içinde gerçek registry list/detail
uygulamasıdır. Pinned `v0.26.4` kaynakta bu route'lar yoktur.

2026-08-13 canlı hybrid nginx smoke sonucu:

| Probe                                                                               |                              Sonuç |
| ----------------------------------------------------------------------------------- | ---------------------------------: |
| `GET http://127.0.0.1/api/v1/pipelines`                                             |                           HTTP 404 |
| `GET http://127.0.0.1/api/v1/pipelines/general`                                     |                           HTTP 404 |
| `GET http://127.0.0.1/api/v1/providers` (auth olmadan)                              |                           HTTP 401 |
| `GET http://127.0.0.1/api/v1/providers/OpenAI-API-Compatible/models` (auth olmadan) | HTTP 401; Go auth katmanına erişir |
| `POST http://127.0.0.1/api/v1/chat/to_model` (auth olmadan)                         |                           HTTP 401 |

İlk iki sonuç pipeline route'unun deploy runtime'ında bulunmadığını; son iki
sonuç Phase 3 canonical route'larının nginx üzerinden doğru auth katmanına
ulaştığını gösterir. Pipeline UI'ı sahte seçenek üretmez ve açık
`runtime-disabled` nedeni gösterir. Route bazlı kaynak/proxy kayıtları
`route-inventory.json` ve `runtime-disabled.md` içinde üretilir.

## Otomatik kanıt

- `model-api.test.ts`: provider instance CRUD, secret redaction, task/balance,
  draft + saved connection test, `supported=true` live model discovery, model
  CRUD/default, yedi utility, pipeline list/detail, error ve abort contract'ları.
- `model-readiness.test.ts`: chat/embedding readiness ve exact dataset pipeline
  mapping.
- `platform-model-tools.test.tsx`: permission, mixed-runtime provider katalog
  izolasyonu + instance model fallback'i, saved connection test + remote model
  discovery + model-add kullanıcı yolu, capability mismatch, tam vektör
  redaction, MIME/boyut doğrulaması ve object URL cleanup.
- `client.test.ts`: ortak timeout, abort, retry ve cleanup sözleşmesi.
