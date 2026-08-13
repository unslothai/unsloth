# Faz 2 Sonuç Raporu

> Durum: **COMPLETE.**
> Yalnız Faz 2 uygulandı; Faz 3 başlatılmadı.
> Backend kaynak repository'sine dokunulmadı ve mevcut kullanıcı değişiklikleri
> korundu.

## Yapılan değişiklikler

- Tek opaque bearer token kullanan native Rag Platform oturumu eklendi. Token
  yalnız `rag-platform.auth-token` anahtarında tutulur; refresh token veya
  `/auth/refresh` çağrısı yoktur.
- RSA PKCS#1 v1.5 parola wire formatı backend sözleşmesiyle aynı biçimde
  uygulandı ve public key yoksa/geçersizse fail-closed davranır.
- `platformRequest` platform token'ını otomatik ekler; 401'de oturumu bir kez
  temizleyip `/login`'e yönlendirir. Mutasyonlar retry edilmez; timeout, abort ve
  event-listener/timer cleanup korunur.
- Login, doğrudan kayıt, dört adımlı parola kurtarma, runtime capability probe,
  OAuth kanal seçimi/callback bridge, profil ve parola değiştirme typed service
  + domain mapper üzerinden bağlandı. Aktif workspace typed servisleri ve
  contract testleri korunur; Profile UI'dan kaldırılan tenant-model seçiminin
  görsel yerleşimi ürün sahibi onayıyla Faz 2 kapanışını engellemez.
- OAuth callback'te query içindeki `auth` yalnız işaretleyici kabul edilir;
  credential sadece backend'in SameSite cookie'sinden okunur ve hemen silinir.
- Guest/protected route guard'ları platform oturum hidrasyonu, offline hata ve
  yönlendirme döngüsü olmadan çalışacak biçimde tamamlandı. Tauri'nin legacy
  auto-auth davranışı yalnız platform auth açıkken devre dışı kalır.
- Runtime RSA özel anahtarı backend'in PyCryptodome sözleşmesine uygun, şifreli
  PKCS#8 biçimine geçirildi. Eski şifresiz anahtar aynı RSA materyali korunarak
  atomik olarak migrate edilir; public half her başlangıçta yeniden türetilir.
- Route inventory, endpoint coverage matrix, contract matrix, runtime-disabled
  kaydı, ADR 0002 ve yeni ADR 0007 güncellendi.
- Plain Vite başlangıcında env dosyası kopyalanmadığında `/api/v1` isteklerinin
  legacy 8888 proxy'sine düşmesi engellendi; platform auth açıkken varsayılan
  hedef aktif hibrit nginx `127.0.0.1:80` oldu.
- Platform modunda karşılığı ve tüketicisi olmayan legacy Studio `/api/health`
  donanım ve native-path polling döngüleri kapatıldı. Connections readiness
  kontrolü oturumda bir kez otomatik çalışır; kullanıcı isterse manuel olarak
  yeniden kontrol edebilir.
- Sol menüdeki haricî görsel marka kaldırıldı; merkezî branding kaynağından
  yalnız `Rag Platform` yazı kilidi gösterilir. Settings yüzeyinin masaüstü
  yüksekliği çalışan `localhost:8888` referansındaki 680 px sınırına döndürüldü;
  kısa viewport ve mobil taşma davranışı korunur.

## Frontend ekranları ve aksiyonları

- **Login → Giriş yap:** e-posta/parola, RSA wire encryption, header token
  extraction, loading/error/timeout/abort.
- **Login → Hesap oluştur:** runtime `registerEnabled` ile görünür; nickname,
  e-posta, parola doğrulama ve otomatik oturum açma.
- **Login → Parolamı unuttum:** captcha görseli ve yenileme, OTP gönderme,
  OTP doğrulama, yeni parola ve otomatik oturum açma. Object URL ve request
  cleanup uygulanır.
- **Login → Kurumsal giriş:** yalnız canlı `/auth/login/channels` yanıtındaki
  provider'lar gösterilir; boş kanal listesi dead button üretmez.
- **Root OAuth bridge:** state/error sonucu güvenli hata koduna çevrilir; cookie
  token alınır, query temizlenir ve yalnız `/chat` veya `/login` seçilir.
- **Ayarlar → Profile → profil kimliği:** e-posta görünen adın hemen altında;
  aynı form ritmindeki salt-okunur input içinde gösterilir. Oluşturulma ve son
  güncelleme tarihleri aynı blokta küçük, soluk bir metadata satırındadır. Ayrı
  hesap kartı yoktur. Veriler korumalı route hidrasyonunun canonical `/users/me`
  oturum kaydından gelir; tarihler backend `create_time`/`update_time`
  alanlarından biçimlendirilir.
- **Ayarlar → Profile → kişiselleştirme:** görünen ad ve avatar canonical
  `/users/me` profilinden hydrate edilir ve aynı typed servisle kaydedilir;
  backend'de ikinci bir ad alanı olmadığı için yerel-only takma ad girdisi
  platform modunda gösterilmez. Avatar şekli yalnız sunum tercihi olarak yerel
  kalır.
- **Ayarlar → Change password:** mevcut ve yeni parolayı backend sözleşmesiyle
  gönderir; başarılı değişiklikten sonra oturumu temizler ve login'e döner.
- **Hesap menüsü → Log out:** önce server-side revoke dener, her durumda yerel
  oturumu temizler.
- **Uygulama kabuğu → marka ve Settings:** sol menü üstünde görselsiz
  `Rag Platform` wordmark'ı bulunur; Settings popup'ı masaüstünde en çok 680 px
  yüksekliğinde, içerik alanları kendi scroll sınırları içinde çalışır.

UI component'lerinden doğrudan network çağrısı yapılmaz; yollar
`component → typed auth service → platformRequest → backend` biçimindedir.

## Kullanılan backend endpoint'leri

| Endpoint | Aktif hedef | Sınıf / durum | Typed servis ve UI yolu |
|---|---:|---|---|
| `POST /api/v1/auth/login` | Python 9380 | frontend-action / implemented | `loginPlatformUser`; Login |
| `POST /api/v1/auth/logout` | Python 9380 | frontend-action / implemented | `logoutPlatformUser`; account menu |
| `GET /api/v1/system/config` | Python 9380 | frontend-action / implemented | `getPlatformAuthCapabilities`; Login |
| `GET /api/v1/auth/login/channels` | Go 9384 | frontend-action / implemented | capability probe; OAuth buttons |
| `GET /api/v1/auth/login/:channel` | Go 9384 | frontend-action / implemented | `getPlatformOAuthLoginUrl`; provider redirect |
| `GET /api/v1/auth/oauth/:channel/callback` | Go 9384 | external-callback / implemented | root OAuth bridge |
| `POST /api/v1/users` | Python 9380 | frontend-action / implemented | `registerPlatformUser`; Hesap oluştur |
| `POST /api/v1/auth/password/forgot/captcha` | Python 9380 | frontend-action / implemented | recovery captcha |
| `POST /api/v1/auth/password/forgot/otp` | Python 9380 | frontend-action / implemented | recovery OTP send |
| `POST /api/v1/auth/password/forgot/otp/verify` | Python 9380 | frontend-action / implemented | recovery OTP verify |
| `POST /api/v1/auth/password/reset` | Python 9380 | frontend-action / implemented | recovery reset/login |
| `GET /api/v1/users/me` | Python 9380 | frontend-action / implemented | hydration + settings |
| `PATCH /api/v1/users/me` | Python 9380 | frontend-action / implemented | profile + password |
| `GET /api/v1/users/me/models` | Python 9380 | api-only / contract-verified | typed service ve contract testi var; UI yerleşimi ürün sahibi onayıyla ertelendi |
| `PATCH /api/v1/users/me/models` | Python 9380 | api-only / contract-verified | typed service ve contract testi var; UI yerleşimi ürün sahibi onayıyla ertelendi |

Altı aktif Go `/v1/user/*` compatibility route'u `api-only / contract-verified`
olarak tutulur. UI canonical `/api/v1` sözleşmesini kullanır; ikinci bir auth
state machine oluşturulmaz.

## Route coverage sonucu

- Inventory: **709** top-level route.
- Coverage: **819** kayıt; erişilebilir **516**; `unclassified=0`.
- Runtime-disabled: **188**; bunların **7** tanesinde somut isteği karşılayan
  erişilebilir eşdeğer yoktur, **181** tanesi başka aktif route tarafından
  karşılanır.
- Faz 2: **45** kayıt:
  - `frontend-action / implemented`: 12
  - `api-only / contract-verified`: 8
  - `external-callback / implemented`: 1
  - `unsupported / runtime-disabled`: 24
  - `planned` veya `in-progress`: 0
- Implemented her Faz 2 kaydında typed service, gerçek UI yolu ve otomatik test
  kanıtı matriste bulunur.

## Runtime-disabled kayıtlar ve kanıt

Faz 2'deki 24 runtime-disabled kaydın 15'i hibrit proxy'nin diğer çalışan
Python/Go implementasyonunu seçtiği duplicate sözleşmelerdir; capability kaybı
yoktur. Kalan dokuz route yalnız backend `main` içindeki
`internal/router/router_ee.go` dosyasında bulunur, sabitlenmiş `v0.26.4` image'da
yoktur ve `user_auth_ee.go` handler'ları `CodeNotImplemented` döndürür.

| Source-only route grubu | Canlı hibrit smoke | Karar |
|---|---|---|
| OAuth generic, ICBC, Azure callback/login (4) | HTTP 404 | runtime-disabled; UI yok |
| Registration captcha/OTP/verify (3) | HTTP 404 | runtime-disabled; UI doğrudan aktif kayıt sözleşmesini kullanır |
| GitHub ve Lark statik callback (2) | HTTP 302 | statik EE stub disabled; somut URL aktif `oauth/:channel/callback` tarafından karşılanır |

Bu ayrım envanter algoritmasına işlendi: statik source-only URL'yi yakalayan
aktif parameter route artık reachable equivalent sayılır. Ayrıntılı kaynak,
proxy hedefi ve route bazlı kayıt `runtime-disabled.md` ile
`phase-2-auth-contract.md` içindedir.

## Test ve doğrulama kanıtı

| Komut / kontrol | Sonuç |
|---|---|
| `npm run typecheck` | PASS |
| `npm run lint:all` | PASS; 0 hata |
| `npm test` | PASS; 18 dosya, 57/57 test |
| `npm run i18n:check:strict` | PASS |
| `npm run catalog:check` | PASS |
| `npm run build` | PASS; Vite production build |
| `node scripts/rag-platform/branding-scan.mjs --build` | PASS; source + production artifact |
| Route inventory `--check` | PASS; 709 route |
| Coverage matrix `--check` | PASS; 819 kayıt, `unclassified=0` |
| Contract matrix `--check` | PASS; 272 scanned pair |
| `auth-key-contract.sh` | PASS; encrypted PKCS#8 + eşleşen public half |
| Hibrit servis health smoke | PASS; Python 9380/9381 ve Go 9383/9384 dinlemede |
| In-app browser register → reload/hydrate → logout | PASS |
| Dokuz forward auth route live smoke | PASS; 7×404, 2×302 dynamic equivalent |
| Codebase Memory Auditor coverage | PASS with best-effort caveat; bildirilen parser aralıkları direct source ile okundu |
| `git diff --check` | PASS |

Gerçek browser smoke'ta benzersiz test kullanıcısı oluşturuldu, `/chat` geçişi,
tam navigasyon sonrası `/users/me` hidrasyonu ve UI logout doğrulandı. Ardından
yalnız exact test e-postasına ait 1 root file, 1 user-tenant link, 1 tenant ve 1
user kaydı tek transaction ile silindi.

### Başarısız testler

Final durumda başarısız test veya kabul kriteri yoktur. Geliştirme sırasında:

- fixture beklentileri ve cross-realm Blob assertion'ları test sözleşmesine göre
  düzeltildi;
- yeni abort regresyon testindeki erişilebilir buton adı düzeltildi;
- ilk live kayıt smoke'u şifresiz PKCS#8 anahtar nedeniyle backend RSA import
  hatasını ortaya çıkardı; ADR 0007'deki runtime anahtar migration'ı uygulandı,
  image yeniden oluşturuldu ve gerçek kayıt akışı geçti.

Her düzeltmeden sonra ilgili testler ve final tam kalite kapıları yeniden
çalıştırıldı.

## Değiştirilen dosyalar

- Auth transport/domain:
  - `studio/frontend/src/integrations/platform-backend/{auth-api,auth-crypto,auth-session,auth-types,client,config,index}.ts`
  - `studio/frontend/src/integrations/platform-backend/backend-connection-status.tsx`
- Auth ve settings UI:
  - `studio/frontend/src/components/app-sidebar.tsx`
  - `studio/frontend/src/features/auth/components/{platform-auth-form,auth-form}.tsx`
  - `studio/frontend/src/features/profile/components/profile-personalization-panel.tsx`
  - `studio/frontend/src/features/auth/{api,platform-auth-errors}.ts`
  - `studio/frontend/src/features/settings/components/change-password-dialog.tsx`
  - `studio/frontend/src/features/settings/settings-dialog.tsx`
  - `studio/frontend/src/features/settings/tabs/general-tab.tsx`
  - `studio/frontend/src/app/auth-guards.ts`
  - `studio/frontend/src/app/provider.tsx`
  - `studio/frontend/src/app/routes/__root.tsx`
  - `studio/frontend/src/config/env.ts`
  - `studio/frontend/src/features/native-intents/use-native-readiness.ts`
- Testler:
  - `studio/frontend/src/integrations/platform-backend/__tests__/{auth-api,auth-crypto,auth-guards,auth-session,oauth,platform-auth-errors,platform-auth-form,config}.test.ts(x)`
  - `studio/frontend/src/integrations/platform-backend/__tests__/backend-connection-status.test.tsx`
  - `studio/frontend/src/features/settings/tabs/profile-tab.test.tsx`
  - `studio/frontend/src/features/profile/{components/profile-personalization-panel,hooks/use-platform-profile-sync}.test.tsx`
  - `studio/frontend/src/config/env.test.ts`
  - `studio/frontend/src/features/native-intents/use-native-readiness.test.ts`
- Runtime/config:
  - `studio/frontend/.env.example`
  - `studio/frontend/vite.config.ts`
  - `infra/rag-platform/backend-entrypoint.sh`
  - `scripts/rag-platform/auth-key-contract.sh`
- Governance ve dokümantasyon:
  - `scripts/rag-platform/{route-inventory,coverage-matrix,branding-scan}.mjs`
  - `docs/adr/{0002-platform-auth-strategy,0007-runtime-auth-key-format}.md`
  - `docs/rag-platform/{route-inventory,endpoint-coverage-matrix}.{md,json}`
  - `docs/rag-platform/{runtime-disabled,contract-matrix,phase-2-auth-contract,faz-2-sonuc-raporu}.md`

Backend repository'de dosya değiştirilmedi. Başlangıçta var olan `.gitignore`
değişikliği, iki PEM silme kaydı ve governance workflow dosyası aynen korundu.

## Bilinen sınırlamalar

- Canlı deployment `login/channels` boş döndürdüğü için gerçek üçüncü taraf
  provider round-trip E2E yapılamadı; OAuth state, cookie-only credential,
  callback error ve sabit return-path sözleşmeleri unit/contract testleriyle
  doğrulandı ve UI dead provider göstermedi.
- Parola kurtarma zinciri gerçek e-posta teslimatıyla E2E edilmedi; dört endpoint
  contract testleri ve UI state-machine testleriyle kapsandı.
- Auth smoke sonrası chat sayfasındaki model refresh çağrıları 502 gösterdi. Bu
  Faz 2 auth oturumunun başarısızlığı değildir; model/chat backend entegrasyonu
  sonraki bağlayıcı fazların kapsamındadır.
- Platform token backend sözleşmesi gereği `localStorage` içindedir ve XSS
  riskini taşır. TLS, CSP ve dependency hardening Faz 15 release kapısında ayrıca
  zorunludur; parola, private key, provider key veya refresh token persist edilmez.
- Production build büyük chunk ve ineffective dynamic import warning'leri verir;
  build exit kodu 0'dır ve Faz 2'ye özgü hata yoktur.
- Faz 2 yeni dosyalarında lint hatası veya unused-disable warning'i yoktur.

## Sonraki faz kapısı

Aktif çalışma alanı/tenant-model endpoint'lerinin typed servis ve contract
testleri korunmuştur. Profile dışındaki görsel yerleşimin ertelenmesi ürün sahibi
tarafından açıkça kabul edildiğinden Faz 2 **COMPLETE** durumundadır ve sonraki
çalışmaya geçiş güvenlidir. Bu rapor Faz 3 uygulamasını başlatmaz.
