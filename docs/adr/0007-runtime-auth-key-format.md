# ADR 0007: Runtime kimlik doğrulama anahtar formatı

- Durum: Kabul edildi
- Tarih: 2026-08-13
- Faz: 2

## Bağlam

Sabitlenmiş `v0.26.4` Python kimlik doğrulama uygulaması parola alanlarını tarayıcıda RSA ile şifrelenmiş olarak alır. Backend'in `decrypt()` uygulaması özel anahtarı her zaman `Welcome` parolasıyla `Crypto.PublicKey.RSA.importKey` üzerinden yükler. OpenSSL tarafından geçerli kabul edilen şifresiz PKCS#8 anahtar, PyCryptodome'a parola ile verildiğinde reddedilir. Bunun sonucu kayıt ve giriş isteklerinin kullanıcı kaydı oluşturulmadan önce `RSA key format is not supported` hatasıyla kesilmesidir.

Faz 1 çalışma zamanı giriş betiği şifresiz PKCS#8 üretmekteydi. Backend kaynak kodunu değiştirmek hem sabitlenmiş upstream sözleşmesinden sapacak hem de Python/Go hibrit çalışma zamanı yükseltmelerini zorlaştıracaktı.

## Karar

Rag Platform çalışma zamanı anahtar yöneticisi:

1. yeni 2048-bit RSA özel anahtarlarını AES-256-CBC ile şifrelenmiş PKCS#8 biçiminde ve backend'in sabit sözleşme parolasıyla üretir;
2. önceki şifresiz anahtarları aynı RSA materyalini koruyarak atomik biçimde şifreli PKCS#8'e geçirir;
3. her başlangıçta özel anahtarı doğrular ve açık anahtarı yeniden türetir;
4. dosya izinlerini özel anahtar için `0600`, açık anahtar için `0644` tutar;
5. hiçbir anahtar materyalini veya kullanıcı sırrını günlüğe yazmaz.

Frontend'e dağıtılan `VITE_PLATFORM_RSA_PUBLIC_KEY` yalnızca bu türetilen açık anahtardır. Özel anahtar frontend bundle'ına, tarayıcı depolamasına veya uygulama loglarına girmez.

## Sonuçlar

- Mevcut geliştirme çalışma alanları anahtar rotasyonu ve kullanıcı kaybı olmadan iyileşir; geçiş RSA materyalini değiştirmez.
- Yeni kurulumlarda kayıt, giriş, parola değiştirme ve parola sıfırlama Python sözleşmesiyle uyumlu olur.
- Backend'in sabit parola davranışı değişirse bu ADR ve runtime sözleşme kontrolü birlikte güncellenmelidir.
- Backend kaynak kodunda değişiklik yapılmamıştır.

## Doğrulama

`scripts/rag-platform/auth-key-contract.sh` çalışan container içinde:

- özel anahtarın beklenen biçimde olduğunu,
- PyCryptodome'un anahtarı backend ile aynı çağrı biçiminde yükleyebildiğini,
- özel anahtardan türetilen açık anahtar ile dağıtılan açık anahtarın SHA-256 özetlerinin eşleştiğini

anahtar içeriğini yazdırmadan doğrular.
