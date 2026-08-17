# ADR 0015: Faz 13 Memory, Search ve açık rıza sınırı

- Durum: Kabul edildi
- Tarih: 2026-08-17

## Bağlam

Hybrid runtime Memory ve Search ailelerini Python (9380) ile Go (9384) arasında bölüyor. `GET /memories/{id}` hafıza detayı değil mesaj sayfalama sözleşmesidir; tam ayar kaydı `GET /memories/{id}/config` ile alınır. Mesaj mutasyonları tek path segmentinde `memory_id:message_id` bileşiğini kullanır; hybrid proxy bu üç birleşik mesaj rotasını önce gelen açık override ile Python 9380'e yönlendirir, aynı sözleşmenin Go alternatifi runtime-disabled kalır. Search completion için `/completion` ve `/completions` aynı Go handler’ının aktif alias’larıdır.

Normal Chat completion sözleşmesinde `memory_id` alanı yoktur. Ayrıca Search için kalıcı geçmiş endpoint’i bulunmaz. Bu nedenle otomatik sohbet aktarımı veya server-side Search history varmış gibi davranmak backend sözleşmesini uydurmak olurdu.

## Karar

1. Memory ve Search için ayrı typed domain/adapter katmanı ve ayrı `/memory`, `/search` ürün yolları kullanılır.
2. Hafıza detayı config endpoint’inden, mesajlar ise mesaj sayfalama endpoint’inden ayrı yüklenir.
3. Sohbet içeriği yalnızca kullanıcı hafıza bazında açık rıza verdiğinde manuel `POST /messages` aksiyonuyla gönderilir. Rıza kaydı yalnızca hafıza kimliği ve boolean değer içerir; secret veya içerik saklamaz. Hafıza silindiğinde rıza kaydı da temizlenir.
4. Saklama sınırı ve runtime'ın desteklediği tek unutma politikası olan FIFO backend `PUT /memories/{id}` sözleşmesine; tek mesaj unutma aksiyonu backend’in `forget_at` soft-delete davranışına bağlanır. Python `ForgettingPolicy` ve Go Memory service LRU kabul etmediği için UI LRU seçeneği sunmaz.
5. Memory ve Search model seçimleri katalogdaki çıplak model adıyla değil, backend resolver’ın istediği `model@provider` veya named instance için `model@instance@provider` referansıyla kaydedilir.
6. Memory güncellemesi yalnızca başlangıç config’ine göre değişen alanları gönderir; değişmeyen model/tür/ad alanları backend’e yeniden yazılmaz.
7. Search kapsamı erişilebilir veri kümeleri ve tenant model kataloğundan seçilir; completion kaynak chunk’ları kullanıcıya gösterilir.
8. Search geçmişi yalnızca component state’inde, açıkça “bu oturum” etiketiyle tutulur; persistent store kullanılmaz.
9. UI çoğul `/completions` yolunu kullanır. Tekil alias aynı typed SSE adapter’ıyla tutulur ve contract testinde doğrulanır. Provider'ın cevap gövdesine sızdırdığı `[DONE]` terminal işareti metin olarak gösterilmez ve işaretten sonraki tekrar cevap parçaları yok sayılır; final reference frame yine işlenir.
10. Feature flag’ler implementasyonu ertelemez; yalnızca deployment policy kapatma anahtarıdır ve varsayılanları açıktır.

## Güvenlik ve yaşam döngüsü

- Bütün istekler merkezi authenticated platform client’tan geçer.
- 401/403, timeout, abort, network ve backend hata durumları ortak UI hata politikasına çevrilir.
- SSE stream unmount veya kullanıcı iptalinde abort edilir; reader ve timeout cleanup merkezi client/parser tarafından kapatılır.
- Token, provider key, kullanıcı/asistan içeriği ve Search soruları persistent store’a yazılmaz veya loglanmaz.
- Silme ve unutma aksiyonları kullanıcı onayı ister.

## Sonuçlar

Faz 13’teki bütün runtime-enabled Memory/Search rotaları UI yoluna veya aynı UI aksiyonunun alias contract’ına bağlanır. Runtime-disabled Python/Go alternate’ları route inventory ve coverage matrix’te kaynak/proxy gerekçesiyle kalır. Backend kaynak kodu değiştirilmez.

## Geri alma

`VITE_RAG_PLATFORM_MEMORY_ENABLED=false` veya `VITE_RAG_PLATFORM_SEARCH_ENABLED=false` ilgili navigasyon/route’u deployment policy kapsamında kapatır. Kod geri alımı gerekirse Faz 13 frontend dosyaları ve coverage evidence bloğu kaldırılır; backend veya kullanıcı verisi migrasyonu yoktur.
