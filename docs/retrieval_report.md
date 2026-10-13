# Retrieval Raporu: İndeksleme ve Geri Getirme (İP2–İP3)

Kapsam: `backend/indexing/` (chunker, embedder, vectorstore) ve `backend/retrieval/` (retriever, reranker, bm25).
Ham ölçüm tabloları `docs/retrieval_results/` altındadır. Hepsi `scripts/retrieval_bench.py` ile yeniden üretilebilir.

## 1. Bitti kriterleri: durum

| Kriter | Durum | Kanıt |
|---|---|---|
| 10K parçada filtreli arama < 500 ms | ✅ **p50 ≈ 10–15 ms, p95 ≈ 14–20 ms** (vektör araması) | §4.1, `store_hnsw.md` |
| Aynı bölümün parçaları farklı bölümlerden daha benzer (iç tutarlılık) | 🟡 Test ve script hazır; gerçek materyalle **laboratuvar makinesinde çalıştırılacak** | §4.5, `tests/test_retrieval.py` |
| `docs/retrieval_report.md` içinde ölçüm tablosu | ✅ Bu belge. 🟡 Model gerektiren satırlar (embedding hızı, chunk/boost denemeleri) makinede doldurulacak | §4 |
| Gerçek tokenizer'a geçiş | ✅ BGE-m3 tokenizer'ı, cümle sınırına duyarlı kesim | §2.1 |
| GPU kullanımı | ✅ `embedding_device: auto` (cuda/mps/cpu), cuda'da fp16 | §2.2 |
| HNSW parametreleri | ✅ Izgara taraması yapıldı, M=16 / ef_construction=100 / ef_search=100 seçildi | §4.1 |
| timestamp_boost ayarı | 🟡 Yumuşak (mesafeye göre azalan) boost eklendi; değer taraması etiketli soru seti ile yapılacak | §2.4, §4.6 |
| BM25 + vektör birleşik arama (R2 B planı) | ✅ RRF ile hibrit arama, `hybrid: true/false` anahtarı | §2.3, §4.3 |

> **Not:** Bu ortamda HuggingFace ve PyTorch indirmeleri engelli olduğu için BGE-m3 ve reranker çalıştırılamadı.
> ChromaDB/HNSW ölçümleri gerçek ChromaDB (1.5.9) üzerinde **1024 boyutlu sentetik kümelenmiş vektörlerle** yapıldı.
> Model gerektiren ölçümler için §6'daki komutlar laboratuvar makinesinde çalıştırılmalıdır.

## 2. Yapılan değişiklikler

### 2.1 Chunker (`backend/indexing/chunker.py`)
- **Kelime × 1.3 tahmini kaldırıldı.** Boyutlar artık embedding modelinin kendi tokenizer'ıyla (BGE-m3 / XLM-R SentencePiece, `transformers` fast tokenizer) ölçülüyor. `chunk_size: 500` = 500 gerçek model token'ı.
  Türkçe sondan eklemeli bir dil olduğu için eski tahmin sistematik olarak sapıyordu: XLM-R'de bir Türkçe kelime genelde 1.3'ten fazla parçaya bölünür. Bunun sonucu, parçaların hedeflenenden uzun çıkması ve `max_seq_length` sınırına yaklaşmasıydı.
- Kesim token offset'leri üzerinden yapılıyor, yani orijinal metin (boşluk, satır sonu, noktalama) korunuyor.
- **Cümle sınırına duyarlı kesim** (`sentence_aware: true`): pencere sonuna yakın bir cümle sonu (. ! ? ; : veya satır sonu) varsa kesim oraya çekiliyor. Bir parça en fazla %25 kısalabiliyor.
- Video parçalamada bütçe token ile hesaplanıyor, zaman damgaları ve doküman sınırları korunuyor.
- Tokenizer yüklenemezse (internetsiz makine, HF önbelleği yok) kelime+noktalama regex'ine düşüyor ve bir kez uyarı loglanıyor.

### 2.2 Embedder ve vectorstore
- `embedding_device: auto` ayarı: önce cuda, sonra mps, en son cpu deneniyor. cuda'da `embedding_fp16: true` ile model yarım hassasiyete alınıyor (yaklaşık 2 kat hız, yarı VRAM). GPU için ayrı `batch_size_gpu: 64`.
- `device_info()`: ölçümler hangi donanımda alındığını otomatik raporluyor.
- ChromaDB ≥1.0 tipli `configuration` API'sine geçildi (`max_neighbors`, `ef_construction`, `ef_search`). Eski sürümler için legacy `hnsw:*` metadata fallback'i var.
- `hnsw_search_ef` config'e eklendi. **Önemli bulgu:** Chroma `ef_search`'ü koleksiyon yüklenirken uyguluyor. `collection.modify()` ile çalışma anında yapılan değişiklik **aynı süreçte etkisiz kalıyor, yeniden başlatma gerekiyor.** İlk ölçüm turunda bu yüzden tüm satırlar fiilen aynı ef değeriyle koşmuştu. Script artık her değer için ayrı koleksiyon kuruyor.
- Embedding'ler numpy olarak gönderiliyor (`.tolist()` dönüşümü kaldırıldı). Batch boyutu istemcinin `get_max_batch_size()` değerine göre ayarlanıyor.
- Store'a bir `version` sayacı eklendi, her yazma/silmede artıyor. BM25 önbelleği ve filtreli `count()` önbelleği bununla geçersiz kılınıyor. `count()` her sorguda çağrıldığı için önbelleğe alındı.

### 2.3 Hibrit arama (`backend/retrieval/bm25.py`, `retriever.py`)
- **Neden:** Yoğun (dense) arama "Nyquist", "CENG384", formül/terim adları gibi tam eşleşme gerektiren sorgularda zayıf kalabilir (formdaki risk R2).
- **BM25:** Okapi BM25, sparse terim matrisi üzerinde (scikit-learn), ek bağımlılık yok.
  - Türkçe normalizasyon: İ/I sorunu çözülüyor, diakritikler katlanıyor (ç→c, ı→i …) ki ASCII yazılan sorular da eşleşsin.
  - **Önek kökleme (prefix stemming):** her kelimenin ilk 5 karakteri alınıyor. Sabit önek kırpmanın Türkçe bilgi erişiminde morfolojik köklemeye yakın sonuç verdiği bilinmektedir (Can vd., 2008). Değer `bm25_prefix_len` ile ayarlanabilir, 0 ile kapatılır.
- **Birleştirme:** Reciprocal Rank Fusion (RRF, k=60; Cormack vd., 2009). Dense ilk-20 ve BM25 ilk-20 birleştirilip en iyi 20 aday reranker'a gidiyor. Böylece reranker maliyeti artmıyor.
- BM25 indeksi kurs/ders havuzu başına bir kez kurulup önbellekte tutuluyor. Yeni materyal indekslenince otomatik yeniden kuruluyor.
- `hybrid: false` varsayılanı korundu. Etiketli soru setiyle (§4.6) dense ve hybrid karşılaştırıldıktan sonra açılması önerilir.

### 2.4 Timestamp boost
- Eskiden boost sadece izlenen aralıkla **çakışan** video parçalarına veriliyordu (keskin içeride/dışarıda). Aralığın 10 saniye dışındaki parça hiç boost almıyordu.
- Artık: aralık içinde tam boost, dışarıda `timestamp_decay_ms` (varsayılan 2 dk) boyunca doğrusal azalarak 0'a iniyor. `timestamp_decay_ms: 0` eski davranışı verir.
- Boost reranker'ın 0..1 olasılığına ekleniyor. Bunun tutarlı olması için reranker düzeltmesi (§2.5) gerekliydi.

### 2.5 Reranker düzeltmesi (çekirdek hata)
- Eski kod sigmoid uygulanıp uygulanmayacağına **her sorguda** `min < 0 or max > 1` kontrolüyle karar veriyordu. Bir sorgunun tüm logit'leri tesadüfen 0..1 aralığına düşerse ham logit, diğer sorgularda olasılık kullanılıyordu. Sabit büyüklükteki timestamp boost'u bu yüzden sorgudan sorguya farklı ağırlıkta etki ediyordu.
- Artık karar model yüklenirken **bir kez** veriliyor (CrossEncoder'ın kendi aktivasyon fonksiyonuna bakılarak). Cihaz, fp16, `reranker_batch_size` ve `reranker_max_length` config'den okunuyor.
- `retrieve()` artık aşama sürelerini (`timings`: embed_ms, search_ms, bm25_ms, rerank_ms) ve modu (`mode`) döndürüyor. Arayüz/benchmark nerede zaman harcandığını görebilir.

## 3. Ölçüm ortamı

| Makine | Rolü | Kullanıldığı ölçümler |
|---|---|---|
| Bulut sandbox (2 vCPU, GPU yok) | Model gerektirmeyen ChromaDB ölçümleri | §4.1–4.3 |
| Geliştirme laptopu: RTX 3050 Laptop 4 GB, 32 GB RAM, Python 3.12 | Geliştirme, ilk model ölçümleri | §4.4 |
| Ana makine: RTX 5070 8 GB | Sunum ve nihai ölçümler | §4.4 (doldurulacak), §4.5–4.6 |

Nihai rapor rakamları ana makineden alınmalıdır; diğerleri karşılaştırma ve alt sınır olarak tutulur.

### Bu turdaki sandbox ölçümleri
Linux x86_64, 2 vCPU, 7 GB RAM, GPU yok, Python 3.13, chromadb 1.5.9.
Veri: 1024-d sentetik vektörler (5 kurs × 14 ders, kurs merkezi + ders sapması + gürültü), 100 sorgu, k=20.
Recall@20, aynı filtre altında numpy ile kaba kuvvet (tam) aramaya göre hesaplandı.

## 4. Sonuçlar

### 4.1 HNSW ızgarası, 10.000 parça (`store_hnsw.md`)
Filtreli arama p50 gecikmesi (ms) ve recall@20:

| M | ef_construction | ef_search | recall@20 | filtresiz p50 | kurs filtresi p50 | kurs+ders filtresi p50 | yazma (parça/sn) |
|---|---|---|---|---|---|---|---|
| 16 | 100 | 20 | 0.960 | 5.0 | 15.9 | 12.5 | ~1255 |
| 16 | 100 | 50 | 0.998 | 4.8 | 18.1 | 10.2 | ~1255 |
| **16** | **100** | **100** | **1.000** | **5.5** | **15.9** | **11.5** | **~1255** |
| 16 | 100 | 200 | 1.000 | 5.9 | 22.0 | 21.6 | ~1255 |
| 16 | 200 | 100 | 1.000 | 5.7 | 20.5 | 13.6 | ~1070 |
| 32 | 100 | 100 | 1.000 | 6.2 | 23.2 | 12.7 | ~1160 |
| 32 | 200 | 100 | 1.000 | 5.7 | 22.4 | 13.3 | ~995 |

**Yorum:**
- Recall'ü belirleyen parametre `ef_search`: 20'de ~%96, 50'de ~%99.8, 100'de %100. M veya ef_construction'ı artırmak bu ölçekte recall kazandırmıyor, sadece yazmayı %10–20 yavaşlatıyor.
- **Seçim:** M=16, ef_construction=100, ef_search=100 (config varsayılanları). 500 ms hedefinin 30–40 kat altında.
- Filtreli arama filtresizden yavaş (≈5 ms yerine ≈12–16 ms), çünkü Chroma önce meta veri filtresini uyguluyor. Yine de ihmal edilebilir düzeyde.

### 4.2 Ölçekleme, M=16 / efc=100 / ef=100 (`store_scale.md`)

| Parça | yazma (parça/sn) | kurs havuzunu okuma (ms) | filtresiz p50 / p95 | kurs filtresi p50 / p95 | kurs+ders p50 / p95 | recall@20 |
|---|---|---|---|---|---|---|
| 1.000 | 1461 | 23 | 3.1 / 5.7 | 5.2 / 7.6 | 5.7 / 11.6 | 1.000 |
| 10.000 | 1202 | 234 | 5.3 / 8.2 | 14.5 / 19.4 | 9.8 / 14.0 | 1.000 |
| 50.000 | 1096 | 1136 | 10.6 / 20.0 | 62.9 / 82.6 | 40.7 / 53.8 | 0.987 |

50K'da bile filtreli arama p95 < 85 ms. Recall %98.7'ye iniyor. 100K+ için `ef_search: 150–200` denenmeli (İP4 benchmark'ı).

### 4.3 Hibrit aramanın maliyeti (10K parça, reranker kapalı, sahte sorgu embedding'i)

| Mod | ilk sorgu (ms) | önbellekten sonra p50 / p95 (ms) |
|---|---|---|
| dense | 45 | 26.6 / 35.3 |
| hybrid (BM25+dense, RRF) | 1424 (BM25 indeksi kuruluyor) | 26.1 / 32.6 |

- BM25 sorgusu ≈0,2 ms, hibrit modun sorgu başına ek maliyeti pratikte sıfır.
- Tek seferlik indeks kurulumu: 10K parça ≈ 3,5–4,5 sn (tek kurs havuzu, ~380 kelime/parça). Gerçekte bir kursun havuzu genelde birkaç bin parçadır (≈0,5–1 sn).
  Bu maliyet ilk sorguda ve her yeni indekslemeden sonra bir kez ödeniyor. İleride indeksleme bitince arka planda ısıtılabilir.

### 4.4 Embedding hızı (`embedding_speed.md`)
`python -m scripts.retrieval_bench embed --batch-sizes 8 16 32`. Parça başına 491 gerçek BGE-m3 token'ı (gerçek tokenizer devrede), fp16.

| Makine | GPU (VRAM) | batch | parça/sn | 10K parça (dk) | sorgu embed (ms) |
|---|---|---|---|---|---|
| Geliştirme (laptop) | RTX 3050 Laptop (4 GB) | 8 | 35.8 | 4.7 | 17.1 |
| Geliştirme (laptop) | RTX 3050 Laptop (4 GB) | 16 | 36.4 | 4.6 | 17.1 |
| Geliştirme (laptop) | RTX 3050 Laptop (4 GB) | 32 | 36.6 | 4.6 | 17.1 |
| Sunum / ana makine | RTX 5070 (8 GB) | 16 | _ölçülecek_ | | |
| Sunum / ana makine | RTX 5070 (8 GB) | 32 | _ölçülecek_ | | |

**Yorum:**
- Batch boyutunun hıza etkisi yok (≈36 parça/sn). 500 token'lık parçalarda GPU zaten 8'lik batch'te doymuş durumda. Bu yüzden `batch_size_gpu: 16` seçildi: 4 GB VRAM'de güvenli, 8 GB'ta da aynı hızda çalışır.
- 10K parçalık bir külliyatın ilk indekslenmesi 3050'de ≈4,6 dk sürüyor. Bu tek seferlik bir maliyet; yeni materyal eklendiğinde yalnızca o doküman indeksleniyor.
- Sorgu embedding'i 17 ms. Arama (≈10–15 ms) ile birlikte reranker hariç geri getirme ≈30 ms.

> Uçtan uca `retrieval_ms` = sorgu embedding + arama + (BM25) + **reranker**. Reranker gecikmesi `timings.rerank_ms` alanında ayrıca raporlanıyor. Sunumdaki rakamlar ana makineden (RTX 5070) alınmalı.

### 4.5 İç tutarlılık testi (🟡 gerçek indeksle doldurulacak)
`python -m scripts.retrieval_bench consistency [--course CENG384]`

Her (kurs, ders) için iki değer hesaplanıyor:
- **intra:** aynı dersin parçaları arasındaki ortalama kosinüs benzerliği,
- **inter:** aynı kursun diğer derslerinin parçalarıyla ortalama benzerlik.

Ders başarılı sayılırsa `intra > inter`. Rapor başarı oranını ve ortalama marjı veriyor. Mantığın doğruluğu `tests/test_retrieval.py::test_consistency_table_detects_clusters` ile sınanıyor.

| Kurs | Ders sayısı | Başarı oranı | Ortalama marj |
|---|---|---|---|
| | | | |

### 4.6 Chunk boyutu, hibrit ve timestamp_boost denemeleri (🟡 etiketli soru seti gerekli)
Bu üç karar da **gerçek sorular ve doğru cevabın yeri bilinmeden** verilemez. Bunun için:
1. `data/retrieval_eval.example.json` dosyasını `data/retrieval_eval.json` olarak kopyalayın ve 30–50 gerçek soru etiketleyin. Her soru için doğru kaynağı girin: doküman + sayfa/slayt veya video + saniye aralığı. Video izlerken sorulan sorulara `ts_range_min` ekleyin.
   Etiketler parça kimliğine değil sayfa/zaman damgasına bağlı. Bu sayede aynı set farklı chunk boyutlarında da kullanılabiliyor.
2. `python -m scripts.retrieval_bench eval`: dense / dense+rerank / hybrid / hybrid+rerank için hit@1, hit@4, MRR ve gecikme. `ts_range_min` olan sorularda `timestamp_boost` ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3} taraması.
3. `python -m scripts.retrieval_bench chunking --manifest data/sample/manifest.json`: chunk_size ∈ {300, 400, 500, 600} × overlap ∈ {50, 100} ızgarası. Her kombinasyon geçici bir koleksiyona yeniden indeksleniyor.

## 5. Öneriler ve açık noktalar
- **HNSW:** M=16 / ef_construction=100 / ef_search=100 kalsın. 100K+ ölçeğinde ef_search 150–200 denensin.
- **Hibrit:** §4.6'da hybrid+rerank, dense+rerank'ten en az birkaç puan iyiyse `hybrid: true` yapılsın. Maliyeti ihmal edilebilir.
- **Sentetik veri sınırlaması:** §4.1–4.2'deki vektörler güçlü kümelenmiş. Bu yüzden filtreli ve filtresiz recall aynı çıktı. Gerçek embedding'lerle recall biraz düşebilir. Gecikme sonuçları ise vektör içeriğinden büyük ölçüde bağımsız.
- **Benchmark yöntemi (İP4 için not):** `evaluation/benchmark.py` çeldirici parçaları başka kurslara atıyor. Sorgu önce kursa göre filtrelendiği için çeldiriciler doğruluğu neredeyse hiç etkilemez, esas olarak gecikmeyi ölçer. Doğruluğun ölçekle değişimi ölçülmek isteniyorsa çeldiricilerin bir kısmı aynı kursa atanmalı.
- **Video belirsizliği:** UI sadece ders + dakika aralığı gönderiyor. Bir derste birden çok video varsa boost hepsine uygulanıyor. İleride UI'dan doküman adı da gönderilebilir.
- **Python sürümü:** Ekip makinesinde Python 3.14 görünüyor (`__pycache__` dosyaları). torch / chromadb / faster-whisper için 3.14 wheel'leri kısıtlı olabilir. TÜBİTAK formundaki **Python 3.11–3.12** önerilir.

## 6. Yeniden üretme
```bash
pip install -r requirements.txt
python -m pytest -q                                    # 15 test (model indirmez)
python -m scripts.retrieval_bench store --m 16 32 --ef-construction 100 200 --ef-search 20 50 100 200
python -m scripts.retrieval_bench store --sizes 1000 10000 50000 --ef-search 100 --name store_scale
python -m scripts.retrieval_bench embed                # BGE-m3 indirir (~2.3 GB)
python -m scripts.retrieval_bench consistency          # önce materyal indeksleyin
python -m scripts.retrieval_bench eval                 # data/retrieval_eval.json gerekir
python -m scripts.retrieval_bench chunking --manifest data/sample/manifest.json
```

## Kaynaklar
- Can, F., Kocberber, S., Balcik, E., Kaynak, C., Ocalan, H. C., & Vursavas, O. M. (2008). Information retrieval on Turkish texts. *JASIST*, 59(3), 407–421.
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR '09*.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using HNSW graphs. *IEEE TPAMI*, 42(4).
