Query embedding latency: 17.1 ms. Machine: Windows 10, Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Python 3.12.15, chromadb 1.5.9

| model | device | fp16 | gpu | torch | tokenizer | tokens/chunk | batch | chunks/s | 10K chunks (min) |
|---|---|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | cuda | True | NVIDIA GeForce RTX 3050 Laptop GPU | 2.14.1+cu126 | BAAI/bge-m3 | 491 | 8 | 35.8 | 4.7 |
| BAAI/bge-m3 | cuda | True | NVIDIA GeForce RTX 3050 Laptop GPU | 2.14.1+cu126 | BAAI/bge-m3 | 491 | 16 | 36.4 | 4.6 |
| BAAI/bge-m3 | cuda | True | NVIDIA GeForce RTX 3050 Laptop GPU | 2.14.1+cu126 | BAAI/bge-m3 | 491 | 32 | 36.6 | 4.6 |
