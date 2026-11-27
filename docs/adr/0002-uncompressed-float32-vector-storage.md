---
status: accepted
---

# Uncompressed Float32Array In-Memory Vector Storage

We store L2-normalized `gemini-embedding-001` chunk embeddings (default `768` dimensions via Matryoshka truncation) as raw `Float32Array` blobs in SQLite (`chunk_embeddings`) and load them into one contiguous in-memory `Float32Array` at startup for exact cosine similarity search. At our target scale (~80K chunks across ~21.6K documents), uncompressed `Float32Array` uses ~234 MiB of RAM (keeping total peak RSS around 550 to 850 MiB, far below our 4 GB limit) and scans the full corpus in under 15 ms with zero quantization recall loss.

## Considered Options

- **`int8` scalar quantization:** Cuts vector RAM from ~234 MiB to ~59 MiB, but adds calibration code and risks recall loss on fine-grained enterprise facts without solving a real memory bottleneck at this corpus size.
- **External vector database or HNSW index (`sqlite-vec`, Qdrant, Chroma):** Adds native build risk, container overhead, or approximate search recall loss when exact brute-force dot product over 80K vectors takes under 15 ms on CPU.
- **Higher dimensions (`1536` or `3072`):** Doubles or quadruples vector RAM (~468 MiB or ~936 MiB) and scan time; kept tunable via `embeddingDims` if `val` recall stalls.

## Consequences

- `embeddingModel` and `embeddingDims` remain first-class tunable config knobs, and `chunk_embeddings` is keyed by `(dataset_doc_uuid, content_hash, chunk_index, model, dims)` so switching models or dimensions never corrupts cached vectors.
- Revisit `int8` quantization only if a corpus grows past ~250K chunks (pushing vector RAM past 1 GB) or if peak RSS approaches our 4 GB ceiling.
