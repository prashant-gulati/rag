# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for PDF documents, using OpenAI embeddings and Supabase (pgvector) as the vector store.

**Working implementation: [rag-nutrition.lovable.app](https://rag-nutrition.lovable.app/)**

## Architecture

```
PDF → chunking → embeddings (OpenAI) → Supabase (pgvector)
                                              ↑
                                        rag-chat UI queries here
```

## Files

| File | Description |
|------|-------------|
| `ingest.py` | Ingests a PDF by extracting text, chunking by sentences, and uploading embeddings to Supabase |
| `ingest_vision.py` | Extended ingestion that also extracts images, describes them with GPT-4o vision, and embeds the descriptions |
| `supabase_code.sql` | SQL to set up the `chunks` table and `match_documents` similarity search function in Supabase |
| `test_embeddings.py` | Tests for the embedding pipeline |
| `rag-chat/` | Next.js chat UI that queries the vector store and streams RAG responses |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Supabase

Run `supabase_code.sql` in the Supabase SQL editor to enable pgvector and create the `chunks` table.

### 2. Environment variables

Create a `.env` file in this directory:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest a PDF

Place your PDF in this directory and update `PDF_PATH` in the ingest script, then run:

```bash
# Text-only ingestion
python ingest.py

# Text + image ingestion (uses GPT-4o vision)
python ingest_vision.py
```

## Chunking strategy

- Splits text into sentences using punctuation boundaries
- Groups 20 sentences per chunk with 2-sentence overlap
- Enforces a 1300-token ceiling and 50-token minimum per chunk
- Embeds chunks using `text-embedding-3-small` (1536 dimensions)

## Vector search

Supabase uses `pgvector` with an IVFFlat cosine distance index. The `match_documents` function takes a query embedding and returns the top-k most similar chunks with cosine similarity scores.
