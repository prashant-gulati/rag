# local_rag.py

A local Retrieval-Augmented Generation (RAG) pipeline built around a human nutrition textbook PDF.

## What it does

### 1. Document ingestion
Downloads a human nutrition textbook PDF from an open-access source and reads it page by page using PyMuPDF, collecting per-page statistics (character count, word count, sentence count, token estimate).

### 2. Text chunking (5 strategies)
Splits the extracted text into chunks using five different approaches, each producing a list of chunk dicts with metadata (page number, character/word/token counts, chunk text):

- **Fixed-size** — splits text into chunks of approximately 500 characters by word boundary.
- **Semantic** — uses `nltk` to split into sentences, uses `all-MiniLM-L6-v2` to create sentence embeddings and cosine similarity to group semantically similar sentences together, bounded by a max token limit.
- **Recursive** — splits hierarchically: double newlines → single newlines → sentences, recursing until each piece is under a max size.
- **Structure-based (chapter)** — detects chapter boundaries in the PDF by identifying recurring header text, then groups all pages within each chapter into a single chunk.
- **LLM-based** — calls `gpt-4o-mini` via the OpenAI API to identify semantically coherent split points within a sliding text window.

### 3. Chunking analysis and visualization
Computes average size, chunk count, and size variance for each chunking method, then plots bar charts and boxplots comparing all five strategies.

### 4. Sentence-level chunking with spaCy
Re-processes pages using spaCy's sentencizer to split text into sentences, groups every 10 sentences into a chunk, and filters out chunks with fewer than 30 tokens (typically headers/footers).

### 5. Embedding
Encodes surviving sentence chunks into 768-dimensional dense vectors using `all-mpnet-base-v2` (via `sentence-transformers`). Embeddings are computed in batches on the best available device (CUDA > MPS > CPU) and saved to `text_chunks_and_embeddings_df.csv` for reuse.

### 6. Semantic search
Loads the saved embeddings into a PyTorch tensor and retrieves the top-k most relevant chunks for any query using dot-product similarity (equivalent to cosine similarity because the model outputs normalized vectors).

### 7. LLM generation
Loads a local Gemma instruction-tuned model (2B or 7B, chosen based on available GPU memory) via Hugging Face `transformers`. Given a query, retrieves relevant chunks, injects them into a structured prompt with few-shot examples, and generates a grounded answer using the local LLM.

### 8. RAG evaluation
Evaluates the end-to-end pipeline on five nutrition questions using the [RAGAS](https://github.com/explodinggradients/ragas) framework, reporting context precision, context recall, answer relevancy, and faithfulness scores.

## Dependencies
`PyMuPDF`, `sentence-transformers`, `transformers`, `torch`, `nltk`, `spacy`, `openai`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `ragas`, `datasets`, `huggingface_hub`, `python-dotenv`

## Environment variables
- `HF_TOKEN` — Hugging Face token (required to download gated Gemma models)
- `OPENAI_API_KEY` — OpenAI API key (required for LLM-based chunking and RAGAS evaluation)
