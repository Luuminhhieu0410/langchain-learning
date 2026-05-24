# CSV RAG API

Production-style Retrieval-Augmented Generation over **any CSV file**, built with
**LangChain + Elasticsearch + Express**. Ingest a CSV once, then ask questions
about it through a single HTTP endpoint.

## How it works

```
CSV ──(npm run ingest)──► CSVLoader → split → OpenAI embeddings → Elasticsearch
                                                  (dense_vector + text fields)

POST /query ─► HybridElasticRetriever ─► RAG chain ─► answer + sources
                 │                          │
                 │ kNN (semantic)           prompt + gpt-4o-mini
                 │ BM25 (keyword)           (answers only from context)
                 └─ RRF fusion (in app)
```

**Hybrid search (Approach A):** runs a semantic kNN query and a BM25 keyword
query, then fuses the two rankings with **Reciprocal Rank Fusion in application
code** (`src/rrf.js`). This avoids Elasticsearch's licensed native `rrf`
retriever, so it runs on any cluster (including a basic local one).

## Project layout

| File | Responsibility |
|------|----------------|
| `src/config.js` | Loads + validates env (`zod`), fails fast |
| `src/elasticsearch.js` | Shared ES client |
| `src/vectorstore.js` | OpenAI embeddings + LangChain `ElasticVectorSearch` |
| `src/ingest.js` | CLI: CSV → chunks → embed → index |
| `src/rrf.js` | Pure Reciprocal Rank Fusion (unit-tested) |
| `src/hybridRetriever.js` | kNN + BM25 + RRF, as a LangChain retriever |
| `src/ragChain.js` | Retrieve → prompt → `gpt-4o-mini` → answer + sources |
| `src/server.js` | Express app: `POST /query`, `GET /health` |

## Setup

1. **Elasticsearch 9.x** running (e.g. `http://localhost:9200`).
2. Install deps: `npm install`
3. Copy env and fill in your OpenAI key:
   ```
   cp .env.example .env
   # set OPENAI_API_KEY=...
   ```

## Usage

**Ingest a CSV** (run once per dataset; `--recreate` wipes the index first):

```bash
npm run ingest -- ./data/sample.csv --recreate
```

**Start the API:**

```bash
npm start
```

**Ask a question:**

```bash
curl -X POST http://localhost:3000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which book teaches how to build good habits?"}'
```

Response:

```json
{
  "answer": "The book that teaches how to build good habits is \"Atomic Habits\" ... [1].",
  "sources": [
    { "rank": 1, "content": "id: 5\ntitle: Atomic Habits\n...", "metadata": { "source": "sample.csv", "line": 5, "rrfScore": 0.0328 }, "score": 0.0328 }
  ]
}
```

`question` is required; optional `k` (1–20) overrides how many rows are retrieved.

## Tests

```bash
npm test     # unit tests for the RRF fusion logic
```

## Configuration

All tunables live in `.env` (see `.env.example`): models, `TOP_K`,
`RANK_WINDOW_SIZE`, `RRF_RANK_CONSTANT`, `KNN_CANDIDATES`, chunking sizes, etc.
