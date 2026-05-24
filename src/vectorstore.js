import { OpenAIEmbeddings } from "@langchain/openai";
import { ElasticVectorSearch } from "@langchain/community/vectorstores/elasticsearch";
import { config } from "./config.js";
import { esClient } from "./elasticsearch.js";

/**
 * Shared embeddings model. Used both to embed documents at ingestion time and
 * to embed the user's question at query time, so the vectors live in the same
 * space.
 */
export const embeddings = new OpenAIEmbeddings({
  apiKey: config.OPENAI_API_KEY,
  model: config.EMBEDDING_MODEL,
});

/**
 * LangChain Elasticsearch vector store. We only use it on the ingestion path
 * (index creation + embedding + bulk insert via `addDocuments`). It maps each
 * document to the fields: `embedding` (dense_vector), `text` (BM25), `metadata`.
 *
 * The query path reads those same fields directly through the ES client in the
 * hybrid retriever, so we get full control over the RRF fusion (Approach A)
 * without depending on Elasticsearch's licensed native `rrf` retriever.
 */
export function getVectorStore() {
  return new ElasticVectorSearch(embeddings, {
    client: esClient,
    indexName: config.ELASTICSEARCH_INDEX,
    vectorSearchOptions: { similarity: "cosine" },
  });
}
