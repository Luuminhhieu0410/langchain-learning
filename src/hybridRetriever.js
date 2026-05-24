import { BaseRetriever } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import { reciprocalRankFusion } from "./rrf.js";

/**
 * Hybrid retriever (Approach A): runs a semantic kNN search and a BM25
 * keyword search against the same Elasticsearch index, then fuses the two
 * rankings with RRF in application code.
 *
 * This deliberately avoids Elasticsearch's native `rrf` retriever (a licensed
 * feature) so it runs on any cluster, including a basic local one.
 *
 * It reads the fields written by LangChain's ElasticVectorSearch:
 *   - `embedding` : dense_vector (kNN)
 *   - `text`      : full text     (BM25)
 *   - `metadata`  : original document metadata
 */
export class HybridElasticRetriever extends BaseRetriever {
  lc_namespace = ["custom", "retrievers", "hybrid_elastic"];

  constructor(fields) {
    super(fields);
    this.client = fields.client;
    this.embeddings = fields.embeddings;
    this.indexName = fields.indexName;
    this.k = fields.k ?? 5;
    this.candidates = fields.candidates ?? 100;
    this.rankWindowSize = fields.rankWindowSize ?? 50;
    this.rankConstant = fields.rankConstant ?? 60;
  }

  /**
   * @param {string} query
   * @param {number} [k] number of fused documents to return (defaults to this.k)
   * @returns {Promise<Document[]>}
   */
  async search(query, k = this.k) {
    // Pull a wide window from each retriever so fusion has material to work
    // with, then trim to k after merging.
    const window = Math.max(this.rankWindowSize, k);
    const queryVector = await this.embeddings.embedQuery(query);

    const [knnRes, bm25Res] = await Promise.all([
      this.client.search({
        index: this.indexName,
        size: window,
        _source: ["text", "metadata"],
        knn: {
          field: "embedding",
          query_vector: queryVector,
          k: window,
          num_candidates: Math.max(this.candidates, window),
        },
      }),
      this.client.search({
        index: this.indexName,
        size: window,
        _source: ["text", "metadata"],
        query: { match: { text: query } },
      }),
    ]);

    const toItems = (res) =>
      res.hits.hits.map((hit) => ({
        id: hit._id,
        doc: {
          pageContent: hit._source?.text ?? "",
          metadata: hit._source?.metadata ?? {},
        },
      }));

    const fused = reciprocalRankFusion([toItems(knnRes), toItems(bm25Res)], {
      k,
      rankConstant: this.rankConstant,
    });

    return fused.map(
      (f) =>
        new Document({
          pageContent: f.pageContent,
          metadata: { ...f.metadata, rrfScore: f.rrfScore },
        }),
    );
  }

  // BaseRetriever entry point so this also works anywhere a LangChain
  // retriever is expected (e.g. `.invoke(query)`).
  async _getRelevantDocuments(query) {
    return this.search(query, this.k);
  }
}
