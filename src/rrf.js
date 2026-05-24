/**
 * Reciprocal Rank Fusion.
 *
 * Merges several independently-ranked result lists into one ranking, using
 * only each item's *position* in its list (not raw scores, which are not
 * comparable across a BM25 query and a kNN query). For every list an item
 * appears in, it earns `1 / (rankConstant + rank)`; contributions are summed.
 *
 * Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
 * and individual Rank Learning Methods" (SIGIR 2009).
 *
 * @param {Array<Array<{ id: string, doc: object }>>} rankedLists
 *   Each inner array is one retriever's results, ordered best-first. `id` is
 *   the dedup key shared across lists; `doc` is the payload returned for that id.
 * @param {object} [options]
 * @param {number} [options.k=5]            How many fused results to return.
 * @param {number} [options.rankConstant=60] RRF constant; higher flattens weighting.
 * @returns {Array<object>} Fused `doc` payloads (best-first), each with an added
 *   `rrfScore` field.
 */
export function reciprocalRankFusion(rankedLists, { k = 5, rankConstant = 60 } = {}) {
  /** @type {Map<string, { score: number, doc: object }>} */
  const fused = new Map();

  for (const list of rankedLists) {
    list.forEach((item, index) => {
      const rank = index + 1; // ranks are 1-based
      const contribution = 1 / (rankConstant + rank);
      const existing = fused.get(item.id);
      if (existing) {
        existing.score += contribution;
      } else {
        fused.set(item.id, { score: contribution, doc: item.doc });
      }
    });
  }

  return [...fused.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(({ doc, score }) => ({ ...doc, rrfScore: score }));
}
