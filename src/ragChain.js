import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const SYSTEM_PROMPT = [
  "You are a precise assistant that answers questions about a dataset loaded from a CSV file.",
  "Answer using ONLY the information in the provided context.",
  "If the answer is not contained in the context, say you don't have enough information in the dataset — do not guess.",
  "When useful, cite the supporting rows by their bracketed number, e.g. [2].",
  "",
].join(" ");

const prompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_PROMPT],
  ["human", "Context:\n{context}\n\nQuestion: {question}"],
]);

/** Render retrieved documents into a numbered context block for the prompt. */
function formatDocs(docs) {
  return docs
    .map((doc, i) => `[${i + 1}] ${doc.pageContent}`)
    .join("\n\n---\n\n");
}

/**
 * Builds the RAG pipeline: retrieve (hybrid) -> stuff context into the prompt
 * -> generate with the chat model. Returns an object exposing `answer()`.
 *
 * @param {object} deps
 * @param {import('./hybridRetriever.js').HybridElasticRetriever} deps.retriever
 * @param {import('@langchain/openai').ChatOpenAI} deps.llm
 * @param {number} deps.defaultK
 */
export function buildRagChain({ retriever, llm, defaultK = 5 }) {
  const generation = prompt.pipe(llm).pipe(new StringOutputParser());

  return {
    /**
     * @param {string} question
     * @param {number} [k] override the number of retrieved documents
     */
    async answer(question, k) {
      const docs = await retriever.search(question, k ?? defaultK);

      if (docs.length === 0) {
        return {
          answer:
            "I couldn't find anything relevant in the dataset to answer that.",
          sources: [],
        };
      }

      const answer = await generation.invoke({
        context: formatDocs(docs),
        question,
      });

      return {
        answer,
        sources: docs.map((doc, i) => ({
          rank: i + 1,
          content: doc.pageContent,
          metadata: doc.metadata,
          score: doc.metadata?.rrfScore,
        })),
      };
    },
  };
}
