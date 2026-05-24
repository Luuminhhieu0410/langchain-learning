import express from "express";
import pinoHttp from "pino-http";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "./config.js";
import { logger } from "./logger.js";
import { esClient } from "./elasticsearch.js";
import { embeddings } from "./vectorstore.js";
import { HybridElasticRetriever } from "./hybridRetriever.js";
import { buildRagChain } from "./ragChain.js";

// ---- Wire up the RAG pipeline once at startup -------------------------------

const retriever = new HybridElasticRetriever({
  client: esClient,
  embeddings,
  indexName: config.ELASTICSEARCH_INDEX,
  k: config.TOP_K,
  candidates: config.KNN_CANDIDATES,
  rankWindowSize: config.RANK_WINDOW_SIZE,
  rankConstant: config.RRF_RANK_CONSTANT,
});

const llm = new ChatOpenAI({
  apiKey: config.OPENAI_API_KEY,
  model: config.CHAT_MODEL,
  temperature: 0,
});

const ragChain = buildRagChain({ retriever, llm, defaultK: config.TOP_K });

// ---- HTTP layer ------------------------------------------------------------

const querySchema = z.object({
  question: z.string().trim().min(1, "question is required").max(2000),
  k: z.coerce.number().int().min(1).max(20).optional(),
});

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(pinoHttp({ logger }));

// Infrastructure health check (not the RAG business endpoint).
app.get("/health", async (_req, res) => {
  try {
    await esClient.ping();
    res.json({ status: "ok", elasticsearch: "up" });
  } catch {
    res.status(503).json({ status: "error", elasticsearch: "down" });
  }
});

// The one RAG endpoint: ask a question about the ingested CSV.
app.post("/query", async (req, res, next) => {
  const parsed = querySchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    return res.status(400).json({
      error: "Invalid request",
      details: parsed.error.issues.map((i) => ({
        field: i.path.join(".") || "(root)",
        message: i.message,
      })),
    });
  }

  try {
    const result = await ragChain.answer(parsed.data.question, parsed.data.k);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// Centralized error handler.
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  (req.log ?? logger).error(err, "Request failed");
  res.status(500).json({ error: "Internal server error" });
});

const server = app.listen(config.PORT, () => {
  logger.info(`RAG API listening on http://localhost:${config.PORT}`);
});

// ---- Graceful shutdown -----------------------------------------------------

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    logger.info(`${signal} received, shutting down`);
    server.close(async () => {
      await esClient.close().catch(() => {});
      process.exit(0);
    });
  });
}
