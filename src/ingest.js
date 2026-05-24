import fs from "node:fs";
import path from "node:path";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { config } from "./config.js";
import { esClient } from "./elasticsearch.js";
import { getVectorStore } from "./vectorstore.js";
import { logger } from "./logger.js";

const BATCH_SIZE = 200;

function parseArgs(argv) {
  const args = argv.slice(2);
  const recreate = args.includes("--recreate");
  const filePath = args.find((a) => !a.startsWith("--"));
  return { filePath, recreate };
}

async function main() {
  const { filePath, recreate } = parseArgs(process.argv);

  if (!filePath) {
    logger.error("Usage: npm run ingest -- <path-to-csv> [--recreate]");
    process.exit(1);
  }

  const absPath = path.resolve(filePath);
  if (!fs.existsSync(absPath)) {
    logger.error(`CSV file not found: ${absPath}`);
    process.exit(1);
  }

  const store = getVectorStore();

  if (recreate) {
    logger.info({ index: config.ELASTICSEARCH_INDEX }, "Dropping existing index (--recreate)");
    await store.deleteIfExists();
  }

  // CSVLoader with no `column` option turns each row into one document whose
  // pageContent is "header: value" for every column — so a query can match on
  // any field. One CSV row = one Document.
  logger.info({ file: absPath }, "Loading CSV");
  const rows = await new CSVLoader(absPath).load();
  logger.info({ rows: rows.length }, "Loaded rows");

  if (rows.length === 0) {
    logger.warn("CSV produced no rows; nothing to ingest.");
    await esClient.close();
    return;
  }

  // Long free-text columns can exceed embedding limits; split defensively.
  // Short rows pass through as a single chunk.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.CHUNK_SIZE,
    chunkOverlap: config.CHUNK_OVERLAP,
  });
  const chunks = await splitter.splitDocuments(rows);

  const source = path.basename(absPath);
  for (const chunk of chunks) {
    chunk.metadata = { ...chunk.metadata, source };
  }
  logger.info({ chunks: chunks.length }, "Split into chunks; embedding + indexing");

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    // addDocuments creates the index (dense_vector + text mapping) on first
    // call, embeds the batch, and bulk-inserts it.
    await store.addDocuments(batch);
    logger.info(
      { indexed: Math.min(i + BATCH_SIZE, chunks.length), total: chunks.length },
      "Indexed batch",
    );
  }

  await esClient.indices.refresh({ index: config.ELASTICSEARCH_INDEX });
  logger.info({ index: config.ELASTICSEARCH_INDEX }, "Ingestion complete");
  await esClient.close();
}

main().catch(async (err) => {
  logger.error(err, "Ingestion failed");
  await esClient.close().catch(() => {});
  process.exit(1);
});
