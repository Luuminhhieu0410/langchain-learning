import "dotenv/config";
import { z } from "zod";

const schema = z.object({
  OPENAI_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),

  ELASTICSEARCH_URL: z.string().url().default("http://localhost:9200"),
  ELASTICSEARCH_INDEX: z.string().min(1).default("rag_csv"),

  EMBEDDING_MODEL: z.string().min(1).default("text-embedding-3-large"),
  CHAT_MODEL: z.string().min(1).default("gpt-4o-mini"),

  PORT: z.coerce.number().int().positive().default(3000),

  TOP_K: z.coerce.number().int().positive().default(5),
  RANK_WINDOW_SIZE: z.coerce.number().int().positive().default(50),
  RRF_RANK_CONSTANT: z.coerce.number().int().positive().default(60),
  KNN_CANDIDATES: z.coerce.number().int().positive().default(100),

  CHUNK_SIZE: z.coerce.number().int().positive().default(1000),
  CHUNK_OVERLAP: z.coerce.number().int().nonnegative().default(200),

  LOG_LEVEL: z.string().default("info"),
});

const parsed = schema.safeParse(process.env);

if (!parsed.success) {
  // Fail fast with a readable message instead of crashing deep inside a request.
  const lines = parsed.error.issues.map(
    (issue) => `  - ${issue.path.join(".") || "(root)"}: ${issue.message}`,
  );
  console.error("Invalid environment configuration:\n" + lines.join("\n"));
  console.error("\nCopy .env.example to .env and fill in the values.");
  process.exit(1);
}

export const config = parsed.data;
