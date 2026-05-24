import { Client } from "@elastic/elasticsearch";
import { config } from "./config.js";

/**
 * Single shared Elasticsearch client. Both the ingestion CLI and the API
 * server reuse this so connections are pooled rather than re-created.
 */
export const esClient = new Client({ node: config.ELASTICSEARCH_URL });
