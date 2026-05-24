import { HTMLWebBaseLoader } from "@langchain/community/document_loaders/web/html";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HtmlToTextTransformer } from "@langchain/community/document_transformers/html_to_text";
import {client} from "./elasticsearch.js";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
const embeddings = new OpenAIEmbeddings({
    apiKey: "YOUR-API-KEY", // In Node.js defaults to process.env.OPENAI_API_KEY
    batchSize: 512, // Default value if omitted is 512. Max is 2048
    model: "text-embedding-3-large",
});
export async function parseDocsFromLink (links) {
    const loader = new HTMLWebBaseLoader(
        links
    );
    const docs = await loader.load();

    const splitter = RecursiveCharacterTextSplitter.fromLanguage("html");
    const transformer = new HtmlToTextTransformer();

    const sequence = splitter.pipe(transformer);

    return await sequence.invoke(docs);

}
async function createIndex(indexName){
   try{
       return await client.indices.create({
           index: indexName,
           settings: {
               number_of_shards: 3,
               number_of_replicas: 2,
           },
       });
   }catch(e){
       return null;
   }

}
async function embedding(text){
    const vectorstore = await MemoryVectorStore.fromDocuments(
        [{ pageContent: text, metadata: {} }],
        embeddings,
    );
}
async function insertElasticsearch () {
    //
    try {
        const indexName = 'docs'
        const [statusCreate,data] = await Promise.all([createIndex(indexName) , parseDocsFromLink('https://laptrinhcuocsong.com/lap-trinh-vien-10-tieng.html')]);
        if(!statusCreate) console.log('table created')

        const operations = data.reduce((acc, doc,currentIndex) => {
            const index = {index : {_index: indexName,_id :currentIndex}};
            acc.push(index);
            acc.push({content: doc.pageContent , meta_data: doc.metadata.loc})
            return acc;
        },[])
        const res = await client.bulk({operations: [...operations]});
        console.log(res);
    }catch (e) {
        console.log(e)
    }
}

async function main() {
    // await insertElasticsearch();
    const response = await client.search({
        index: "docs",
        query: {
            match:{content: 'tắt hết face'}
        },
    });

    console.log(response.hits.hits);
}
main();