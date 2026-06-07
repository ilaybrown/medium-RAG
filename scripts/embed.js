const fs = require('fs');
const csv = require('csv-parser');
const OpenAI = require('openai').default;
const { Pinecone } = require('@pinecone-database/pinecone');

const COURSE_API_KEY = process.env.COURSE_API_KEY;
const COURSE_BASE_URL = process.env.COURSE_BASE_URL;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'medium-articles';
const PINECONE_INDEX_HOST = process.env.PINECONE_INDEX_HOST;
const PINECONE_NAMESPACE = process.env.PINECONE_NAMESPACE || 'final';

if (!COURSE_API_KEY) throw new Error('Missing COURSE_API_KEY');
if (!COURSE_BASE_URL) throw new Error('Missing COURSE_BASE_URL');
if (!PINECONE_API_KEY) throw new Error('Missing PINECONE_API_KEY');

const openai = new OpenAI({
  apiKey: COURSE_API_KEY,
  baseURL: COURSE_BASE_URL,
});

const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = PINECONE_INDEX_HOST
  ? pc.index(PINECONE_INDEX_NAME, PINECONE_INDEX_HOST)
  : pc.index(PINECONE_INDEX_NAME);

const CHUNK_SIZE_APPROX_TOKENS = 512;
const OVERLAP_RATIO = 0.2;
const CHUNK_CHARS = CHUNK_SIZE_APPROX_TOKENS * 4;
const OVERLAP_CHARS = Math.floor(CHUNK_CHARS * OVERLAP_RATIO);
const STEP_CHARS = CHUNK_CHARS - OVERLAP_CHARS;

const CSV_PATH = './data/medium-english-50mb.csv';
const EMBEDDING_MODEL = '4UHRUIN-text-embedding-3-small';

const START_ARTICLE_INDEX = 1601;
const ARTICLE_LIMIT = null;

function cleanText(value) {
  return String(value || '')
    .replace(/[\uD800-\uDFFF]/g, '')
    .replace(/\u0000/g, '')
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function chunkText(text) {
  const clean = cleanText(text);
  if (!clean) return [];
  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    const end = Math.min(start + CHUNK_CHARS, clean.length);
    const chunk = clean.slice(start, end).trim();
    if (chunk.length > 0) chunks.push(chunk);
    if (end === clean.length) break;
    start += STEP_CHARS;
  }
  return chunks;
}

async function embedBatch(texts) {
  if (texts.length === 0) return [];
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  return res.data.map((item) => item.embedding);
}

async function loadArticles() {
  const articles = [];
  await new Promise((resolve, reject) => {
    fs.createReadStream(CSV_PATH)
      .pipe(csv())
      .on('data', (row) => articles.push(row))
      .on('end', resolve)
      .on('error', reject);
  });
  return articles;
}

async function main() {
  const articles = await loadArticles();
  console.log('Loaded ' + articles.length + ' articles');

  const endIndex = ARTICLE_LIMIT === null
    ? articles.length
    : Math.min(articles.length, START_ARTICLE_INDEX + ARTICLE_LIMIT);

  console.log('Resuming from article ' + START_ARTICLE_INDEX + ' to ' + (endIndex - 1));
  console.log('Using Pinecone index: ' + PINECONE_INDEX_NAME);
  console.log('Using namespace: ' + PINECONE_NAMESPACE);

  let totalVectors = 0;

  for (let i = START_ARTICLE_INDEX; i < endIndex; i++) {
    const article = articles[i];
    const title = cleanText(article.title);
    const text = cleanText(article.text);
    const authors = cleanText(article.authors);
    const url = cleanText(article.url);
    const timestamp = cleanText(article.timestamp);
    const tags = cleanText(article.tags);

    if (!text) {
      console.log('Skipping article ' + i + ': empty text');
      continue;
    }

    const chunks = chunkText(text);
    if (chunks.length === 0) {
      console.log('Skipping article ' + i + ': no chunks');
      continue;
    }

    const records = [];

    for (let j = 0; j < chunks.length; j += 20) {
      const chunkBatch = chunks.slice(j, j + 20);
      const embeddingInputs = chunkBatch.map((chunk) =>
        'Title: ' + title + '\nAuthors: ' + authors + '\nTags: ' + tags + '\nText: ' + chunk
      );
      const embeddings = await embedBatch(embeddingInputs);
      for (let k = 0; k < chunkBatch.length; k++) {
        records.push({
          id: 'art' + i + '_chunk' + (j + k),
          values: embeddings[k],
          metadata: {
            article_id: String(i),
            title, authors, url, timestamp, tags,
            chunk: chunkBatch[k],
            chunk_index: j + k,
          },
        });
      }
    }

    for (let j = 0; j < records.length; j += 100) {
      const recordBatch = records.slice(j, j + 100);
      if (recordBatch.length === 0) continue;
      await index.upsert({
        namespace: PINECONE_NAMESPACE,
        records: recordBatch,
      });
      totalVectors += recordBatch.length;
    }

    console.log('Article ' + (i + 1) + '/' + articles.length + ' done — chunks: ' + chunks.length + ', total vectors: ' + totalVectors);
  }

  console.log('Done! Total vectors: ' + totalVectors);
}

main().catch((error) => {
  console.error('Embedding script failed:', error);
  process.exit(1);
});