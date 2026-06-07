import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

const COURSE_API_KEY = process.env.COURSE_API_KEY;
const COURSE_BASE_URL = process.env.COURSE_BASE_URL;

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'medium-articles';
const PINECONE_INDEX_HOST = process.env.PINECONE_INDEX_HOST;
const PINECONE_NAMESPACE = process.env.PINECONE_NAMESPACE || 'final';

const EMBEDDING_MODEL = '4UHRUIN-text-embedding-3-small';
const CHAT_MODEL = '4UHRUIN-gpt-5-mini';

const TOP_K = 15;

const SYSTEM_PROMPT = `
You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Additional rules:
- Use only the retrieved context.
- Do not invent article titles, authors, URLs, dates, or article content.
- If the user asks for exactly 3 article titles, return exactly 3 distinct article titles and nothing else.
- Treat chunks with the same article_id as belonging to the same article.
- If the retrieved context is not relevant enough, answer exactly:
I don’t know based on the provided Medium articles data.
`.trim();

function validateEnv() {
  if (!COURSE_API_KEY) {
    throw new Error('Missing COURSE_API_KEY');
  }

  if (!COURSE_BASE_URL) {
    throw new Error('Missing COURSE_BASE_URL');
  }

  if (!PINECONE_API_KEY) {
    throw new Error('Missing PINECONE_API_KEY');
  }
}

function createOpenAIClient() {
  return new OpenAI({
    apiKey: COURSE_API_KEY,
    baseURL: COURSE_BASE_URL,
  });
}

function createPineconeIndex() {
  const pc = new Pinecone({
    apiKey: PINECONE_API_KEY,
  });

  if (PINECONE_INDEX_HOST) {
    return pc.index(PINECONE_INDEX_NAME, PINECONE_INDEX_HOST);
  }

  return pc.index(PINECONE_INDEX_NAME);
}

async function embedQuestion(openai, question) {
  const result = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: question,
  });

  return result.data[0].embedding;
}

function normalizeContext(matches) {
  return matches.map((match) => {
    const metadata = match.metadata || {};

    return {
      article_id: String(metadata.article_id || ''),
      title: String(metadata.title || ''),
      authors: String(metadata.authors || ''),
      url: String(metadata.url || ''),
      timestamp: String(metadata.timestamp || ''),
      tags: String(metadata.tags || ''),
      chunk: String(metadata.chunk || ''),
      score: Number(match.score || 0),
    };
  });
}

function dedupeByArticleId(context) {
  const seen = new Set();
  const result = [];

  for (const item of context) {
    if (!item.article_id) {
      continue;
    }

    if (seen.has(item.article_id)) {
      continue;
    }

    seen.add(item.article_id);
    result.push(item);
  }

  return result;
}

function buildUserPrompt(question, context) {
  const contextText = context
    .map((item, index) => {
      return `
[Context ${index + 1}]
article_id: ${item.article_id}
title: ${item.title}
authors: ${item.authors}
url: ${item.url}
timestamp: ${item.timestamp}
tags: ${item.tags}
score: ${item.score}

chunk:
${item.chunk}
`.trim();
    })
    .join('\n\n---\n\n');

  return `
Retrieved Medium article context:

${contextText}

User question:
${question}

Answer the user question using only the retrieved Medium article context.
`.trim();
}

async function askChatModel(openai, systemPrompt, userPrompt) {
  const result = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: 'system',
        content: systemPrompt,
      },
      {
        role: 'user',
        content: userPrompt,
      },
    ],
  });

  return result.choices[0]?.message?.content || '';
}

export async function POST(request) {
  try {
    validateEnv();

    const body = await request.json();
    const question = body.question;

    if (!question || typeof question !== 'string') {
      return Response.json(
        {
          error: 'Missing or invalid question',
        },
        {
          status: 400,
        }
      );
    }

    const openai = createOpenAIClient();
    const index = createPineconeIndex();

    const questionEmbedding = await embedQuestion(openai, question);

    const searchResult = await index.query({
      namespace: PINECONE_NAMESPACE,
      vector: questionEmbedding,
      topK: TOP_K,
      includeMetadata: true,
    });

    const rawContext = normalizeContext(searchResult.matches || []);
    const context = dedupeByArticleId(rawContext);

    const userPrompt = buildUserPrompt(question, context);

    const response = await askChatModel(openai, SYSTEM_PROMPT, userPrompt);

    return Response.json({
      response,
      context: context.map((item) => ({
        article_id: item.article_id,
        title: item.title,
        chunk: item.chunk,
        score: item.score,
      })),
      Augmented_prompt: {
        System: SYSTEM_PROMPT,
        User: userPrompt,
      },
    });
  } catch (error) {
    console.error('POST /api/prompt failed:', error);

    return Response.json(
      {
        error: 'Internal server error',
        details: String(error.message || error),
      },
      {
        status: 500,
      }
    );
  }
}