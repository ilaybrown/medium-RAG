export async function GET() {
  return Response.json({
    chunk_size: 512,
    overlap_ratio: 0.2,
    top_k: 30,
  });
}