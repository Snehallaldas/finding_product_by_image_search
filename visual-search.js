export function normalizeEmbedding(values) {
  if (!values || values.length === 0) {
    throw new TypeError("Expected a non-empty image embedding.");
  }
  const embedding = Float32Array.from(values);
  let magnitude = 0;
  for (const value of embedding) {
    if (!Number.isFinite(value)) throw new TypeError("Image embedding values must be finite.");
    magnitude += value * value;
  }
  magnitude = Math.sqrt(magnitude);
  if (magnitude === 0) throw new TypeError("Image embedding must not be empty or all zero.");
  for (let index = 0; index < embedding.length; index += 1) {
    embedding[index] /= magnitude;
  }
  return embedding;
}

export function similarity(left, right) {
  const leftVectors = getEmbeddingVectors(left);
  const rightVectors = getEmbeddingVectors(right);
  let globalScore = cosineSimilarity(leftVectors[0], rightVectors[0]);
  let bestCropScore = Number.NEGATIVE_INFINITY;

  for (let leftIndex = 0; leftIndex < leftVectors.length; leftIndex += 1) {
    for (let rightIndex = 0; rightIndex < rightVectors.length; rightIndex += 1) {
      if (leftIndex === 0 && rightIndex === 0) continue;
      bestCropScore = Math.max(bestCropScore, cosineSimilarity(leftVectors[leftIndex], rightVectors[rightIndex]));
    }
  }
  if (bestCropScore === Number.NEGATIVE_INFINITY) return globalScore;
  return Math.max(-1, Math.min(1, 0.35 * globalScore + 0.65 * bestCropScore));
}

function getEmbeddingVectors(embedding) {
  const vectors = embedding?.version === 2 ? embedding.vectors : ArrayBuffer.isView(embedding) ? [embedding] : null;
  if (!Array.isArray(vectors) || vectors.length === 0) {
    throw new TypeError("Expected a non-empty image embedding.");
  }
  for (const vector of vectors) {
    if (!ArrayBuffer.isView(vector) || vector.length === 0) {
      throw new TypeError("Image embedding crops must be non-empty numeric vectors.");
    }
  }
  return vectors;
}

function cosineSimilarity(left, right) {
  if (left.length !== right.length) {
    throw new TypeError("Expected image embedding vectors with matching dimensions.");
  }
  let score = 0;
  for (let index = 0; index < left.length; index += 1) {
    score += left[index] * right[index];
  }
  return Math.max(-1, Math.min(1, score));
}

export function rankProducts(queryEmbedding, products, category = "all") {
  if (!Array.isArray(products)) throw new TypeError("Products must be an array.");
  return products
    .filter((product) => category === "all" || product.category === category)
    .map((product) => ({ ...product, score: similarity(queryEmbedding, product.embedding) }))
    .sort((left, right) => right.score - left.score);
}
