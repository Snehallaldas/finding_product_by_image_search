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
  const leftObjects = getObjectVectors(left);
  const rightObjects = getObjectVectors(right);
  if (leftObjects.length && rightObjects.length) {
    return bestPairwiseSimilarity(leftObjects, rightObjects);
  }

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

function bestPairwiseSimilarity(leftVectors, rightVectors) {
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const left of leftVectors) {
    for (const right of rightVectors) {
      bestScore = Math.max(bestScore, cosineSimilarity(left, right));
    }
  }
  return bestScore;
}

function getObjectVectors(embedding) {
  if (embedding?.version !== 3) return [];
  if (!Array.isArray(embedding.objectVectors)) {
    throw new TypeError("Expected detected footwear embeddings for version 3.");
  }
  for (const vector of embedding.objectVectors) {
    if (!ArrayBuffer.isView(vector) || vector.length === 0) {
      throw new TypeError("Detected footwear embeddings must be non-empty numeric vectors.");
    }
  }
  return embedding.objectVectors;
}

function getEmbeddingVectors(embedding) {
  const vectors = embedding?.version === 2 || embedding?.version === 3
    ? embedding.vectors
    : ArrayBuffer.isView(embedding) ? [embedding] : null;
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

export function retrievalMetrics(rankedIds, relevantIds, k = 10) {
  if (!Array.isArray(rankedIds) || !Array.isArray(relevantIds)) {
    throw new TypeError("Ranked and relevant product IDs must be arrays.");
  }
  if (!Number.isInteger(k) || k < 1) {
    throw new RangeError("The retrieval cutoff must be a positive integer.");
  }
  const relevant = new Set(relevantIds);
  if (relevant.size === 0) {
    throw new TypeError("At least one relevant product ID is required.");
  }
  const topResults = rankedIds.slice(0, k);
  let hits = 0;
  let precisionSum = 0;
  const seen = new Set();
  for (const [index, productId] of topResults.entries()) {
    if (seen.has(productId)) continue;
    seen.add(productId);
    if (relevant.has(productId)) {
      hits += 1;
      precisionSum += hits / (index + 1);
    }
  }
  return {
    recallAtK: hits / relevant.size,
    averagePrecisionAtK: precisionSum / Math.min(relevant.size, k),
  };
}
