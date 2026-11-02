import { normalizeEmbedding } from "./visual-search.js?crop-aware=2";

const MODEL_ID = "Xenova/mobileclip_s0";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
export const EMBEDDING_VERSION = 2;
let extractorPromise;
let reportProgress = () => {};

export async function embedImage(imageUrl, onProgress = () => {}) {
  reportProgress = onProgress;
  reportProgress("Loading the AI model…");
  const extractor = await getExtractor();
  const image = new Image();
  image.src = imageUrl;
  await image.decode();

  const crops = await createImageCrops(image);
  const vectors = [];
  try {
    for (const [index, cropUrl] of crops.entries()) {
      reportProgress(`Analyzing image crop ${index + 1} of ${crops.length}…`);
      const output = await extractor(cropUrl, { pooling: "mean", normalize: true });
      vectors.push(normalizeEmbedding(poolOutput(output)));
    }
  } finally {
    for (const cropUrl of crops) {
      if (cropUrl !== imageUrl) URL.revokeObjectURL(cropUrl);
    }
  }
  reportProgress("AI model ready. Images are analyzed in this browser.");
  return { version: EMBEDDING_VERSION, vectors };
}

async function createImageCrops(image) {
  const imageWidth = image.naturalWidth;
  const imageHeight = image.naturalHeight;
  const crops = [image.src];
  const cropSize = Math.min(imageWidth, imageHeight);
  const positions = [];

  if (imageWidth > imageHeight) {
    const lastX = imageWidth - cropSize;
    positions.push([0, 0], [lastX / 2, 0], [lastX, 0]);
  } else if (imageHeight > imageWidth) {
    const lastY = imageHeight - cropSize;
    positions.push([0, 0], [0, lastY / 2], [0, lastY]);
  } else {
    const smallerSize = cropSize * 0.8;
    const inset = cropSize - smallerSize;
    positions.push([0, 0, smallerSize], [inset, 0, smallerSize], [0, inset, smallerSize], [inset, inset, smallerSize]);
  }

  try {
    for (const [x, y, optionalSize] of positions) {
      const size = optionalSize ?? cropSize;
      const canvas = document.createElement("canvas");
      const outputSize = Math.min(512, size);
      canvas.width = outputSize;
      canvas.height = outputSize;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not prepare an image crop for visual search.");
      context.drawImage(image, x, y, size, size, 0, 0, outputSize, outputSize);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
      if (!blob) throw new Error("Could not create an image crop for visual search.");
      crops.push(URL.createObjectURL(blob));
    }
  } catch (error) {
    for (const cropUrl of crops.slice(1)) URL.revokeObjectURL(cropUrl);
    throw error;
  }
  return crops;
}

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = import(TRANSFORMERS_URL)
      .then(({ pipeline, env }) => {
        env.useBrowserCache = true;
        return pipeline("image-feature-extraction", MODEL_ID, {
          dtype: "q8",
          progress_callback: (progress) => {
            if (progress.status === "progress" && Number.isFinite(progress.progress)) {
              reportProgress(`Downloading AI model: ${Math.round(progress.progress)}%`);
            } else if (progress.status === "initiate" || progress.status === "download") {
              reportProgress("Downloading AI model for the first time…");
            } else if (progress.status === "ready") {
              reportProgress("AI model loaded. Preparing image search…");
            }
          },
        });
      })
      .catch((error) => {
        extractorPromise = undefined;
        throw error;
      });
  }
  return extractorPromise;
}

function poolOutput(output) {
  const tensor = Array.isArray(output) && output.length === 1 ? output[0] : output;
  if (!tensor?.data || !tensor?.dims || tensor.dims.length === 0) {
    throw new TypeError("The vision model returned an unsupported embedding.");
  }

  const dimensions = tensor.dims;
  const values = tensor.data;
  if (dimensions.length <= 2) return values;

  const embeddingSize = dimensions[dimensions.length - 1];
  const tokenCount = values.length / embeddingSize;
  if (!Number.isInteger(tokenCount) || embeddingSize < 1) {
    throw new TypeError("The vision model returned an invalid embedding shape.");
  }

  const pooled = new Float32Array(embeddingSize);
  for (let token = 0; token < tokenCount; token += 1) {
    for (let index = 0; index < embeddingSize; index += 1) {
      pooled[index] += values[token * embeddingSize + index] / tokenCount;
    }
  }
  return pooled;
}
