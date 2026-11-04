import { normalizeEmbedding } from "./visual-search.js?object-aware=1&multi-reference=1";

const MODEL_OPTIONS = {
  mobileclip: "Xenova/mobileclip_s0",
  dinov2: "Xenova/dinov2-small",
};
const requestedModel = new URL(import.meta.url).searchParams.get("model");
const MODEL_ID = MODEL_OPTIONS[requestedModel] || MODEL_OPTIONS.mobileclip;
const DETECTOR_MODEL_ID = "Xenova/owlvit-base-patch32";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";
const FOOTWEAR_LABELS = ["sandal", "shoe", "footwear"];
const MAX_OBJECT_CROPS = 2;
const DETECTION_THRESHOLD = 0.06;
export const EMBEDDING_VERSION = 3;
export const EMBEDDING_MODEL_ID = MODEL_ID;
let extractorPromise;
let detectorPromise;
let extractorDevice = "wasm";
let disableWebGpu = false;
let reportProgress = () => {};

export async function embedImage(imageUrl, onProgress = () => {}, { detailed = false } = {}) {
  reportProgress = onProgress;
  reportProgress(detailed
    ? "Loading image analysis models…"
    : "Loading the fast image analysis model…");
  await getExtractor();
  const image = new Image();
  image.src = imageUrl;
  await image.decode();

  const crops = await createImageCrops(image, detailed);
  const vectors = [];
  const objectVectors = [];
  try {
    for (const [index, cropUrl] of crops.entries()) {
      reportProgress(`Analyzing image crop ${index + 1} of ${crops.length}…`);
      vectors.push(await extractVector(cropUrl));
    }

    if (detailed) {
      reportProgress("Loading the optional footwear detector (about 127 MB on first use)…");
      const detector = await getDetector();
      const detections = await detector(imageUrl, FOOTWEAR_LABELS, {
        top_k: 8,
        threshold: DETECTION_THRESHOLD,
      });
      const objectCrops = await createObjectCrops(image, selectFootwearDetections(detections));
      try {
        for (const [index, cropUrl] of objectCrops.entries()) {
          reportProgress(`Analyzing detected footwear ${index + 1} of ${objectCrops.length}…`);
          objectVectors.push(await extractVector(cropUrl));
        }
      } finally {
        objectCrops.forEach((cropUrl) => URL.revokeObjectURL(cropUrl));
      }
    }
  } finally {
    for (const cropUrl of crops) {
      if (cropUrl !== imageUrl) URL.revokeObjectURL(cropUrl);
    }
  }
  reportProgress(detailed
    ? "Detailed image analysis complete."
    : "Fast image analysis complete. The footwear detector was skipped.");
  return { version: EMBEDDING_VERSION, vectors, objectVectors };
}

export function selectFootwearDetections(detections) {
  if (!Array.isArray(detections)) {
    throw new TypeError("The footwear detector returned an unsupported result.");
  }
  const selected = [];
  for (const detection of [...detections].sort((left, right) => right.score - left.score)) {
    const { xmin, ymin, xmax, ymax } = detection?.box ?? {};
    if (
      !Number.isFinite(detection?.score)
      || detection.score < DETECTION_THRESHOLD
      || ![xmin, ymin, xmax, ymax].every(Number.isFinite)
      || xmax <= xmin
      || ymax <= ymin
    ) continue;
    if (selected.every((candidate) => intersectionOverUnion(detection.box, candidate.box) < 0.8)) {
      selected.push(detection);
      if (selected.length === MAX_OBJECT_CROPS) break;
    }
  }
  return selected;
}

async function createImageCrops(image, detailed) {
  const imageWidth = image.naturalWidth;
  const imageHeight = image.naturalHeight;
  const crops = [image.src];
  const cropSize = Math.min(imageWidth, imageHeight);
  const positions = [];

  if (imageWidth > imageHeight) {
    const lastX = imageWidth - cropSize;
    positions.push(...(detailed
      ? [[0, 0], [lastX / 2, 0], [lastX, 0]]
      : [[lastX / 2, 0]]));
  } else if (imageHeight > imageWidth) {
    const lastY = imageHeight - cropSize;
    positions.push(...(detailed
      ? [[0, 0], [0, lastY / 2], [0, lastY]]
      : [[0, lastY / 2]]));
  } else {
    const smallerSize = cropSize * 0.8;
    const inset = cropSize - smallerSize;
    positions.push(...(detailed
      ? [[0, 0, smallerSize], [inset, 0, smallerSize], [0, inset, smallerSize], [inset, inset, smallerSize]]
      : [[inset / 2, inset / 2, smallerSize]]));
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

async function createObjectCrops(image, detections) {
  const crops = [];
  try {
    for (const { xmin, ymin, xmax, ymax } of detections.map(({ box }) => box)) {
      const x = Math.max(0, Math.floor(xmin));
      const y = Math.max(0, Math.floor(ymin));
      const width = Math.min(image.naturalWidth - x, Math.ceil(xmax) - x);
      const height = Math.min(image.naturalHeight - y, Math.ceil(ymax) - y);
      if (width <= 0 || height <= 0) continue;

      const canvas = document.createElement("canvas");
      canvas.width = 384;
      canvas.height = 384;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not prepare a detected footwear crop.");
      context.fillStyle = "#fff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / width, canvas.height / height);
      const outputWidth = width * scale;
      const outputHeight = height * scale;
      context.drawImage(
        image,
        x,
        y,
        width,
        height,
        (canvas.width - outputWidth) / 2,
        (canvas.height - outputHeight) / 2,
        outputWidth,
        outputHeight,
      );
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      if (!blob) throw new Error("Could not create a detected footwear crop.");
      crops.push(URL.createObjectURL(blob));
    }
    return crops;
  } catch (error) {
    crops.forEach((cropUrl) => URL.revokeObjectURL(cropUrl));
    throw error;
  }
}

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = createExtractor()
      .catch((error) => {
        extractorPromise = undefined;
        throw error;
      });
  }
  return extractorPromise;
}

async function createExtractor(forceCpu = false) {
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.useBrowserCache = true;
  const device = forceCpu ? "wasm" : await getPreferredDevice();
  const options = {
    dtype: "q8",
    device,
    progress_callback: (progress) => {
      if (progress.status === "progress" && Number.isFinite(progress.progress)) {
        reportProgress(`Downloading AI model: ${Math.round(progress.progress)}%`);
      } else if (progress.status === "initiate" || progress.status === "download") {
        reportProgress("Downloading AI model for the first time…");
      } else if (progress.status === "ready") {
        reportProgress("Image model ready. Preparing image search…");
      }
    },
  };
  extractorDevice = device;
  try {
    return await pipeline("image-feature-extraction", MODEL_ID, options);
  } catch (error) {
    if (device !== "webgpu") throw error;
    disableWebGpu = true;
    extractorDevice = "wasm";
    reportProgress("GPU acceleration is unavailable; switching to the CPU model.");
    return pipeline("image-feature-extraction", MODEL_ID, { ...options, device: "wasm" });
  }
}

async function getPreferredDevice() {
  if (disableWebGpu) return "wasm";
  if (!navigator.gpu) {
    reportProgress("GPU acceleration is unavailable; using the CPU image model.");
    return "wasm";
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) return "webgpu";
  } catch {
    reportProgress("GPU acceleration could not be initialized; using the CPU image model.");
    return "wasm";
  }
  reportProgress("No compatible GPU was found; using the CPU image model.");
  return "wasm";
}

async function extractVector(imageUrl) {
  let extractor = await getExtractor();
  try {
    const output = await extractor(imageUrl, { pooling: "mean", normalize: true });
    return normalizeEmbedding(poolOutput(output));
  } catch (error) {
    if (extractorDevice !== "webgpu") throw error;
    disableWebGpu = true;
    reportProgress("GPU inference failed; retrying this image on the CPU.");
    extractorPromise = createExtractor(true).catch((fallbackError) => {
      extractorPromise = undefined;
      throw fallbackError;
    });
    extractor = await extractorPromise;
    const output = await extractor(imageUrl, { pooling: "mean", normalize: true });
    return normalizeEmbedding(poolOutput(output));
  }
}

function getDetector() {
  if (!detectorPromise) {
    detectorPromise = import(TRANSFORMERS_URL)
      .then(({ pipeline, env }) => {
        env.useBrowserCache = true;
        return pipeline("zero-shot-object-detection", DETECTOR_MODEL_ID, {
          dtype: "q4f16",
          progress_callback: (progress) => {
            if (progress.status === "progress" && Number.isFinite(progress.progress)) {
              reportProgress(`Downloading footwear detector: ${Math.round(progress.progress)}%`);
            } else if (progress.status === "initiate" || progress.status === "download") {
              reportProgress("Downloading footwear detector for the first time…");
            } else if (progress.status === "ready") {
              reportProgress("Footwear detector loaded. Finding footwear…");
            }
          },
        });
      })
      .catch((error) => {
        detectorPromise = undefined;
        throw error;
      });
  }
  return detectorPromise;
}

function intersectionOverUnion(left, right) {
  const intersectionWidth = Math.max(0, Math.min(left.xmax, right.xmax) - Math.max(left.xmin, right.xmin));
  const intersectionHeight = Math.max(0, Math.min(left.ymax, right.ymax) - Math.max(left.ymin, right.ymin));
  const intersection = intersectionWidth * intersectionHeight;
  const leftArea = (left.xmax - left.xmin) * (left.ymax - left.ymin);
  const rightArea = (right.xmax - right.xmin) * (right.ymax - right.ymin);
  const union = leftArea + rightArea - intersection;
  return union > 0 ? intersection / union : 0;
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
