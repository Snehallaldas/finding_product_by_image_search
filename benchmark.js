import { embedImage } from "./image-embeddings.js?object-aware=1";
import { retrievalMetrics, similarity } from "./visual-search.js?object-aware=1";

const MAX_CATALOG_IMAGES = 1000;
const MAX_QUERIES = 500;
const TOP_K = 10;

const elements = {
  folder: document.querySelector("#benchmark-folder"),
  dataset: document.querySelector("#benchmark-dataset"),
  mode: document.querySelector("#benchmark-mode"),
  run: document.querySelector("#benchmark-run"),
  status: document.querySelector("#benchmark-status"),
  results: document.querySelector("#benchmark-results"),
  summary: document.querySelector("#benchmark-summary"),
  tableBody: document.querySelector("#benchmark-table-body"),
  queryResults: document.querySelector("#benchmark-query-results"),
  download: document.querySelector("#benchmark-download"),
};

let dataset = null;
let reportUrl = null;
let running = false;

elements.folder.addEventListener("change", loadDataset);
elements.run.addEventListener("click", runBenchmark);
elements.download.addEventListener("click", downloadReport);

async function loadDataset() {
  dataset = null;
  elements.run.disabled = true;
  elements.results.hidden = true;
  elements.status.textContent = "";
  const files = [...elements.folder.files];
  if (!files.length) {
    elements.dataset.textContent = "No benchmark folder selected.";
    return;
  }
  try {
    dataset = await parseDataset(files);
    elements.dataset.textContent =
      `Loaded ${dataset.catalog.length} catalog images and ${dataset.queries.length} labeled queries.`;
    elements.run.disabled = false;
  } catch (error) {
    elements.dataset.textContent = `Could not load benchmark: ${error.message}`;
    console.error(error);
  }
}

async function parseDataset(files) {
  const filesByPath = new Map();
  const rootFolders = new Set();
  for (const file of files) {
    const path = normalizePath(file.webkitRelativePath || file.name);
    const segments = path.split("/");
    rootFolders.add(segments.length > 1 ? segments[0] : "");
    const relativePath = segments.length > 1 ? segments.slice(1).join("/") : path;
    const key = relativePath.toLowerCase();
    if (filesByPath.has(key)) throw new Error(`Duplicate file path: ${relativePath}`);
    filesByPath.set(key, file);
  }
  if (rootFolders.size > 1) throw new Error("Select one benchmark folder at a time.");
  const manifestFile = filesByPath.get("benchmark.json");
  if (!manifestFile) throw new Error("The selected folder must contain benchmark.json.");

  let manifest;
  try {
    manifest = JSON.parse(await manifestFile.text());
  } catch (error) {
    throw new Error(`benchmark.json is invalid: ${error.message}`);
  }
  validateManifest(manifest);

  const productIds = new Set();
  const catalogPaths = new Set();
  const catalog = manifest.catalog.map((entry) => {
    if (productIds.has(entry.id)) throw new Error(`Duplicate catalog product ID: ${entry.id}`);
    productIds.add(entry.id);
    const path = normalizePath(entry.image);
    if (catalogPaths.has(path.toLowerCase())) throw new Error(`Duplicate catalog image: ${path}`);
    catalogPaths.add(path.toLowerCase());
    return { ...entry, path, file: findImageFile(filesByPath, path) };
  });
  const queryIds = new Set();
  const queryPaths = new Set();
  const queries = manifest.queries.map((entry) => {
    if (queryIds.has(entry.id)) throw new Error(`Duplicate query ID: ${entry.id}`);
    queryIds.add(entry.id);
    const path = normalizePath(entry.image);
    if (queryPaths.has(path.toLowerCase()) || catalogPaths.has(path.toLowerCase())) {
      throw new Error(`Query images must be unique and separate from catalog images: ${path}`);
    }
    queryPaths.add(path.toLowerCase());
    const missingIds = entry.relevant_product_ids.filter((id) => !productIds.has(id));
    if (missingIds.length) {
      throw new Error(`Query ${entry.id} references unknown product ID(s): ${missingIds.join(", ")}`);
    }
    if (entry.category && entry.category !== "all") {
      const wrongCategory = catalog.find((product) =>
        entry.relevant_product_ids.includes(product.id) && product.category !== entry.category);
      if (wrongCategory) {
        throw new Error(`Query ${entry.id} is labeled ${entry.category} but relevant product ${wrongCategory.id} is ${wrongCategory.category}.`);
      }
    }
    return { ...entry, path, file: findImageFile(filesByPath, path) };
  });
  return { catalog, queries };
}

function validateManifest(manifest) {
  if (manifest?.version !== 1 || !Array.isArray(manifest.catalog) || !Array.isArray(manifest.queries)) {
    throw new Error("Expected version 1 with catalog and queries arrays.");
  }
  if (!manifest.catalog.length || manifest.catalog.length > MAX_CATALOG_IMAGES) {
    throw new Error(`Catalog must contain 1–${MAX_CATALOG_IMAGES} images.`);
  }
  if (!manifest.queries.length || manifest.queries.length > MAX_QUERIES) {
    throw new Error(`Queries must contain 1–${MAX_QUERIES} images.`);
  }
  for (const product of manifest.catalog) {
    if (!isNonEmptyString(product.id) || !isNonEmptyString(product.image)) {
      throw new Error("Every catalog entry needs a non-empty id and image path.");
    }
    if (!["shoes", "sandals"].includes(product.category)) {
      throw new Error(`Catalog product ${product.id} needs category shoes or sandals.`);
    }
  }
  for (const query of manifest.queries) {
    if (!isNonEmptyString(query.id) || !isNonEmptyString(query.image)) {
      throw new Error("Every query needs a non-empty id and image path.");
    }
    if (!Array.isArray(query.relevant_product_ids) || !query.relevant_product_ids.length
      || query.relevant_product_ids.some((id) => !isNonEmptyString(id))) {
      throw new Error(`Query ${query.id} needs at least one relevant_product_id.`);
    }
    if (new Set(query.relevant_product_ids).size !== query.relevant_product_ids.length) {
      throw new Error(`Query ${query.id} contains duplicate relevant_product_ids.`);
    }
    if (query.category !== undefined && !["all", "shoes", "sandals"].includes(query.category)) {
      throw new Error(`Query ${query.id} has an invalid category.`);
    }
  }
}

function findImageFile(filesByPath, path) {
  const file = filesByPath.get(path.toLowerCase());
  if (!file || !file.type.startsWith("image/")) {
    throw new Error(`Image file not found or unsupported: ${path}`);
  }
  return file;
}

function normalizePath(path) {
  const normalized = String(path).replaceAll("\\", "/").replace(/^\.\/+/, "");
  const segments = normalized.split("/");
  if (!normalized || segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new Error(`Invalid relative image path: ${path}`);
  }
  return normalized;
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

async function runBenchmark() {
  if (!dataset || running) return;
  running = true;
  elements.run.disabled = true;
  elements.folder.disabled = true;
  elements.mode.disabled = true;
  elements.status.textContent = "Starting benchmark… Keep this tab open while images are analyzed.";
  elements.results.hidden = true;
  const modes = elements.mode.value === "both" ? ["fast", "detailed"] : [elements.mode.value];
  const modeResults = [];
  try {
    for (const mode of modes) {
      modeResults.push(await runMode(mode));
    }
    const report = {
      reportVersion: 1,
      generatedAt: new Date().toISOString(),
      model: "Xenova/mobileclip_s0",
      detector: "Xenova/owlvit-base-patch32",
      browser: navigator.userAgent,
      device: {
        webgpuAvailable: Boolean(navigator.gpu),
        hardwareConcurrency: navigator.hardwareConcurrency || null,
        deviceMemoryGb: navigator.deviceMemory || null,
      },
      dataset: {
        catalogImageCount: dataset.catalog.length,
        queryCount: dataset.queries.length,
        productCount: new Set(dataset.catalog.map((product) => product.id)).size,
      },
      metricDefinitions: {
        recallAtK: "Relevant catalog product IDs in the top K divided by all relevant IDs for that query.",
        meanAveragePrecisionAt10: "Macro-average of AP@10; AP@10 divides precision-at-relevant-ranks by min(number of relevant IDs, 10).",
        queryLatencyMs: "Query image embedding plus in-browser ranking after model initialization.",
        catalogImageLatencyMs: "Catalog image embedding; the first image includes model initialization and any first-use downloads.",
      },
      modes: modeResults,
    };
    renderReport(report);
    elements.status.textContent = "Benchmark complete. Download the JSON report to keep your measurements.";
  } catch (error) {
    elements.status.textContent = `Benchmark failed: ${error.message}`;
    console.error(error);
  } finally {
    running = false;
    elements.run.disabled = !dataset;
    elements.folder.disabled = false;
    elements.mode.disabled = false;
  }
}

async function runMode(mode) {
  const modeStartedAt = performance.now();
  const detailed = mode === "detailed";
  const catalog = [];
  const catalogTimes = [];
  for (const [index, product] of dataset.catalog.entries()) {
    setProgress(mode, "catalog", index + 1, dataset.catalog.length, product.id);
    const start = performance.now();
    const embedding = await embedFile(product.file, detailed);
    catalogTimes.push(performance.now() - start);
    catalog.push({ ...product, embedding });
  }

  const queryResults = [];
  const queryTimes = [];
  for (const [index, query] of dataset.queries.entries()) {
    setProgress(mode, "query", index + 1, dataset.queries.length, query.id);
    const start = performance.now();
    const embedding = await embedFile(query.file, detailed);
    const candidates = query.category && query.category !== "all"
      ? catalog.filter((product) => product.category === query.category)
      : catalog;
    const ranked = candidates
      .map((product) => ({ id: product.id, score: similarity(embedding, product.embedding) }))
      .sort((left, right) => right.score - left.score);
    const elapsedMs = performance.now() - start;
    queryTimes.push(elapsedMs);
    const ids = ranked.map((product) => product.id);
    queryResults.push({
      id: query.id,
      category: query.category || "all",
      relevantProductIds: query.relevant_product_ids,
      recallAt1: retrievalMetrics(ids, query.relevant_product_ids, 1).recallAtK,
      recallAt5: retrievalMetrics(ids, query.relevant_product_ids, 5).recallAtK,
      recallAt10: retrievalMetrics(ids, query.relevant_product_ids, 10).recallAtK,
      averagePrecisionAt10: retrievalMetrics(ids, query.relevant_product_ids, TOP_K).averagePrecisionAtK,
      latencyMs: round(elapsedMs),
      topResults: ranked.slice(0, TOP_K),
    });
  }

  return {
    mode,
    totalDurationMs: round(performance.now() - modeStartedAt),
    recallAt1: mean(queryResults.map((query) => query.recallAt1)),
    recallAt5: mean(queryResults.map((query) => query.recallAt5)),
    recallAt10: mean(queryResults.map((query) => query.recallAt10)),
    meanAveragePrecisionAt10: mean(queryResults.map((query) => query.averagePrecisionAt10)),
    queryLatencyMedianMs: percentile(queryTimes, 0.5),
    queryLatencyP95Ms: percentile(queryTimes, 0.95),
    catalogImageLatencyMedianMs: percentile(catalogTimes, 0.5),
    catalogImageLatencyP95Ms: percentile(catalogTimes, 0.95),
    catalogEmbeddingTotalMs: round(catalogTimes.reduce((sum, value) => sum + value, 0)),
    queryResults,
  };
}

async function embedFile(file, detailed) {
  const imageUrl = URL.createObjectURL(file);
  try {
    return await embedImage(imageUrl, (message) => {
      elements.status.textContent = message;
    }, { detailed });
  } finally {
    URL.revokeObjectURL(imageUrl);
  }
}

function setProgress(mode, kind, current, total, id) {
  const label = mode === "fast" ? "Fast" : "Detailed";
  const item = kind === "catalog" ? "catalog image" : "query";
  elements.status.textContent = `${label}: embedding ${item} ${current}/${total} (${id}).`;
}

function renderReport(report) {
  elements.tableBody.replaceChildren();
  elements.queryResults.replaceChildren();
  for (const result of report.modes) {
    const row = document.createElement("tr");
    const cells = [
      result.mode === "fast" ? "Fast" : "Detailed",
      formatMetric(result.recallAt1),
      formatMetric(result.recallAt5),
      formatMetric(result.recallAt10),
      formatMetric(result.meanAveragePrecisionAt10),
      result.queryLatencyMedianMs.toLocaleString(),
      result.queryLatencyP95Ms.toLocaleString(),
      result.catalogImageLatencyMedianMs.toLocaleString(),
      result.totalDurationMs.toLocaleString(),
    ];
    for (const text of cells) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    elements.tableBody.append(row);
    renderQueryResults(result);
  }
  elements.summary.textContent =
    `${report.dataset.queryCount} held-out queries searched against ${report.dataset.catalogImageCount} catalog images representing ${report.dataset.productCount} product IDs.`;
  elements.results.hidden = false;
  if (reportUrl) URL.revokeObjectURL(reportUrl);
  reportUrl = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
  elements.download.dataset.reportUrl = reportUrl;
}

function renderQueryResults(result) {
  const heading = document.createElement("h3");
  heading.textContent = `${result.mode === "fast" ? "Fast" : "Detailed"} per-query results`;
  elements.queryResults.append(heading);
  for (const query of result.queryResults) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent =
      `${query.id} · R@1 ${formatMetric(query.recallAt1)} · R@5 ${formatMetric(query.recallAt5)} · AP@10 ${formatMetric(query.averagePrecisionAt10)} · ${query.latencyMs.toLocaleString()} ms`;
    const relevant = document.createElement("p");
    relevant.textContent = `Relevant IDs: ${query.relevantProductIds.join(", ")}`;
    const ranked = document.createElement("p");
    ranked.textContent = `Top results: ${query.topResults.map((product, index) =>
      `${index + 1}. ${product.id} (${product.score.toFixed(3)})`).join(" · ") || "No results"}`;
    details.append(summary, relevant, ranked);
    elements.queryResults.append(details);
  }
}

function downloadReport() {
  const url = elements.download.dataset.reportUrl;
  if (!url) return;
  const link = document.createElement("a");
  link.href = url;
  link.download = "pairup-retrieval-benchmark.json";
  link.click();
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, percentileValue) {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * percentileValue;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return round(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower));
}

function formatMetric(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function round(value) {
  return Math.round(value);
}
