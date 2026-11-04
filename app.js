const requestedModel = new URL(location.href).searchParams.get("model");
const appModel = requestedModel === "dinov2" ? "dinov2" : "mobileclip";
const {
  embedImage,
  EMBEDDING_VERSION,
} = await import(`./image-embeddings.js?object-aware=1&model=${appModel}`);
import { loadProducts } from "./catalog-store.js";

const CATALOG_PAGE_SIZE = 32;
const RESULTS_PAGE_SIZE = 20;

const elements = {
  accountForm: document.querySelector("#account-form"),
  accountEmail: document.querySelector("#account-email"),
  accountPassword: document.querySelector("#account-password"),
  accountSubmit: document.querySelector("#account-submit"),
  accountMode: document.querySelector("#account-mode"),
  accountNotice: document.querySelector("#account-notice"),
  accountPanel: document.querySelector("#account-panel"),
  accountTitle: document.querySelector("#account-title"),
  signedIn: document.querySelector("#signed-in"),
  accountEmailLabel: document.querySelector("#account-email-label"),
  analysisMode: document.querySelector("#analysis-mode"),
  signOut: document.querySelector("#sign-out"),
  migrateCatalog: document.querySelector("#migrate-catalog"),
  searchSection: document.querySelector("#search-section"),
  resultsSection: document.querySelector("#results-section"),
  catalogSection: document.querySelector("#catalog-section"),
  queryFile: document.querySelector("#query-file"),
  queryImage: document.querySelector("#query-image"),
  previewPlaceholder: document.querySelector("#preview-placeholder"),
  cameraButton: document.querySelector("#camera-button"),
  captureButton: document.querySelector("#capture-button"),
  cameraVideo: document.querySelector("#camera-video"),
  clearQuery: document.querySelector("#clear-query"),
  searchButton: document.querySelector("#search-button"),
  categoryFilter: document.querySelector("#category-filter"),
  searchNotice: document.querySelector("#search-notice"),
  modelStatus: document.querySelector("#model-status"),
  resultsGrid: document.querySelector("#results-grid"),
  resultsEmpty: document.querySelector("#results-empty"),
  resultCount: document.querySelector("#result-count"),
  resultsPagination: document.querySelector("#results-pagination"),
  resultsPrevious: document.querySelector("#results-previous"),
  resultsNext: document.querySelector("#results-next"),
  resultsPageLabel: document.querySelector("#results-page-label"),
  catalogFiles: document.querySelector("#catalog-files"),
  catalogFolder: document.querySelector("#catalog-folder"),
  addCategory: document.querySelector("#add-category"),
  catalogGrid: document.querySelector("#catalog-grid"),
  catalogEmpty: document.querySelector("#catalog-empty"),
  catalogCount: document.querySelector("#catalog-count"),
  clearCatalog: document.querySelector("#clear-catalog"),
  catalogPagination: document.querySelector("#catalog-pagination"),
  catalogPrevious: document.querySelector("#catalog-previous"),
  catalogNext: document.querySelector("#catalog-next"),
  catalogPageLabel: document.querySelector("#catalog-page-label"),
};

let products = [];
let currentAccount = null;
let queryEmbedding = null;
let queryUrl = null;
let cameraStream = null;
let catalogPage = 1;
let catalogTotal = 0;
let resultsPage = 1;
let resultsTotal = 0;
let catalogImporting = false;
let searchInProgress = false;
let accountMode = "login";
let activeAnalysisMode = "fast";

elements.accountForm.addEventListener("submit", authenticate);
elements.accountMode.addEventListener("click", toggleAccountMode);
elements.signOut.addEventListener("click", signOut);
elements.migrateCatalog.addEventListener("click", migrateLocalCatalog);
elements.queryFile.addEventListener("change", async (event) => {
  const input = event.currentTarget;
  const [file] = input.files;
  input.disabled = true;
  try {
    if (file) await setQueryImage(file);
  } finally {
    input.value = "";
    input.disabled = false;
  }
});
elements.catalogFiles.addEventListener("change", (event) => addCatalogFiles(event.currentTarget));
elements.catalogFolder.addEventListener("change", (event) => addCatalogFiles(event.currentTarget));
elements.searchButton.addEventListener("click", () => searchCatalog(1));
elements.analysisMode.addEventListener("change", reanalyzeQueryImage);
elements.categoryFilter.addEventListener("change", () => {
  if (queryEmbedding) searchCatalog(1);
});
elements.cameraButton.addEventListener("click", startCamera);
elements.captureButton.addEventListener("click", captureCameraImage);
elements.clearQuery.addEventListener("click", clearQuery);
elements.clearCatalog.addEventListener("click", clearCatalog);
elements.catalogPrevious.addEventListener("click", () => loadCatalog(catalogPage - 1));
elements.catalogNext.addEventListener("click", () => loadCatalog(catalogPage + 1));
elements.resultsPrevious.addEventListener("click", () => searchCatalog(resultsPage - 1));
elements.resultsNext.addEventListener("click", () => searchCatalog(resultsPage + 1));

initializeApp();

async function initializeApp() {
  try {
    currentAccount = await api("/api/auth/me");
    await showSignedIn();
  } catch (error) {
    if (error.status !== 401) {
      elements.accountNotice.textContent = error.message;
      console.error(error);
    }
    showSignedOut();
  }
  await updateLocalMigrationOption();
}

async function authenticate(event) {
  event.preventDefault();
  elements.accountSubmit.disabled = true;
  elements.accountNotice.textContent = "";
  try {
    const endpoint = accountMode === "register" ? "/api/auth/register" : "/api/auth/login";
    currentAccount = await api(endpoint, {
      method: "POST",
      body: JSON.stringify({
        email: elements.accountEmail.value,
        password: elements.accountPassword.value,
      }),
    });
    elements.accountPassword.value = "";
    await showSignedIn();
    await updateLocalMigrationOption();
  } catch (error) {
    elements.accountNotice.textContent = error.message;
  } finally {
    elements.accountSubmit.disabled = false;
  }
}

function toggleAccountMode() {
  accountMode = accountMode === "login" ? "register" : "login";
  elements.accountTitle.textContent = accountMode === "register" ? "Create your account" : "Sign in to your catalog";
  elements.accountSubmit.textContent = accountMode === "register" ? "Create account" : "Sign in";
  elements.accountMode.textContent = accountMode === "register" ? "Already have an account? Sign in" : "Create an account";
  elements.accountPassword.autocomplete = accountMode === "register" ? "new-password" : "current-password";
  elements.accountNotice.textContent = "";
}

function showSignedOut() {
  currentAccount = null;
  elements.accountForm.hidden = false;
  elements.signedIn.hidden = true;
  elements.searchSection.hidden = true;
  elements.resultsSection.hidden = true;
  elements.catalogSection.hidden = true;
}

async function showSignedIn() {
  elements.accountForm.hidden = true;
  elements.signedIn.hidden = false;
  elements.accountEmailLabel.textContent = `Signed in as ${currentAccount.email}`;
  elements.searchSection.hidden = false;
  elements.resultsSection.hidden = false;
  elements.catalogSection.hidden = false;
  elements.modelStatus.textContent = appModel === "dinov2"
    ? "DINOv2-small is active (about 25 MB). Internet is needed for the initial download."
    : "Fast mode loads MobileCLIP on first use (about 12 MB). Internet is needed for the initial download.";
  showNotice("");
  await loadCatalog(1);
}

async function signOut() {
  elements.signOut.disabled = true;
  try {
    await api("/api/auth/logout", { method: "POST" });
    clearQuery();
    products = [];
    catalogTotal = 0;
    renderCatalog();
    showSignedOut();
  } catch (error) {
    showNotice(error.message);
  } finally {
    elements.signOut.disabled = false;
  }
}

async function updateLocalMigrationOption() {
  try {
    const localProducts = await loadProducts();
    elements.migrateCatalog.hidden = !currentAccount || localProducts.length === 0;
    if (localProducts.length) {
      elements.migrateCatalog.textContent = `Import ${localProducts.length} previous browser photos into this account`;
    }
  } catch (error) {
    elements.migrateCatalog.hidden = true;
    console.error(error);
  }
}

async function migrateLocalCatalog() {
  elements.migrateCatalog.disabled = true;
  elements.analysisMode.disabled = true;
  try {
    const localProducts = await loadProducts();
    let imported = 0;
    let skipped = 0;
    let failed = 0;
    for (const [index, product] of localProducts.entries()) {
      let imageUrl;
      try {
        const existing = await api(`/api/products/check?content_hash=${await hashFile(product.blob)}`);
        if (existing.exists) {
          skipped += 1;
          continue;
        }
        let embedding = product.embedding;
        if (embedding?.version !== EMBEDDING_VERSION) {
          imageUrl = URL.createObjectURL(product.blob);
          embedding = await analyzeImage(imageUrl, (message) => {
            updateModelStatus(`Preparing saved photo ${index + 1} of ${localProducts.length}: ${message}`);
          });
        }
        await uploadProduct(product.blob, product.name, product.category, embedding);
        imported += 1;
        updateModelStatus(`Imported ${index + 1} of ${localProducts.length} saved photos.`);
      } catch (error) {
        failed += 1;
        console.error(error);
      } finally {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
      }
    }
    await loadCatalog(1);
    showNotice(failed
      ? `Imported ${imported}; skipped ${skipped} already saved; ${failed} failed. Your browser copy is still intact.`
      : `Imported ${imported}; skipped ${skipped} already saved. Your original browser copy is still intact.`);
    await updateLocalMigrationOption();
  } catch (error) {
    showNotice(`Could not import the previous collection: ${error.message}`);
    console.error(error);
  } finally {
    elements.migrateCatalog.disabled = false;
    elements.analysisMode.disabled = false;
  }
}

async function addCatalogFiles(input) {
  if (!currentAccount) return;
  const selectedFiles = [...input.files];
  const files = selectedFiles.filter((file) => file.type.startsWith("image/"));
  if (!files.length) {
    showNotice(selectedFiles.length
      ? "No supported images were found. Choose JPG, PNG, GIF or WEBP photos."
      : "Choose one or more image files, or a folder containing photos.");
    input.value = "";
    return;
  }
  showNotice("");
  const category = elements.addCategory.value;
  catalogImporting = true;
  elements.analysisMode.disabled = true;
  input.disabled = true;
  elements.clearCatalog.disabled = true;
  elements.catalogFiles.disabled = true;
  elements.catalogFolder.disabled = true;
  let added = 0;
  let skipped = 0;
  let failed = 0;
  try {
    for (const [index, file] of files.entries()) {
      let imageUrl;
      try {
        const checksum = await hashFile(file);
        const existing = await api(`/api/products/check?content_hash=${checksum}`);
        if (existing.exists) {
          skipped += 1;
          updateModelStatus(`Skipping already indexed photo ${index + 1} of ${files.length}: ${file.name}`);
          continue;
        }
        imageUrl = URL.createObjectURL(file);
        const name = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
        updateModelStatus(`Analyzing photo ${index + 1} of ${files.length}: ${file.name}`);
        const embedding = await analyzeImage(imageUrl, (message) => {
          updateModelStatus(`Photo ${index + 1} of ${files.length}: ${message}`);
        });
        await uploadProduct(file, name, category, embedding);
        added += 1;
      } catch (error) {
        failed += 1;
        showNotice(`Could not add "${file.name}". Continuing with the next photo.`);
        console.error(error);
      } finally {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
      }
    }
    await loadCatalog(1);
    showNotice(failed
      ? `Added ${added}; skipped ${skipped} already indexed; ${failed} could not be uploaded. Check the browser console for details.`
      : `Added ${added}; skipped ${skipped} already indexed in your private catalog.`);
    updateModelStatus(`${added} new ${added === 1 ? "photo" : "photos"} uploaded to your account.`);
  } finally {
    input.value = "";
    input.disabled = false;
    catalogImporting = false;
    elements.analysisMode.disabled = false;
    elements.catalogFiles.disabled = false;
    elements.catalogFolder.disabled = false;
    renderCatalog();
  }
}

async function uploadProduct(file, name, category, embedding) {
  const form = new FormData();
  form.append("image", file, file.name || `${name}.jpg`);
  form.append("name", name);
  form.append("category", category);
  form.append("embedding", JSON.stringify(serializeEmbedding(embedding)));
  return api("/api/products", { method: "POST", body: form });
}

async function hashFile(file) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function serializeEmbedding(embedding) {
  if (ArrayBuffer.isView(embedding)) return Array.from(embedding);
  return {
    ...embedding,
    vectors: embedding.vectors.map((vector) => Array.from(vector)),
    objectVectors: (embedding.objectVectors ?? []).map((vector) => Array.from(vector)),
  };
}

async function searchCatalog(page) {
  if (!queryEmbedding || !currentAccount || searchInProgress) return;
  searchInProgress = true;
  elements.searchButton.disabled = true;
  try {
    const response = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({
        embedding: serializeEmbedding(queryEmbedding),
        category: elements.categoryFilter.value,
        page,
        page_size: RESULTS_PAGE_SIZE,
      }),
    });
    resultsPage = response.page;
    resultsTotal = response.total;
    renderResults(response.products);
    showNotice(response.total ? "" : "No products are available for this category yet. Add photos or switch categories.");
    updateModelStatus(`Found ${response.total.toLocaleString()} products in your private catalog.`);
  } catch (error) {
    showNotice(`Could not search your private catalog: ${error.message}`);
    console.error(error);
  } finally {
    searchInProgress = false;
    elements.searchButton.disabled = !queryEmbedding;
  }
}

async function setQueryImage(file) {
  if (!file.type.startsWith("image/")) {
    showNotice("Choose an image file to search.");
    return;
  }
  let image;
  try {
    stopCamera();
    image = await loadImage(file);
  } catch (error) {
    showNotice("We couldn’t read that image. Try a different photo.");
    console.error(error);
    return;
  }
  try {
    elements.analysisMode.disabled = true;
    const selectedMode = elements.analysisMode.value;
    const embedding = await analyzeImage(image.url, updateModelStatus);
    revokeQueryUrl();
    queryUrl = image.url;
    queryEmbedding = embedding;
    activeAnalysisMode = selectedMode;
    elements.queryImage.src = queryUrl;
    elements.queryImage.hidden = false;
    elements.previewPlaceholder.hidden = true;
    elements.searchButton.disabled = false;
    elements.clearQuery.hidden = false;
    elements.searchNotice.textContent = "";
    resultsPage = 1;
    resultsTotal = 0;
    renderResults([]);
  } catch (error) {
    URL.revokeObjectURL(image.url);
    showNotice("We couldn’t analyze that image. Check your connection and try again.");
    updateModelStatus("AI model could not process that image. Check the browser console for details.");
    console.error(error);
  } finally {
    elements.analysisMode.disabled = false;
  }
}

async function reanalyzeQueryImage() {
  if (!queryUrl || catalogImporting || !queryEmbedding) return;
  const previousMode = activeAnalysisMode;
  elements.analysisMode.disabled = true;
  elements.searchButton.disabled = true;
  try {
    queryEmbedding = await analyzeImage(queryUrl, updateModelStatus);
    activeAnalysisMode = elements.analysisMode.value;
    resultsPage = 1;
    await searchCatalog(1);
  } catch (error) {
    elements.analysisMode.value = previousMode;
    showNotice(`Could not change image analysis mode: ${error.message}`);
    console.error(error);
  } finally {
    elements.analysisMode.disabled = false;
    elements.searchButton.disabled = !queryEmbedding;
  }
}

function analyzeImage(imageUrl, onProgress) {
  return embedImage(imageUrl, onProgress, {
    detailed: elements.analysisMode.value === "detailed",
  });
}

async function loadImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return { url };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showNotice("Live camera is unavailable here. Open this page on localhost or use an image upload.");
    return;
  }
  try {
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    elements.cameraVideo.srcObject = cameraStream;
    elements.cameraVideo.hidden = false;
    elements.queryImage.hidden = true;
    elements.previewPlaceholder.hidden = true;
    elements.cameraButton.hidden = true;
    elements.captureButton.hidden = false;
    elements.clearQuery.hidden = false;
    showNotice("");
  } catch (error) {
    showNotice(error.name === "NotAllowedError" ? "Camera access was denied. Allow camera permission or upload a photo." : "Could not start the camera. Try uploading a photo instead.");
    console.error(error);
  }
}

async function captureCameraImage() {
  if (!cameraStream || !elements.cameraVideo.videoWidth) {
    showNotice("The camera is still starting. Please try again in a moment.");
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = elements.cameraVideo.videoWidth;
  canvas.height = elements.cameraVideo.videoHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    showNotice("Could not capture the camera image. Please upload a photo instead.");
    return;
  }
  context.drawImage(elements.cameraVideo, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) {
    showNotice("Could not save the camera image. Please try again.");
    return;
  }
  const url = URL.createObjectURL(blob);
  try {
    elements.analysisMode.disabled = true;
    const selectedMode = elements.analysisMode.value;
    const embedding = await analyzeImage(url, updateModelStatus);
    revokeQueryUrl();
    queryUrl = url;
    queryEmbedding = embedding;
    activeAnalysisMode = selectedMode;
    elements.queryImage.src = queryUrl;
    elements.queryImage.hidden = false;
    stopCamera();
    elements.previewPlaceholder.hidden = true;
    elements.searchButton.disabled = false;
    showNotice("");
    resultsPage = 1;
    resultsTotal = 0;
    renderResults([]);
  } catch (error) {
    URL.revokeObjectURL(url);
    showNotice("We couldn’t analyze that camera photo. Check your connection and try again.");
    updateModelStatus("AI model could not process that image. Check the browser console for details.");
    console.error(error);
  } finally {
    elements.analysisMode.disabled = false;
  }
}

function clearQuery() {
  stopCamera();
  revokeQueryUrl();
  queryEmbedding = null;
  elements.queryImage.removeAttribute("src");
  elements.queryImage.hidden = true;
  elements.previewPlaceholder.hidden = false;
  elements.searchButton.disabled = true;
  elements.clearQuery.hidden = true;
  elements.cameraButton.hidden = false;
  elements.captureButton.hidden = true;
  resultsPage = 1;
  resultsTotal = 0;
  renderResults([]);
  showNotice("");
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = null;
  }
  elements.cameraVideo.srcObject = null;
  elements.cameraVideo.hidden = true;
  elements.cameraButton.hidden = false;
  elements.captureButton.hidden = true;
}

function revokeQueryUrl() {
  if (queryUrl) URL.revokeObjectURL(queryUrl);
  queryUrl = null;
}

async function clearCatalog() {
  elements.clearCatalog.disabled = true;
  try {
    const response = await api("/api/products", { method: "DELETE" });
    catalogTotal = 0;
    catalogPage = 1;
    products = [];
    renderCatalog();
    renderResults([]);
    showNotice(`Removed ${response.deleted} products from your private catalog.`);
  } catch (error) {
    showNotice(`Could not clear the catalog: ${error.message}`);
    console.error(error);
  } finally {
    elements.clearCatalog.disabled = false;
  }
}

async function loadCatalog(page) {
  const response = await api(`/api/products?page=${page}&page_size=${CATALOG_PAGE_SIZE}`);
  catalogPage = response.page;
  catalogTotal = response.total;
  products = response.products;
  renderCatalog();
}

function renderCatalog() {
  elements.catalogGrid.replaceChildren();
  elements.catalogEmpty.hidden = catalogTotal > 0;
  elements.clearCatalog.hidden = catalogTotal === 0;
  elements.clearCatalog.disabled = catalogTotal === 0 || catalogImporting;
  elements.catalogCount.textContent = `${catalogTotal.toLocaleString()} ${catalogTotal === 1 ? "product" : "products"} in your private catalog`;
  if (!catalogTotal) {
    elements.catalogGrid.append(elements.catalogEmpty);
  } else {
    products.forEach((product) => elements.catalogGrid.append(createProductCard(product, false)));
  }
  renderPagination(elements.catalogPagination, elements.catalogPageLabel, elements.catalogPrevious, elements.catalogNext, catalogPage, catalogTotal, CATALOG_PAGE_SIZE);
}

function renderResults(ranked) {
  elements.resultsGrid.replaceChildren();
  elements.resultCount.textContent = `${resultsTotal.toLocaleString()} ${resultsTotal === 1 ? "MATCH" : "MATCHES"}`;
  if (!ranked.length) {
    elements.resultsGrid.append(elements.resultsEmpty);
    elements.resultsEmpty.hidden = false;
  } else {
    elements.resultsEmpty.hidden = true;
    ranked.forEach((product) => elements.resultsGrid.append(createProductCard(product, true)));
  }
  renderPagination(elements.resultsPagination, elements.resultsPageLabel, elements.resultsPrevious, elements.resultsNext, resultsPage, resultsTotal, RESULTS_PAGE_SIZE);
}

function renderPagination(container, label, previous, next, page, total, pageSize) {
  const pageCount = Math.ceil(total / pageSize);
  container.hidden = pageCount <= 1;
  label.textContent = `Page ${page} of ${pageCount.toLocaleString()}`;
  previous.disabled = page <= 1;
  next.disabled = page >= pageCount;
}

function createProductCard(product, showScore) {
  const card = document.createElement("article");
  card.className = "product-card";
  const imageWrap = document.createElement("div");
  imageWrap.className = "product-image-wrap";
  const image = document.createElement("img");
  image.src = product.imageUrl;
  image.alt = product.name;
  image.loading = "lazy";
  imageWrap.append(image);

  if (showScore) {
    const badge = document.createElement("span");
    badge.className = "match-badge";
    badge.textContent = `SIMILARITY ${product.score.toFixed(2)}`;
    imageWrap.append(badge);
  } else {
    const remove = document.createElement("button");
    remove.className = "remove-product";
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${product.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => removeProduct(product.id));
    imageWrap.append(remove);
  }

  const info = document.createElement("div");
  info.className = "product-info";
  const name = document.createElement("p");
  name.className = "product-name";
  name.textContent = product.name;
  const category = document.createElement("p");
  category.className = "product-category";
  category.textContent = product.category;
  info.append(name, category);
  card.append(imageWrap, info);
  return card;
}

async function removeProduct(id) {
  try {
    await api(`/api/products/${encodeURIComponent(id)}`, { method: "DELETE" });
    const remaining = Math.max(0, catalogTotal - 1);
    const targetPage = Math.min(catalogPage, Math.max(1, Math.ceil(remaining / CATALOG_PAGE_SIZE)));
    await loadCatalog(targetPage);
    if (queryEmbedding) await searchCatalog(resultsPage);
  } catch (error) {
    showNotice(`Could not remove the product: ${error.message}`);
    console.error(error);
  }
}

async function api(url, options = {}) {
  const headers = new Headers(options.headers ?? {});
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...options, headers, credentials: "same-origin" });
  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const error = new Error(body?.detail ?? `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function showNotice(message) {
  elements.searchNotice.textContent = message;
}

function updateModelStatus(message) {
  elements.modelStatus.textContent = message;
}
