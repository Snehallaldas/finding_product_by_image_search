import { embedImage, EMBEDDING_VERSION } from "./image-embeddings.js?crop-aware=2";
import { rankProducts } from "./visual-search.js?crop-aware=2";
import { clearProducts, deleteProduct, loadProducts, saveProduct } from "./catalog-store.js";

const elements = {
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
  catalogFiles: document.querySelector("#catalog-files"),
  catalogFolder: document.querySelector("#catalog-folder"),
  addCategory: document.querySelector("#add-category"),
  catalogGrid: document.querySelector("#catalog-grid"),
  catalogEmpty: document.querySelector("#catalog-empty"),
  catalogCount: document.querySelector("#catalog-count"),
  clearCatalog: document.querySelector("#clear-catalog"),
};

const products = [];
let queryEmbedding = null;
let queryUrl = null;
let cameraStream = null;
let nextProductId = 1;
let catalogReady = false;
let catalogImporting = false;
let searchInProgress = false;

initializeCatalog();

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

async function addCatalogFiles(input) {
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
  input.disabled = true;
  elements.clearCatalog.disabled = true;
  elements.catalogFiles.disabled = true;
  elements.catalogFolder.disabled = true;
  let added = 0;
  let failed = 0;
  try {
    for (const [index, file] of files.entries()) {
      let image;
      try {
        image = await loadImage(file);
      } catch (error) {
        failed += 1;
        showNotice(`Could not read "${file.name}". Continuing with the next photo.`);
        console.error(error);
        continue;
      }

      const id = nextProductId++;
      const name = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
      try {
        updateModelStatus(`Analyzing photo ${index + 1} of ${files.length}: ${file.name}`);
        const embedding = await embedImage(image.url, (message) => {
          updateModelStatus(`Photo ${index + 1} of ${files.length}: ${message}`);
        });
        const product = {
          id,
          name,
          category,
          relativePath: file.webkitRelativePath || file.name,
          blob: file,
          embedding,
        };
        await saveProduct(product);
        products.push({ ...product, url: image.url });
        added += 1;
        renderCatalog();
      } catch (error) {
        failed += 1;
        URL.revokeObjectURL(image.url);
        showNotice(`Could not add "${file.name}". Continuing with the next photo.`);
        updateModelStatus("The product was not added because model analysis or local storage failed.");
        console.error(error);
      }
    }
    if (failed > 0) {
      showNotice(`Added ${added} of ${files.length} photos. ${failed} could not be added; check the browser console for details.`);
    } else {
      showNotice(`Added ${added} ${added === 1 ? "photo" : "photos"} to your saved catalog.`);
    }
    updateModelStatus(`${added} ${added === 1 ? "photo" : "photos"} saved on this device.`);
  } finally {
    input.value = "";
    input.disabled = false;
    catalogImporting = false;
    elements.catalogFiles.disabled = !catalogReady;
    elements.catalogFolder.disabled = !catalogReady;
    renderCatalog();
  }
}

elements.searchButton.addEventListener("click", searchCatalog);

elements.categoryFilter.addEventListener("change", () => {
  if (queryEmbedding) searchCatalog();
});

elements.cameraButton.addEventListener("click", startCamera);
elements.captureButton.addEventListener("click", captureCameraImage);
elements.clearQuery.addEventListener("click", clearQuery);
elements.clearCatalog.addEventListener("click", clearCatalog);

async function searchCatalog() {
  if (!queryEmbedding || searchInProgress) return;
  searchInProgress = true;
  elements.searchButton.disabled = true;
  try {
    const oldProducts = products.filter((product) => product.embedding?.version !== EMBEDDING_VERSION);
    for (const [index, product] of oldProducts.entries()) {
      updateModelStatus(`Improving saved image ${index + 1} of ${oldProducts.length}: ${product.name}`);
      const embedding = await embedImage(product.url, (message) => {
        updateModelStatus(`Saved image ${index + 1} of ${oldProducts.length}: ${message}`);
      });
        await saveProduct({
          id: product.id,
          name: product.name,
          category: product.category,
          relativePath: product.relativePath,
          blob: product.blob,
          embedding,
        });
      product.embedding = embedding;
    }

    const ranked = rankProducts(queryEmbedding, products, elements.categoryFilter.value);
    renderResults(ranked);
    showNotice(ranked.length ? "" : "No products are available for this category yet. Add photos or switch categories.");
    if (oldProducts.length) {
      updateModelStatus(`Updated ${oldProducts.length} saved photos for crop-aware visual search.`);
    }
  } catch (error) {
    showNotice("Could not complete the improved visual search. Your saved catalog was not removed; try again.");
    updateModelStatus("Image analysis or local storage failed. Check your connection and browser storage.");
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
    const embedding = await embedImage(image.url, updateModelStatus);
    revokeQueryUrl();
    queryUrl = image.url;
    queryEmbedding = embedding;
    elements.queryImage.src = queryUrl;
    elements.queryImage.hidden = false;
    elements.previewPlaceholder.hidden = true;
    elements.searchButton.disabled = false;
    elements.clearQuery.hidden = false;
    elements.searchNotice.textContent = "";
    elements.resultsGrid.replaceChildren();
    elements.resultsGrid.append(elements.resultsEmpty);
    elements.resultsEmpty.hidden = false;
    elements.resultCount.textContent = "0 MATCHES";
  } catch (error) {
    URL.revokeObjectURL(image.url);
    showNotice("We couldn’t analyze that image. Check your connection and try again.");
    updateModelStatus("AI model could not process that image. Check the browser console for details.");
    console.error(error);
  }
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
    const embedding = await embedImage(url, updateModelStatus);
    revokeQueryUrl();
    queryUrl = url;
    queryEmbedding = embedding;
    elements.queryImage.src = queryUrl;
    elements.queryImage.hidden = false;
    stopCamera();
    elements.previewPlaceholder.hidden = true;
    elements.searchButton.disabled = false;
    showNotice("");
  } catch (error) {
    URL.revokeObjectURL(url);
    showNotice("We couldn’t analyze that camera photo. Check your connection and try again.");
    updateModelStatus("AI model could not process that image. Check the browser console for details.");
    console.error(error);
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
  elements.resultsGrid.replaceChildren();
  elements.resultsGrid.append(elements.resultsEmpty);
  elements.resultsEmpty.hidden = false;
  elements.resultCount.textContent = "0 MATCHES";
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
    await clearProducts();
  } catch (error) {
    showNotice("Could not clear the saved catalog. Check browser storage and try again.");
    console.error(error);
    return;
  } finally {
    elements.clearCatalog.disabled = false;
  }
  for (const product of products) URL.revokeObjectURL(product.url);
  products.length = 0;
  renderCatalog();
  elements.resultsGrid.replaceChildren();
  elements.resultsGrid.append(elements.resultsEmpty);
  elements.resultsEmpty.hidden = false;
  elements.resultCount.textContent = "0 MATCHES";
}

function renderCatalog() {
  elements.catalogGrid.replaceChildren();
  elements.catalogEmpty.hidden = products.length > 0;
  elements.clearCatalog.hidden = products.length === 0;
  elements.clearCatalog.disabled = !catalogReady || products.length === 0 || catalogImporting;
  elements.catalogCount.textContent = catalogReady
    ? `${products.length} ${products.length === 1 ? "product" : "products"} saved on this device`
    : "Loading saved products…";
  if (!products.length) {
    elements.catalogGrid.append(elements.catalogEmpty);
    return;
  }
  for (const product of products) {
    elements.catalogGrid.append(createProductCard(product, false));
  }
}

function renderResults(ranked) {
  elements.resultsGrid.replaceChildren();
  elements.resultCount.textContent = `${ranked.length} ${ranked.length === 1 ? "MATCH" : "MATCHES"}`;
  if (!ranked.length) {
    elements.resultsGrid.append(elements.resultsEmpty);
    elements.resultsEmpty.hidden = false;
    return;
  }
  elements.resultsEmpty.hidden = true;
  for (const product of ranked) {
    elements.resultsGrid.append(createProductCard(product, true));
  }
}

function createProductCard(product, showScore) {
  const card = document.createElement("article");
  card.className = "product-card";
  const imageWrap = document.createElement("div");
  imageWrap.className = "product-image-wrap";
  const image = document.createElement("img");
  image.src = product.url;
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
  const index = products.findIndex((product) => product.id === id);
  if (index < 0) return;
  try {
    await deleteProduct(id);
  } catch (error) {
    showNotice("Could not remove the saved product. Check browser storage and try again.");
    console.error(error);
    return;
  }
  URL.revokeObjectURL(products[index].url);
  products.splice(index, 1);
  renderCatalog();
  if (queryEmbedding) searchCatalog();
}

function showNotice(message) {
  elements.searchNotice.textContent = message;
}

function updateModelStatus(message) {
  elements.modelStatus.textContent = message;
}

async function initializeCatalog() {
  elements.catalogFiles.disabled = true;
  elements.catalogFolder.disabled = true;
  elements.clearCatalog.disabled = true;
  elements.catalogCount.textContent = "Loading saved products…";
  try {
    const savedProducts = await loadProducts();
    for (const product of savedProducts) {
      if (!(product.blob instanceof Blob) || !product.embedding) {
        throw new Error(`Saved product ${product.id} is missing its image or embedding.`);
      }
      products.push({ ...product, url: URL.createObjectURL(product.blob) });
      nextProductId = Math.max(nextProductId, product.id + 1);
    }
    catalogReady = true;
    renderCatalog();
  } catch (error) {
    showNotice("Could not load the saved catalog. Your saved images were not changed.");
    elements.catalogCount.textContent = "Saved catalog could not be loaded";
    console.error(error);
  } finally {
    elements.catalogFiles.disabled = !catalogReady;
    elements.catalogFolder.disabled = !catalogReady;
    elements.clearCatalog.disabled = !catalogReady || products.length === 0;
  }
}
