# PairUp: Footwear Visual Search

## Product goal

Build a portfolio-ready web app that finds visually similar shoes and sandals from a user-provided product catalog using an uploaded photo or a live camera capture. Keep user images local to the browser unless a later, explicit product decision changes that privacy model.

## Checkpoints

### MVP — browser-based visual search

- [x] Create a responsive, accessible search experience for footwear.
- [x] Let users add, remove, and clear shoe and sandal photos in a local catalog.
- [x] Keep catalog photos and embeddings saved locally after a page refresh.
- [x] Import multiple images from a folder into the local catalog.
- [x] Let users search with an uploaded image and rank the catalog by visual similarity.
- [x] Let users capture a query photo using a live camera.
- [x] Add category filtering and useful empty, loading, and error states.
- [x] Test embedding normalization, ranking, filtering, and invalid input.
- [x] Document setup, usage, limitations, and the next project steps.

### Retrieval quality

- [x] Replace handcrafted color-and-edge features with pretrained MobileCLIP-S0 image embeddings for catalog and query photos.
- [x] Add visible model download/inference status and avoid silent heuristic fallback.
- [x] Compare whole-image and overlapping crop embeddings to reduce sensitivity to product count and placement.
- [ ] Assemble a legally usable, labeled footwear catalog with varied shoes and sandals.
- [ ] Define a repeatable query set with different views, backgrounds, and lighting.
- [ ] Measure Recall@K and search latency; record a baseline in the project documentation.
- [ ] Compare candidate embedding models based on measured quality, runtime, and deployment constraints.

### Portfolio polish

- [ ] Improve product metadata and filtering (for example style, color, and price).
- [ ] Add automated end-to-end coverage for upload, search, camera permission states, and catalog actions.
- [ ] Publish a demo and include screenshots, architecture, evaluation results, and known limitations.

## Current implementation notes

- The retrieval prototype uses browser-side MobileCLIP-S0 image embeddings with cosine similarity.
- Each image is represented by its whole-image embedding plus overlapping square crops; ranking blends global similarity with the best crop similarity.
- Older single-vector catalog entries are upgraded and saved when they are first included in a search.
- The inference runtime and quantized weights load from public CDNs on first use and are cached by the browser.
- Images are passed to the local browser runtime; the app does not upload them to the model provider.
- Product catalog images and embeddings are saved locally in IndexedDB; query photos are temporary.
- The catalog is local to this browser/device and does not sync to a server or other devices.
- Similarity scores are ranking signals, not probabilities. This prototype does not search online product listings.
