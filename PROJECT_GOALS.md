# PairUp: Footwear Visual Search

## Product goal

Build a portfolio-ready web app that finds visually similar shoes and sandals from a user's private catalog using an uploaded photo or live camera capture. Query photos stay in the browser; catalog photos are stored on the server in per-account private storage.

## Checkpoints

### MVP — browser-based visual search

- [x] Create a responsive, accessible search experience for footwear.
- [x] Let users add, remove, and clear shoe and sandal photos in a local catalog.
- [x] Keep the original browser-only catalog saved locally after a page refresh.
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
- [x] Add fast-by-default analysis, optional detailed OWL-ViT footwear detections, and WebGPU acceleration with CPU fallback.
- [x] Add MobileCLIP object-crop embeddings for detector-focused matching when detailed analysis is selected.
- [x] Build a browser-local labeled retrieval benchmark that compares Fast and Detailed modes and exports metrics and latency.
- [x] Compare single-view and multiple-view product references on the same held-out real-photo queries.
- [x] Measure category filtering against unfiltered retrieval on the same 23 held-out queries.
- [x] Compare Fast and Detailed retrieval on the same category-filtered real-photo queries.
- [ ] Validate detector-focused matching on a broader labeled footwear query set; the initial six-photo S1 experiment moved the matching photo from 5th to 2nd.
- [x] Assemble an initial legally usable, labeled footwear catalog with shoes and sandals.
- [x] Prepare a separate real-photo benchmark set from the latest numbered images (11 products, 31 queries; exclude the group flagged for relabeling).
- [x] Run Fast and Detailed on the new photo set and save per-query results.
- [x] Confirm product-level non-overlap with the existing six-product pilot and run Fast/Detailed on the combined labeled set.
- [x] Exclude pair-016 after identifying a heel/back mismatch between its catalog and query photos; prepare a corrected evaluation set.
- [x] Rerun Fast and Detailed on the corrected 16-product, 34-query set and save per-query results.
- [x] Remove duplicate catalog entries for pairs 002/006 and 004/005; replace pair-002's back-only reference with the user's clearer photo.
- [x] Rerun Fast and Detailed on the final 14-product, 34-query label-cleaned set and save per-query results.
- [ ] Create and label a broader repeatable query set with real-world photos, varied backgrounds, and lighting; the current ABO queries are seller alternate views only.
- [x] Measure Recall@K and search latency on the initial ABO set and record the baseline in the project documentation.
- [x] Compare candidate embedding models based on measured quality, runtime, and deployment constraints.

### Private backend and catalog scale

- [x] Add per-account registration, sign-in, and private catalog APIs.
- [x] Store catalog photos and embeddings on the server and restrict API/image access to the owning account.
- [x] Add paginated catalog and search results with exact vector scoring for catalogs up to 10,000 items.
- [x] Save each completed import immediately and skip previously uploaded files on re-import.
- [ ] Move image analysis into a server-side resumable worker so indexing does not require an open browser tab.
- [ ] Add password recovery, email verification, account deletion/export, rate limiting, backup, and deployment hardening before public launch.

### Portfolio polish

- [ ] Improve product metadata and filtering (for example style, color, and price).
- [ ] Add automated end-to-end coverage for upload, search, camera permission states, and catalog actions.
- [ ] Publish a demo and include screenshots, architecture, evaluation results, and known limitations.

## Current implementation notes

- Fast analysis is the default: MobileCLIP-S0 computes a whole-image and centered-crop embedding, using WebGPU when available and falling back to CPU. Detailed mode adds overlapping crops, runs OWL-ViT, and computes up to two footwear-focused embeddings.
- Detailed embeddings compare detected footwear crops when both query and product have them. Fast and detailed entries can still be compared through their whole-image/crop embeddings.
- The optional detector downloads about 127 MB on first use and runs for each image; leave analysis in Fast mode for quicker searches and catalog imports.
- `/benchmark.html` runs Fast and Detailed retrieval evaluation locally from a labeled `benchmark.json` folder and exports Recall@K, mAP@10, and inference latency. No benchmark dataset or user images are included in the repository.
- Paired Fast-mode reference-view experiment (14 products, same 23 held-out queries): adding 11 catalog views raised Recall@5 from 43.5% to 60.9%, while Recall@1 changed from 13.0% to 8.7%, Recall@10 stayed at 82.6%, and mAP@10 changed from 27.2% to 26.9%. Median query latency was similar (2,038 ms vs. 2,031 ms); the 25-image catalog took longer to embed than the 14-image catalog. This small, device-specific set does not establish a general benefit. Reports and ignored datasets are under `benchmark-data/multi-view-split-single-reference/` and `benchmark-data/multi-view-split-multiple-references/`.
- Paired Fast-mode category-filter experiment (14 products, same 23 held-out queries and 25 references): assigning each query its labeled shoe/sandal category left Recall@1 at 8.7%, while Recall@5 rose from 60.9% to 69.6%, Recall@10 from 82.6% to 95.7%, and mAP@10 from 26.9% to 36.2%. Category filtering helped retrieval depth but did not improve top-1 accuracy. The report and ignored datasets are under `benchmark-data/multi-view-split-category-unfiltered/` and `benchmark-data/multi-view-split-category-filtered/`.
- Paired Fast-vs-Detailed category-filtered run (14 products, same 23 held-out queries and 25 references): Detailed raised Recall@1 from 8.7% to 17.4%, but reduced Recall@5 from 69.6% to 65.2%, Recall@10 from 95.7% to 91.3%, and mAP@10 from 36.2% to 34.7%. Median query latency increased from 1,235 ms to 11,466 ms, and the Detailed run took 721,616 ms total. Pair-008 queries ranked 2nd/3rd in Fast and 10th/6th in Detailed. Results are device- and dataset-specific; see the report in the Git-ignored `benchmark-data/multi-view-split-category-filtered/detailed-results.json`.
- Five-query alternative-model pilot: DINOv2-small reached Recall@1 20.0%, Recall@5 100.0%, Recall@10 100.0%, and mAP@10 49.0%, versus MobileCLIP's 20.0%, 80.0%, 100.0%, and 48.9% on the same queries. Median query latency was similar (1,274 ms vs. 1,252 ms). This is too small to select a production model; a full comparison remains open. The ignored pilot report is under `benchmark-data/dinov2-small-pilot/`.
- Full Fast-mode model comparison (14 products, 25 references, 23 category-filtered queries): DINOv2-small reached Recall@1 34.8%, Recall@5 87.0%, Recall@10 91.3%, and mAP@10 53.6%, versus MobileCLIP's 8.7%, 69.6%, 95.7%, and 36.2%. Median query latency was 1,607 ms versus 1,235 ms. DINOv2-small improved top-1, top-5, and mAP but reduced Recall@10 and increased latency; keep the choice configurable until a broader benchmark is available. The ignored report is under `benchmark-data/dinov2-small-full/`.
- Initial ABO Fast baseline (50 products, 100 seller alternate-view queries): Recall@1 10.0%, Recall@5 38.0%, Recall@10 60.0%, mAP@10 22.7%; median query latency 1,921 ms and p95 2,017 ms. See `README.md` for device, split, and interpretation details; the generated report is in the Git-ignored local dataset folder.
- The initial detector-crop experiment on six saved photos moved the matching S1 photo from 5th (0.495) to 2nd (0.833), but this is too small a sample to establish general retrieval quality.
- The inference runtime and quantized weights load from public CDNs on first use and are cached by the browser.
- Images are passed to the browser runtime, not the model providers. Catalog image files are uploaded to the private account's server storage; query photos remain local and only the embeddings are sent for search.
- The older browser catalog remains in IndexedDB and can be copied into an account without deleting the local copy.
- The current browser-side importer processes large folders serially; indexing 10,000 photos may take hours and requires the browser tab to remain open.
- Search uses a NumPy exact scan across account embeddings and paginates returned products. A background model worker and dedicated vector index remain future work.
- Similarity scores are ranking signals, not probabilities. This prototype does not search online product listings.
