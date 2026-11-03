# PairUp — Private Footwear Search

A portfolio prototype for searching a personal shoe and sandal catalog with an image. Accounts keep catalogs separate. The browser analyzes images; a FastAPI service stores catalog photos and embeddings, and searches the account's catalog.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000), create an account with a password of at least 12 characters, then upload or select a folder of footwear photos. Server data is stored in the `pairup-data` directory by default. Set `PAIRUP_DATA_DIR` to choose another persistent location.

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000`.

## Data and privacy

- Catalog photos are uploaded to the server and stored on disk, scoped to the signed-in account. The server requires the account's session cookie to list, search, or retrieve those images.
- Search photos remain in the browser. The browser sends the query embedding, not the query photo, to the API for ranking.
- Account passwords are stored as salted PBKDF2 hashes; session cookies are HttpOnly and SameSite=Lax. For HTTPS deployments, set `PAIRUP_SECURE_COOKIE=1`.
- Each account is limited to 10,000 products and 2 GB of stored images; individual images are limited to 15 MB.
- The previous browser-only collection is retained in IndexedDB. Sign in, then choose **Import previous browser collection** to copy it to the account; the original local copy is not deleted.
- This prototype has no email verification or password-recovery flow. Do not use a password you use elsewhere.

## Large catalogs and search

Image analysis defaults to **Fast**: MobileCLIP-S0 compares the full image and one centered crop, skipping the large footwear detector. Choose **Detailed footwear detection** in the signed-in panel to analyze more crops, run OWL-ViT, and add footwear-focused embeddings; changing modes re-analyzes the current query image. The setting applies to searches and new catalog photos; older browser-catalog photos only need analysis if they do not already have a compatible embedding. WebGPU accelerates MobileCLIP when available; the app falls back to CPU inference when it is not.

Each finished catalog photo is uploaded and saved immediately; paginated UI pages only load a small number of product cards. If a folder import is interrupted, select the same folder again: already-uploaded files are checked by SHA-256 and skipped.

**Analysis is still performed by the browser.** Importing 10,000 images can take hours, consumes browser memory/CPU, and requires the tab to remain open while the current batch is processed. Completed images remain saved on the server if the tab closes, and can be skipped on a later re-import. This version does not yet include a server-side inference worker, a durable server job queue, vector-index acceleration, or unattended background processing.

Search is performed by the API using an exact NumPy cosine-similarity scan over the account's embeddings, then returns one page of ranked results. This is appropriate for the current 10,000-product prototype target, but larger catalogs or concurrent production traffic should move to a dedicated vector index and background indexing workers.

The first Fast analysis in a browser downloads the Transformers.js runtime and MobileCLIP-S0 weights (about 12 MB) from public CDNs. The optional Detailed mode also downloads OWL-ViT (about 127 MB) and runs it for each image, so it is much slower. Downloads are cached by that browser. Images are not uploaded to the model providers. Model details and licenses: [MobileCLIP-S0](https://huggingface.co/Xenova/mobileclip_s0) ([license](https://huggingface.co/Xenova/mobileclip_s0/raw/main/LICENSE)) and [OWL-ViT](https://huggingface.co/Xenova/owlvit-base-patch32) ([Apache-2.0](https://huggingface.co/google/owlvit-base-patch32/raw/main/LICENSE)).

## Retrieval benchmark

Open [http://localhost:8000/benchmark.html](http://localhost:8000/benchmark.html) or select **Evaluate retrieval** in the app footer. The benchmark runs entirely in the browser and does not require an account. Select one folder containing `benchmark.json`, a `catalog` directory, and a separate `queries` directory:

```text
footwear-benchmark/
  benchmark.json
  catalog/
    shoe-001-front.jpg
    sandal-002-side.jpg
  queries/
    shoe-001-outdoor.jpg
```

Manifest example:

```json
{
  "version": 1,
  "catalog": [
    { "id": "shoe-001", "image": "catalog/shoe-001-front.jpg", "additional_images": ["catalog/shoe-001-side.jpg"], "category": "shoes" },
    { "id": "sandal-002", "image": "catalog/sandal-002-side.jpg", "category": "sandals" }
  ],
  "queries": [
    {
      "id": "query-001",
      "image": "queries/shoe-001-outdoor.jpg",
      "relevant_product_ids": ["shoe-001"],
      "category": "shoes"
    }
  ]
}
```

Catalog IDs identify the ground-truth products, not individual filenames. Each product requires one `image` and may include additional reference views in an `additional_images` array. The product score is the best similarity across its reference views. Use different, held-out query photos and list every correct product ID in `relevant_product_ids`; never reuse a reference image as a query. Collect varied views, backgrounds, and lighting, use product identities rather than filenames as labels, and ensure the dataset is legally usable. The browser accepts at most 1,000 catalog images total and 500 queries per run.

The report compares Fast and Detailed analysis, with macro Recall@1/5/10, mAP@10, query-latency median/p95, catalog-image latency, and total per-mode time. Recall@K is the fraction of labeled relevant products retrieved by rank K. AP@10 averages precision at relevant ranks through rank 10, divided by `min(number of relevant products, 10)`; mAP@10 is the mean over queries. Query latency includes query-image embedding and browser-side ranking after model initialization. First-use model setup/download is counted in the first catalog-image latency and total mode time. Save the downloadable JSON report alongside notes about the device, dataset composition, and any labeling limitations. Small or biased datasets are useful for iteration but do not establish general model quality.

### Initial ABO Fast baseline

The first measured run used 50 ABO products (26 shoes and 24 sandals) and 100 held-out alternate-view queries (two per product). Queries were ranked only against catalog products in their labeled category. On this Windows machine in the VS Code-integrated Chromium browser, with WebGPU available, 12 reported logical processors, and 16 GB reported device memory, Fast mode measured:

| Recall@1 | Recall@5 | Recall@10 | mAP@10 | Median query | p95 query | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0% | 38.0% | 60.0% | 22.7% | 1,921 ms | 2,017 ms | 318,670 ms |

This is a preliminary seller-image baseline, not a real-world performance claim: queries are alternate seller listing photos rather than user camera photos, category labels are heuristic with one visually reviewed pump correction, and the run covers one device/browser. The complete JSON report is saved locally as `benchmark-data/abo-footwear/fast-results.json`; the dataset folder is Git-ignored.

### Paired real-photo reference-view experiment

Fast mode was evaluated on the same 23 held-out queries for 14 products, first with one catalog image per product and then with 11 additional reference views. Recall@5 increased from 43.5% to 60.9%, while Recall@1 changed from 13.0% to 8.7%, Recall@10 remained 82.6%, and mAP@10 changed from 27.2% to 26.9%. Median query latency was similar (2,038 ms vs. 2,031 ms); embedding the 25-image catalog took longer than the 14-image catalog. This small, device-specific comparison is inconclusive about general retrieval quality. The datasets and JSON reports are saved locally in the Git-ignored `benchmark-data/multi-view-split-single-reference/` and `benchmark-data/multi-view-split-multiple-references/` folders.

### Prepare an Amazon Berkeley Objects footwear subset

The optional `prepare_abo_benchmark.py` script prepares a reproducible local subset of Amazon Berkeley Objects (ABO): the catalog gets a product's main image and each held-out query gets a distinct alternate image for that same product. The default is 25 shoe products and 25 sandal products, with two held-out queries per product. It downloads only the small-image views needed for that subset; it does not fetch the full image archive. The first run downloads about 110 MB of metadata plus the selected images.

ABO is published under CC BY 4.0. The output folder retains the original license, README, image-source URLs, product IDs, and required attribution. The script selects the sandals category using listing title/style/taxonomy text, with explicit pumps assigned to shoes; **inspect the category labels and images before treating the metrics as a formal evaluation**. These seller catalog alternate views are not a substitute for real user photographs, and the subset does not establish real-world model quality. Dataset use does not imply endorsement by Amazon.com or UC Berkeley.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe prepare_abo_benchmark.py
```

The generated `benchmark-data/abo-footwear` folder is local and Git-ignored. Open the benchmark page, choose that folder, and start with **Fast only**. Detailed mode analyzes every image with the large detector and can take substantially longer. To make a smaller first run, pass for example `--products-per-category 5 --queries-per-product 1`. A new output path is required if the default folder already exists.

## Hosting notes

This repository is a single-instance prototype, not a production-ready public service. Use HTTPS and a persistent private disk for `PAIRUP_DATA_DIR`; set `PAIRUP_SECURE_COOKIE=1`. Keep the data directory out of the web root and do not run multiple workers against this SQLite database. Before accepting public users, add email verification and password recovery, account deletion/export, abuse and rate limits, backups, monitoring, and a deployment security review. A horizontally scaled service should use managed storage/database and a dedicated vector service.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

With the server running, open [http://localhost:8000/visual-search.test.html](http://localhost:8000/visual-search.test.html) for browser-side embedding and ranking tests. Those tests don't exercise account isolation or the API.

See [PROJECT_GOALS.md](./PROJECT_GOALS.md) for milestones and known next steps.
