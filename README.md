# PairUp — Footwear Visual Search

A small, browser-based prototype for finding visually similar shoes and sandals from a collection of photos.

## Run locally

Camera access requires a secure context such as `localhost`. With Python 3 installed, run this from the project folder:

```powershell
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000). No package installation or build step is required.

## Try it

1. Choose whether the catalog photos are shoes or sandals. Select multiple images with **Add product photos**, or choose **Add a photo folder** to import images from a folder (including supported images in its subfolders). Every photo in an import batch gets the selected category. Folder imports are analyzed and saved one photo at a time; unsupported files are skipped, and large folders take longer.
2. Upload a query photo, or allow camera access and capture one.
3. Choose a category and select **Find similar pairs**.
4. Remove individual catalog entries or clear the saved collection when done.

Images are processed locally in the browser. Catalog photos and their embeddings are saved in IndexedDB on this device, so they remain in the catalog after refresh or reopening PairUp in the same browser. Use **Clear collection** to remove them. Browser storage has a device/browser-specific limit; very large imports may not fit. If a photo cannot be processed or saved, the app reports that it was skipped. Browser storage can be cleared by browser settings, and the collection is not synced to other devices.

## How matching works

The app uses the pretrained MobileCLIP-S0 image encoder (`Xenova/mobileclip_s0`) through Transformers.js to turn each image into several 512-value embeddings: one for the full photo and others for overlapping square crops. Ranking blends the whole-image cosine similarity with the best crop-to-crop similarity, helping match a product when the number or placement of shoes changes between photos. Scores are ranking signals, not probabilities or guarantees of exact product identity.

The first image can take longer while the model runtime and quantized model weights (about 12 MB) download from jsDelivr and Hugging Face. They are cached by the browser for later use. An internet connection is needed for this first download. Inference and image handling happen in your browser: product and query images are not sent to those providers. If the model cannot load, the app reports an error instead of silently switching to the old color/edge heuristic. The model is distributed under Apple's MobileCLIP license; review its terms before using it commercially.

Model details: [MobileCLIP-S0 on Hugging Face](https://huggingface.co/Xenova/mobileclip_s0) and its [license](https://huggingface.co/Xenova/mobileclip_s0/raw/main/LICENSE).

This improves semantic visual matching, but it does not fetch products from the web and does not guarantee footwear-specific retrieval quality. The next quality milestone is to evaluate Recall@K and latency on an appropriately licensed set of real footwear photos, then compare models.

## Tests

Existing catalog photos are re-embedded the first time they are included in a search after this update. The app saves the new vectors one product at a time; model or storage errors are reported rather than silently reverting to the old scores.

Start the local server as above, then open [http://localhost:8000/visual-search.test.html](http://localhost:8000/visual-search.test.html). The page reports checks for embedding normalization, crop-aware ranking, legacy-vector compatibility, category filtering, and invalid inputs. These unit tests do not download or validate the pretrained model weights or IndexedDB persistence.

See [PROJECT_GOALS.md](./PROJECT_GOALS.md) for the product checkpoints.
