import argparse
import json
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DATASET_ID = "hyper3labs/amazon-berkeley-objects"
DATASET_REVISION = "ae2befec6225c0e9151112863f48918aa8a50505"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}"
DATASET_RAW = f"{DATASET_PAGE}/resolve/{DATASET_REVISION}"
OFFICIAL_DATASET_PAGE = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
LICENSE_URL = (
    f"{DATASET_PAGE}/raw/{DATASET_REVISION}/ORIGINAL_LICENSE-CC-BY-4.0.txt"
)
README_URL = f"{DATASET_PAGE}/raw/{DATASET_REVISION}/ORIGINAL_README.md"
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 4
IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._+-]+$")
SANDAL_PATTERN = re.compile(
    r"sandal|sandale|sandalia|sandália|sandalen|sandali|flip[\s-]?flop|chancla|tongs",
    re.IGNORECASE,
)
PUMP_PATTERN = re.compile(r"\bpumps?\b", re.IGNORECASE)


def main():
    args = parse_args()
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "The optional Parquet reader is missing. Install it with "
            '".\\.venv\\Scripts\\python.exe -m pip install -r requirements-benchmark.txt".'
        ) from error

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"{output} already exists. Choose a new --output folder.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pairup-abo-stage-", dir=output.parent
    ) as staging_directory:
        staging = Path(staging_directory) / output.name
        (staging / "catalog").mkdir(parents=True)
        (staging / "queries").mkdir()
        with tempfile.TemporaryDirectory(prefix="pairup-abo-metadata-") as temporary:
            metadata = Path(temporary)
            image_rows = download_parquet(
                metadata,
                "images.parquet",
                f"{DATASET_RAW}/data/images/images.parquet",
            )
            image_urls = {
                row["image_id"]: row["small_image_url"]
                for row in image_rows
                if row.get("image_id") and row.get("small_image_url")
            }
            del image_rows

            candidates = read_footwear_candidates(metadata, parquet, image_urls)
            selected = {
                category: choose_candidates(
                    candidates[category],
                    args.products_per_category,
                    args.queries_per_product,
                    args.seed,
                )
                for category in ("shoes", "sandals")
            }
            manifest, provenance = download_benchmark(
                staging,
                selected,
                image_urls,
                args.queries_per_product,
            )

        (staging / "benchmark.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "source-metadata.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        download_text(LICENSE_URL, staging / "LICENSE-CC-BY-4.0.txt")
        download_text(README_URL, staging / "ABO-README.md")
        (staging / "DATASET_ATTRIBUTION.txt").write_text(
            attribution_text(manifest, args),
            encoding="utf-8",
        )
        staging.rename(output)
    print(
        f"Prepared {len(manifest['catalog'])} catalog images and "
        f"{len(manifest['queries'])} held-out queries in {output}"
    )
    print("Review benchmark.json category labels and DATASET_ATTRIBUTION.txt before citing results.")
    print(f"Open http://localhost:8000/benchmark.html and select {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a small, attributed footwear retrieval benchmark from Amazon Berkeley Objects."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-data/abo-footwear"),
        help="Output folder (default: benchmark-data/abo-footwear).",
    )
    parser.add_argument(
        "--products-per-category",
        type=int,
        default=25,
        help="Products per shoes/sandals category (default: 25).",
    )
    parser.add_argument(
        "--queries-per-product",
        type=int,
        default=2,
        help="Held-out alternate views per product (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic selection seed.")
    args = parser.parse_args()
    if not 1 <= args.products_per_category <= 250:
        parser.error("--products-per-category must be between 1 and 250.")
    if not 1 <= args.queries_per_product <= 5:
        parser.error("--queries-per-product must be between 1 and 5.")
    if args.products_per_category * 2 * (1 + args.queries_per_product) > 1000:
        parser.error("The selected dataset would exceed the benchmark's 1,000-image limit.")
    if args.products_per_category * 2 * args.queries_per_product > 500:
        parser.error("The selected dataset would exceed the benchmark's 500-query limit.")
    return args


def download_parquet(directory, filename, url):
    destination = directory / filename
    download_file(url, destination)
    import pyarrow.parquet as parquet

    return parquet.read_table(destination).to_pylist()


def download_file(url, destination):
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc == "amazon-berkeley-objects.s3.amazonaws.com":
        url = urllib.parse.urlunsplit(
            parsed._replace(netloc="amazon-berkeley-objects.s3.us-east-1.amazonaws.com")
        )
    partial_path = destination.with_name(f"{destination.name}.download")
    for attempt in range(DOWNLOAD_ATTEMPTS):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "PairUp-ABO-retrieval-benchmark/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"Download failed with HTTP {response.status}: {url}")
                total = 0
                with partial_path.open("wb") as target:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"Refusing file larger than 15 MB: {url}")
                        target.write(chunk)
            partial_path.replace(destination)
            return destination
        except urllib.error.HTTPError as error:
            partial_path.unlink(missing_ok=True)
            if error.code < 500 or attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            partial_path.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)
    return destination


def download_text(url, destination):
    download_file(url, destination)


def read_footwear_candidates(directory, parquet, image_urls):
    candidates = {"shoes": [], "sandals": []}
    seen_item_ids = set()
    for shard_index in range(16):
        filename = f"listings-{shard_index:05d}-of-00016.parquet"
        shard_url = f"{DATASET_RAW}/data/listings/{filename}"
        shard_path = directory / filename
        download_file(shard_url, shard_path)
        table = parquet.read_table(
            shard_path,
            columns=[
                "item_id",
                "title",
                "style",
                "product_type",
                "node_paths",
                "main_image_id",
                "raw_listing_json",
            ],
        )
        for row in table.to_pylist():
            item_id = row.get("item_id")
            main_id = row.get("main_image_id")
            if (
                row.get("product_type") != "SHOES"
                or not item_id
                or item_id in seen_item_ids
                or not main_id
                or main_id not in image_urls
            ):
                continue
            try:
                listing = json.loads(row.get("raw_listing_json") or "{}")
            except json.JSONDecodeError:
                continue
            alternates = [
                image_id
                for image_id in listing.get("other_image_id", [])
                if image_id != main_id and image_id in image_urls and IMAGE_ID_PATTERN.fullmatch(image_id)
            ]
            if not alternates:
                continue
            seen_item_ids.add(item_id)
            category = category_for_listing(
                row.get("title") or "",
                row.get("style") or "",
                row.get("node_paths") or [],
            )
            candidates[category].append(
                {
                    "id": item_id,
                    "title": row.get("title") or item_id,
                    "style": row.get("style") or "",
                    "node_paths": row.get("node_paths") or [],
                    "main_image_id": main_id,
                    "alternate_image_ids": list(dict.fromkeys(alternates)),
                }
            )
        print(
            f"Scanned {filename}: "
            f"{len(candidates['shoes'])} shoe candidates, "
            f"{len(candidates['sandals'])} sandal candidates"
        )
        shard_path.unlink()
    return candidates


def category_for_listing(title, style, node_paths):
    if PUMP_PATTERN.search(title) or PUMP_PATTERN.search(style):
        return "shoes"
    category_text = " ".join([title, style, *node_paths])
    return "sandals" if SANDAL_PATTERN.search(category_text) else "shoes"


def choose_candidates(candidates, limit, query_count, seed):
    eligible = [
        candidate
        for candidate in candidates
        if len(candidate["alternate_image_ids"]) >= query_count
    ]
    random.Random(f"{seed}:{candidates[0]['id'] if candidates else 'empty'}").shuffle(eligible)
    return eligible[:limit]


def download_benchmark(output, selected, image_urls, query_count):
    manifest = {"version": 1, "catalog": [], "queries": []}
    provenance = {
        "dataset": "Amazon Berkeley Objects (ABO)",
        "dataset_page": DATASET_PAGE,
        "official_dataset_page": OFFICIAL_DATASET_PAGE,
        "metadata_mirror": DATASET_PAGE,
        "metadata_revision": DATASET_REVISION,
        "image_resolution": "ABO small catalog images",
        "labeling_note": (
            "Product identity is the ABO item_id. Category is derived from product_type=SHOES; "
            "sandals are identified by sandal/flip-flop terms in listing title, style, or node paths. "
            "Listings explicitly identified as pumps in the title or style are assigned to shoes. "
            "Review category labels before using scores as a formal evaluation."
        ),
        "products": [],
    }
    used_image_ids = set()

    for category in ("shoes", "sandals"):
        completed = 0
        for candidate in selected[category]:
            image_ids = select_image_ids(candidate, query_count)
            if len(set(image_ids)) != len(image_ids) or used_image_ids.intersection(image_ids):
                continue
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate["id"])
            catalog_name = f"{safe_id}.jpg"
            query_names = [f"{safe_id}-view-{index + 1:02d}.jpg" for index in range(query_count)]
            catalog_path = output / "catalog" / catalog_name
            query_paths = [output / "queries" / name for name in query_names]
            started = time.monotonic()
            try:
                download_image(image_urls[image_ids[0]], catalog_path)
                for image_id, query_path in zip(image_ids[1:], query_paths, strict=True):
                    download_image(image_urls[image_id], query_path)
            except (OSError, RuntimeError, UnidentifiedImageError, urllib.error.URLError) as error:
                catalog_path.unlink(missing_ok=True)
                for query_path in query_paths:
                    query_path.unlink(missing_ok=True)
                print(f"Skipping unavailable product {candidate['id']}: {error}", file=sys.stderr)
                continue

            product_id = f"abo-{candidate['id']}"
            manifest["catalog"].append(
                {"id": product_id, "image": f"catalog/{catalog_name}", "category": category}
            )
            for index, query_name in enumerate(query_names):
                manifest["queries"].append(
                    {
                        "id": f"{product_id}-view-{index + 1:02d}",
                        "image": f"queries/{query_name}",
                        "relevant_product_ids": [product_id],
                        "category": category,
                    }
                )
            provenance["products"].append(
                {
                    "id": product_id,
                    "category": category,
                    "title": candidate["title"],
                    "style": candidate["style"],
                    "node_paths": candidate["node_paths"],
                    "catalog_image_id": image_ids[0],
                    "query_image_ids": image_ids[1:],
                    "catalog_source_url": image_urls[image_ids[0]],
                    "query_source_urls": [image_urls[image_id] for image_id in image_ids[1:]],
                }
            )
            used_image_ids.update(image_ids)
            completed += 1
            print(
                f"Downloaded {category} product {completed}/{len(selected[category])}: "
                f"{candidate['id']} ({time.monotonic() - started:.1f}s)"
            )
        if completed != len(selected[category]):
            raise RuntimeError(
                f"Could only prepare {completed}/{len(selected[category])} {category} products."
            )
    return manifest, provenance


def select_image_ids(candidate, query_count):
    main_id = candidate["main_image_id"]
    alternates = [
        image_id
        for image_id in candidate["alternate_image_ids"]
        if image_id != main_id
    ]
    return [main_id, *alternates[:query_count]]


def download_image(url, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        download_file(url, temporary)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(destination)
    except (OSError, UnidentifiedImageError, urllib.error.URLError):
        temporary.unlink(missing_ok=True)
        raise


def attribution_text(manifest, args):
    catalog_count = len(manifest["catalog"])
    query_count = len(manifest["queries"])
    shoes_count = sum(item["category"] == "shoes" for item in manifest["catalog"])
    sandals_count = sum(item["category"] == "sandals" for item in manifest["catalog"])
    return (
        "Dataset: Amazon Berkeley Objects (ABO), a dataset by Amazon.com and UC Berkeley.\n"
        "Source: https://amazon-berkeley-objects.s3.amazonaws.com/index.html\n"
        f"Metadata: {DATASET_PAGE} at revision {DATASET_REVISION}\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "License text: LICENSE-CC-BY-4.0.txt\n\n"
        "Required attribution from the original dataset README:\n"
        "Data, including images: Amazon.com.\n"
        "Dataset, archives, and benchmark sets: Matthieu Guillaumin (Amazon.com), "
        "Thomas Dideriksen (Amazon.com), Kenan Deng (Amazon.com), Himanshu Arora "
        "(Amazon.com), Jasmine Collins (UC Berkeley), and Jitendra Malik (UC Berkeley).\n\n"
        "Changes: selected a reproducible subset of ABO shoe listings; used each listing's "
        "small main image as a catalog image and distinct alternate listing images as held-out "
        "queries; assigned product IDs from ABO item IDs; assigned shoes/sandals categories "
        "using listing text and node-path heuristics, with explicit pumps assigned to shoes. "
        "Review those category labels before "
        "formal evaluation. The source images are otherwise unmodified.\n"
        f"Selection seed: {args.seed}\n"
        f"Products per category requested: {args.products_per_category}\n"
        f"Alternate query images per product: {args.queries_per_product}\n"
        f"Prepared catalog images: {catalog_count}\n"
        f"Prepared category counts: {shoes_count} shoes, {sandals_count} sandals\n"
        f"Prepared held-out queries: {query_count}\n"
        "This benchmark subset is not endorsed by Amazon.com or UC Berkeley.\n"
    )


if __name__ == "__main__":
    main()
