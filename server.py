import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PAIRUP_DATA_DIR", ROOT / "pairup-data")).resolve()
IMAGE_DIR = DATA_DIR / "images"
DATABASE_PATH = DATA_DIR / "pairup.sqlite3"
COOKIE_NAME = "pairup_session"
SESSION_SECONDS = 60 * 60 * 24 * 14
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_PRODUCTS_PER_USER = 10_000
MAX_ACCOUNT_STORAGE_BYTES = 2 * 1024 * 1024 * 1024
PAGE_SIZE_DEFAULT = 32
PAGE_SIZE_MAX = 100
PASSWORD_ITERATIONS = 600_000
ALLOWED_IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
    "GIF": ("image/gif", ".gif"),
}
STATIC_FILES = {
    "index.html",
    "styles.css",
    "app.js",
    "image-embeddings.js",
    "visual-search.js",
    "catalog-store.js",
    "visual-search.test.html",
    "benchmark.html",
    "benchmark.css",
    "benchmark.js",
}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)


class SearchRequest(BaseModel):
    embedding: dict | list
    category: str = "all"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=PAGE_SIZE_DEFAULT, ge=1, le=PAGE_SIZE_MAX)


def connect_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with connect_database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_user_id ON sessions(user_id);
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                content_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('shoes', 'sandals')),
                image_path TEXT NOT NULL,
                image_type TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(user_id, content_hash)
            );
            CREATE INDEX IF NOT EXISTS products_user_created
                ON products(user_id, created_at DESC, id);
            CREATE INDEX IF NOT EXISTS products_user_category
                ON products(user_id, category);
            """
        )


def normalize_email(email):
    return email.strip().lower()


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )


def validate_embedding(value):
    if isinstance(value, list):
        value = {"version": 1, "vectors": [value], "objectVectors": []}
    if not isinstance(value, dict):
        raise ValueError("Expected an image embedding object.")

    version = value.get("version")
    if version not in (1, 2, 3):
        raise ValueError("Unsupported image embedding version.")
    vectors = value.get("vectors") if version in (2, 3) else value.get("vectors")
    object_vectors = value.get("objectVectors", []) if version == 3 else []
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("Image embedding must contain at least one vector.")
    if not isinstance(object_vectors, list) or len(object_vectors) > 2:
        raise ValueError("Image embedding has an invalid number of detected-object vectors.")
    if len(vectors) > 8:
        raise ValueError("Image embedding contains too many image crops.")

    normalized = []
    dimensions = None
    for vector in [*vectors, *object_vectors]:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 4096:
            raise ValueError("Image embedding vector has an invalid size.")
        values = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Image embedding values must be finite.")
        if dimensions is None:
            dimensions = len(values)
        elif len(values) != dimensions:
            raise ValueError("Image embedding vectors must have matching dimensions.")
        magnitude = float(np.linalg.norm(values))
        if magnitude <= 0:
            raise ValueError("Image embedding must not be all zero.")
        normalized.append((values / magnitude).tolist())

    normalized_vectors = normalized[: len(vectors)]
    normalized_objects = normalized[len(vectors) :]
    if version == 1:
        return normalized_vectors[0]
    if version == 2:
        return {"version": 2, "vectors": normalized_vectors}
    return {
        "version": 3,
        "vectors": normalized_vectors,
        "objectVectors": normalized_objects,
    }


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding vectors have different dimensions.")
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def get_vectors(embedding):
    if isinstance(embedding, list):
        return [embedding]
    if embedding.get("version") in (2, 3):
        return embedding["vectors"]
    raise ValueError("Unsupported image embedding.")


def get_object_vectors(embedding):
    if isinstance(embedding, dict) and embedding.get("version") == 3:
        return embedding.get("objectVectors", [])
    return []


def embedding_similarity(query, product):
    query_objects = get_object_vectors(query)
    product_objects = get_object_vectors(product)
    if query_objects and product_objects:
        return max(cosine(left, right) for left in query_objects for right in product_objects)

    query_vectors = get_vectors(query)
    product_vectors = get_vectors(product)
    global_score = cosine(query_vectors[0], product_vectors[0])
    crop_scores = [
        cosine(left, right)
        for left_index, left in enumerate(query_vectors)
        for right_index, right in enumerate(product_vectors)
        if left_index or right_index
    ]
    if not crop_scores:
        return global_score
    return float(np.clip(0.35 * global_score + 0.65 * max(crop_scores), -1.0, 1.0))


def encode_vectors(value):
    return json.dumps(validate_embedding(value), separators=(",", ":"), allow_nan=False)


def embedding_from_json(value):
    return validate_embedding(json.loads(value))


def product_summary(row, score=None):
    product = {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "imageUrl": f"/api/products/{row['id']}/image",
    }
    if score is not None:
        product["score"] = score
    return product


def create_session(connection, user_id):
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_SECONDS
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    connection.execute(
        "INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (token_hash, user_id, expires_at),
    )
    return token, expires_at


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in to access your private catalog.")
    token_hash = hashlib.sha256(token.encode("ascii", errors="ignore")).hexdigest()
    with connect_database() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.email
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """,
            (token_hash, int(time.time())),
        ).fetchone()
        if not row:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
    return {"id": row["id"], "email": row["email"]}


def enforce_same_origin(request: Request):
    origin = request.headers.get("origin")
    if not origin:
        return
    host = request.headers.get("host", "")
    if origin.split("://", 1)[-1].rstrip("/") != host:
        raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed.")


def set_session_cookie(response, token, expires_at):
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=max(0, expires_at - int(time.time())),
        httponly=True,
        secure=os.environ.get("PAIRUP_SECURE_COOKIE", "0") == "1",
        samesite="lax",
        path="/",
    )


initialize_database()
app = FastAPI(title="PairUp Private Catalog API", docs_url=None, redoc_url=None)


@app.post("/api/auth/register")
def register(payload: Credentials, request: Request, response: Response):
    enforce_same_origin(request)
    email = normalize_email(payload.email)
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    user_id = str(uuid.uuid4())
    salt = secrets.token_bytes(16)
    password_hash = hash_password(payload.password, salt)
    try:
        with connect_database() as connection:
            connection.execute(
                "INSERT INTO users(id, email, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, salt, password_hash, int(time.time())),
            )
            token, expires_at = create_session(connection, user_id)
    except sqlite3.IntegrityError as error:
        raise HTTPException(status_code=409, detail="An account with that email already exists.") from error
    set_session_cookie(response, token, expires_at)
    return {"email": email}


@app.post("/api/auth/login")
def login(payload: Credentials, request: Request, response: Response):
    enforce_same_origin(request)
    email = normalize_email(payload.email)
    with connect_database() as connection:
        user = connection.execute(
            "SELECT id, email, password_salt, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        candidate = hash_password(payload.password, user["password_salt"] if user else bytes(16))
        if not user or not hmac.compare_digest(candidate, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        token, expires_at = create_session(connection, user["id"])
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
    set_session_cookie(response, token, expires_at)
    return {"email": user["email"]}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: Annotated[dict, Depends(get_current_user)]):
    enforce_same_origin(request)
    token = request.cookies.get(COOKIE_NAME, "")
    token_hash = hashlib.sha256(token.encode("ascii", errors="ignore")).hexdigest()
    with connect_database() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax")
    return {"ok": True}


@app.get("/api/auth/me")
def current_account(user: Annotated[dict, Depends(get_current_user)]):
    return user


@app.get("/api/products")
def list_products(
    page: int = 1,
    page_size: int = PAGE_SIZE_DEFAULT,
    user: Annotated[dict, Depends(get_current_user)] = None,
):
    if page < 1 or not 1 <= page_size <= PAGE_SIZE_MAX:
        raise HTTPException(status_code=422, detail="Invalid pagination values.")
    offset = (page - 1) * page_size
    with connect_database() as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS count FROM products WHERE user_id = ?", (user["id"],)
        ).fetchone()["count"]
        rows = connection.execute(
            """
            SELECT id, name, category FROM products
            WHERE user_id = ?
            ORDER BY created_at DESC, id
            LIMIT ? OFFSET ?
            """,
            (user["id"], page_size, offset),
        ).fetchall()
    return {
        "products": [product_summary(row) for row in rows],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


@app.post("/api/products")
async def add_product(
    image: Annotated[UploadFile, File()],
    name: Annotated[str, Form(max_length=120)],
    category: Annotated[str, Form()],
    embedding: Annotated[str, Form()],
    user: Annotated[dict, Depends(get_current_user)],
):
    if category not in ("shoes", "sandals"):
        raise HTTPException(status_code=422, detail="Category must be shoes or sandals.")
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="Product name must not be empty.")
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image files must be 15 MB or smaller.")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as decoded:
                image_format = decoded.format
                decoded.verify()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise HTTPException(status_code=415, detail="Upload a valid JPG, PNG, GIF, or WEBP image.") from error
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(status_code=415, detail="Upload a valid JPG, PNG, GIF, or WEBP image.")
    image_type, extension = ALLOWED_IMAGE_FORMATS[image_format]
    try:
        parsed_embedding = encode_vectors(json.loads(embedding))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid image embedding: {error}") from error

    content_hash = hashlib.sha256(content).hexdigest()
    product_id = str(uuid.uuid4())
    relative_path = Path(user["id"]) / f"{product_id}{extension}"
    image_path = IMAGE_DIR / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id, name, category FROM products WHERE user_id = ? AND content_hash = ?",
            (user["id"], content_hash),
        ).fetchone()
        if existing:
            return product_summary(existing)
        usage = connection.execute(
            """
            SELECT COUNT(*) AS product_count, COALESCE(SUM(byte_size), 0) AS storage_bytes
            FROM products WHERE user_id = ?
            """,
            (user["id"],),
        ).fetchone()
        if usage["product_count"] >= MAX_PRODUCTS_PER_USER:
            raise HTTPException(status_code=409, detail="This account has reached the 10,000-product limit.")
        if usage["storage_bytes"] + len(content) > MAX_ACCOUNT_STORAGE_BYTES:
            raise HTTPException(status_code=409, detail="This account has reached its 2 GB image-storage limit.")
        try:
            image_path.write_bytes(content)
            connection.execute(
                """
                INSERT INTO products(
                    id, user_id, content_hash, name, category,
                    image_path, image_type, byte_size, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    user["id"],
                    content_hash,
                    clean_name,
                    category,
                    str(relative_path),
                    image_type,
                    len(content),
                    parsed_embedding,
                    int(time.time()),
                ),
            )
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
    return {
        "id": product_id,
        "name": clean_name,
        "category": category,
        "imageUrl": f"/api/products/{product_id}/image",
    }


@app.get("/api/products/check")
def check_product(content_hash: str, user: Annotated[dict, Depends(get_current_user)]):
    if len(content_hash) != 64 or any(character not in "0123456789abcdef" for character in content_hash.lower()):
        raise HTTPException(status_code=422, detail="Invalid image checksum.")
    with connect_database() as connection:
        exists = connection.execute(
            "SELECT 1 FROM products WHERE user_id = ? AND content_hash = ?",
            (user["id"], content_hash.lower()),
        ).fetchone()
    return {"exists": exists is not None}


@app.get("/api/products/{product_id}/image")
def get_product_image(
    product_id: str,
    user: Annotated[dict, Depends(get_current_user)],
):
    with connect_database() as connection:
        row = connection.execute(
            "SELECT image_path, image_type FROM products WHERE id = ? AND user_id = ?",
            (product_id, user["id"]),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Product image not found.")
    path = IMAGE_DIR / row["image_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Product image is missing from storage.")
    return FileResponse(path, media_type=row["image_type"], headers={"Cache-Control": "private, no-store"})


@app.delete("/api/products/{product_id}")
def remove_product(
    product_id: str,
    user: Annotated[dict, Depends(get_current_user)],
):
    with connect_database() as connection:
        row = connection.execute(
            "SELECT image_path FROM products WHERE id = ? AND user_id = ?",
            (product_id, user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found.")
        connection.execute("DELETE FROM products WHERE id = ? AND user_id = ?", (product_id, user["id"]))
    (IMAGE_DIR / row["image_path"]).unlink(missing_ok=True)
    return {"ok": True}


@app.delete("/api/products")
def clear_products(user: Annotated[dict, Depends(get_current_user)]):
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT image_path FROM products WHERE user_id = ?", (user["id"],)
        ).fetchall()
        connection.execute("DELETE FROM products WHERE user_id = ?", (user["id"],))
    for row in rows:
        (IMAGE_DIR / row["image_path"]).unlink(missing_ok=True)
    return {"deleted": len(rows)}


@app.post("/api/search")
def search_products(
    payload: SearchRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    if payload.category not in ("all", "shoes", "sandals"):
        raise HTTPException(status_code=422, detail="Invalid product category.")
    try:
        query = validate_embedding(payload.embedding)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid query embedding: {error}") from error
    where = "user_id = ?"
    parameters = [user["id"]]
    if payload.category != "all":
        where += " AND category = ?"
        parameters.append(payload.category)
    with connect_database() as connection:
        rows = connection.execute(
            f"""
            SELECT id, name, category, embedding_json
            FROM products WHERE {where}
            """,
            parameters,
        ).fetchall()
    ranked = []
    for row in rows:
        product_embedding = embedding_from_json(row["embedding_json"])
        score = embedding_similarity(query, product_embedding)
        ranked.append((score, row))
    ranked.sort(key=lambda result: result[0], reverse=True)
    total = len(ranked)
    start = (payload.page - 1) * payload.page_size
    page_rows = ranked[start : start + payload.page_size]
    return {
        "products": [
            {**product_summary(row, score), "score": round(score, 6)}
            for score, row in page_rows
        ],
        "total": total,
        "page": payload.page,
        "pageSize": payload.page_size,
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/{asset_name}")
def static_asset(asset_name: str):
    if asset_name not in STATIC_FILES:
        raise HTTPException(status_code=404, detail="Page not found.")
    return FileResponse(ROOT / asset_name)
