import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import server
from server import embedding_similarity, validate_embedding


class EmbeddingTests(unittest.TestCase):
    def test_detected_object_vectors_take_priority(self):
        query = {
            "version": 3,
            "vectors": [[1.0, 0.0]],
            "objectVectors": [[0.0, 1.0]],
        }
        same_item = {
            "version": 3,
            "vectors": [[1.0, 0.0]],
            "objectVectors": [[0.0, 1.0]],
        }
        different_item = {
            "version": 3,
            "vectors": [[1.0, 0.0]],
            "objectVectors": [[1.0, 0.0]],
        }
        self.assertGreater(
            embedding_similarity(query, same_item),
            embedding_similarity(query, different_item),
        )
        self.assertAlmostEqual(embedding_similarity(query, same_item), 1.0)

    def test_images_without_detections_keep_crop_aware_scoring(self):
        query = {"version": 3, "vectors": [[1.0, 0.0], [0.0, 1.0]], "objectVectors": []}
        product = {"version": 3, "vectors": [[0.0, 1.0], [1.0, 0.0]], "objectVectors": []}
        self.assertAlmostEqual(embedding_similarity(query, product), 0.65)

    def test_embedding_vectors_are_normalized(self):
        embedding = validate_embedding(
            {"version": 3, "vectors": [[3.0, 4.0]], "objectVectors": [[4.0, 3.0]]}
        )
        self.assertAlmostEqual(float(np.linalg.norm(embedding["vectors"][0])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(embedding["objectVectors"][0])), 1.0)

    def test_invalid_embedding_values_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_embedding({"version": 3, "vectors": [[float("nan")]], "objectVectors": []})
        with self.assertRaises(ValueError):
            validate_embedding({"version": 4, "vectors": [[1.0]], "objectVectors": []})


class PrivateProductImageTests(unittest.TestCase):
    def test_product_images_are_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "product.png"
            image_path.write_bytes(b"image")

            connection = MagicMock()
            connection.__enter__.return_value = connection
            connection.execute.return_value.fetchone.return_value = {
                "image_path": image_path.name,
                "image_type": "image/png",
            }

            with (
                patch.object(server, "connect_database", return_value=connection),
                patch.object(server, "IMAGE_DIR", Path(temporary_directory)),
            ):
                response = server.get_product_image("product-id", {"id": "owner-id"})

        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(
            connection.execute.call_args.args[1],
            ("product-id", "owner-id"),
        )


if __name__ == "__main__":
    unittest.main()
