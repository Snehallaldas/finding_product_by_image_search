import unittest

from prepare_abo_benchmark import category_for_listing, select_image_ids


class AmazonBerkeleyObjectsBenchmarkTests(unittest.TestCase):
    def test_sandal_category_uses_listing_text_and_taxonomy(self):
        self.assertEqual(
            category_for_listing(
                "Women's open toe sandals",
                "block heel",
                ["/Categories/Women/Shoes/Sandals"],
            ),
            "sandals",
        )

    def test_non_sandal_footwear_defaults_to_shoes(self):
        self.assertEqual(
            category_for_listing(
                "Men's running shoe",
                "trail runner",
                ["/Categories/Men/Athletic Shoes"],
            ),
            "shoes",
        )

    def test_pumps_are_shoes_even_when_taxonomy_says_sandals(self):
        self.assertEqual(
            category_for_listing(
                "Women's closed-toe pumps",
                "Patent Leather Pump",
                ["/Categories/Women/Shoes/Fashion Sandals"],
            ),
            "shoes",
        )

    def test_query_images_exclude_catalog_image(self):
        candidate = {
            "main_image_id": "main",
            "alternate_image_ids": ["main", "view-1", "view-2"],
        }
        self.assertEqual(
            select_image_ids(candidate, 2),
            ["main", "view-1", "view-2"],
        )


if __name__ == "__main__":
    unittest.main()
