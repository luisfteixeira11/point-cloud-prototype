import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from pc import create_point_cloud, normalize_depth


class NormalizeDepthTests(unittest.TestCase):
    def test_scales_values_to_zero_one(self) -> None:
        depth = np.array([[2, 4], [6, 8]], dtype=np.float32)

        normalized = normalize_depth(depth)

        np.testing.assert_allclose(normalized, [[0, 1 / 3], [2 / 3, 1]])

    def test_constant_depth_returns_zeros(self) -> None:
        normalized = normalize_depth(np.full((2, 3), 5, dtype=np.float32))

        np.testing.assert_array_equal(normalized, np.zeros((2, 3)))

    def test_rejects_non_finite_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "não finitos"):
            normalize_depth(np.array([[np.nan]], dtype=np.float32))


class CreatePointCloudTests(unittest.TestCase):
    def test_builds_matching_points_and_colors(self) -> None:
        image = Image.new("RGB", (4, 2), color=(255, 0, 0))
        depth = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32)

        point_cloud = create_point_cloud(image, depth, step=2)

        self.assertEqual(len(point_cloud.points), 2)
        self.assertEqual(len(point_cloud.colors), 2)
        np.testing.assert_allclose(
            np.asarray(point_cloud.colors),
            [[1, 0, 0], [1, 0, 0]],
        )

    def test_rejects_invalid_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "maior que zero"):
            create_point_cloud(Image.new("RGB", (1, 1)), np.ones((1, 1)), step=0)

    def test_point_cloud_can_be_written_and_read_as_ply(self) -> None:
        point_cloud = create_point_cloud(
            Image.new("RGB", (2, 2), color=(0, 255, 0)),
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            step=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "cloud.ply")

            self.assertTrue(o3d.io.write_point_cloud(output_path, point_cloud))
            loaded_cloud = o3d.io.read_point_cloud(output_path)

        self.assertEqual(len(loaded_cloud.points), 4)
        self.assertEqual(len(loaded_cloud.colors), 4)


if __name__ == "__main__":
    unittest.main()
