import unittest

import numpy as np

from mask_extraction.dataset_tools.build_soft_labels import (
    MASK_SIZE,
    _infer_method,
    _weighted_binary_mask_average,
    _weighted_dense_support_average,
)


class SoftLabelUtilityTests(unittest.TestCase):
    def test_infer_method_groups_mask_variants(self) -> None:
        self.assertEqual(_infer_method("grasp"), "grasp")
        self.assertEqual(_infer_method("scope_64"), "scope")
        self.assertEqual(_infer_method("vispruner_128"), "vispruner")

    def test_weighted_binary_average(self) -> None:
        left = np.zeros(MASK_SIZE, dtype=bool)
        right = np.zeros(MASK_SIZE, dtype=bool)
        left[0] = True
        right[1] = True

        fused = _weighted_binary_mask_average(
            [(1.0, left), (3.0, right)],
            epsilon=0.0,
        )

        self.assertAlmostEqual(float(fused[0]), 0.25)
        self.assertAlmostEqual(float(fused[1]), 0.75)
        self.assertEqual(int(np.count_nonzero(fused)), 2)

    def test_dense_support_is_clipped(self) -> None:
        support = np.full(MASK_SIZE, 2.0, dtype=np.float32)
        fused = _weighted_dense_support_average([(1.0, support)], epsilon=0.0)
        np.testing.assert_array_equal(fused, np.ones(MASK_SIZE, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
