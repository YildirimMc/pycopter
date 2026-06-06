import unittest

import numpy as np

from pycopter.utils import find_interval_idx


class TestUtils(unittest.TestCase):
    def test_find_interval_idx_uses_adjacent_rows_only(self):
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
            ]
        )

        self.assertEqual(1, find_interval_idx(data, 1.5))

    def test_find_interval_idx_raises_when_value_is_outside_table(self):
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 10.0],
            ]
        )

        with self.assertRaises(ValueError):
            find_interval_idx(data, 2.0)


if __name__ == "__main__":
    unittest.main()
