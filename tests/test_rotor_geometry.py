import unittest
from unittest.mock import patch

from pycopter.pycopter import Rotor


class FakePolar:
    def __init__(self, *_args, **_kwargs):
        pass


class RotorGeometryTests(unittest.TestCase):
    def test_rotor_initialization_derives_basic_geometry(self):
        with patch("pycopter.pycopter.Polar", FakePolar):
            rotor = Rotor(
                airfoil="naca23012",
                num_blades=5,
                chord=0.53,
                rotor_diameter=21.29,
                tip_speed_mach=0.624,
                washout=-8.0,
                rotor_root_cutout=0.02,
                new_polar=False,
            )

        self.assertAlmostEqual(rotor.r, 10.645)
        self.assertAlmostEqual(rotor.tip_speed, 214.032)
        self.assertAlmostEqual(rotor.solidity, 0.079223, places=6)
        self.assertFalse(rotor.is_hovered)

    def test_ground_effect_increases_thrust_near_ground(self):
        with patch("pycopter.pycopter.Polar", FakePolar):
            rotor = Rotor(rotor_diameter=10.0, new_polar=False)

        self.assertGreater(rotor.ige(thrust=1000.0, rotor_height=3.0), 1000.0)


if __name__ == "__main__":
    unittest.main()
