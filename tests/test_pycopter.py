#!/usr/bin/env python

"""Tests for `pycopter` package."""


import unittest
import numpy as np
import sys
import os
import json

from pycopter import Rotor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestRotor(unittest.TestCase):
    def setUp(self):
        self.num_blades = 6
        self.chord = 0.025
        self.rotor_diameter = 0.6
        self.rpm = 2500
        self.cl = 1.1
        self.cd = 0.04
        self.tip_loss = 0.95
        self.rotor_disk_hole = 0.1
        self.density = 1.293
        self.rotor_radius = self.rotor_diameter / 2
        self.rpm_in_hz = self.rpm * np.pi / 30
        self.tip_speed = self.rpm_in_hz * self.rotor_radius
        self.rotor_disk_area = np.pi * self.rotor_radius**2 - np.pi * (self.rotor_radius * self.rotor_disk_hole)**2
        self.test_rotor = Rotor(self.num_blades,self.chord,self.rotor_diameter,self.rpm,self.cl,self.cd,self.tip_loss,self.rotor_disk_hole)

        package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "package.json"))
        try:
            with open(package_path, "r") as f:
                self.package_data = json.load(f)
        except:
            self.fail(f"Required file not found at: \n{package_path}")
        # TODO: Learn how to test and whether we need package.json.


    def test_calculate_thrust(self):
        expected_thrust = self.density * self.num_blades * self.chord * (self.rotor_diameter*0.5)**3 * self.rpm_in_hz**2 * self.cl / 6
        self.assertAlmostEqual(self.test_rotor.get_thrust(self.density), expected_thrust)

    def test_calculate_horsepower(self, thrust):

        watts = thrust**1.5 / (self.tip_loss * np.sqrt(2*self.density*self.rotor_disk_area)) + self.density*self.num_blades*self.chord*(self.rotor_radius**4) * (self.rpm_in_hz**3) * self.cd / 8
        expected_horsepower = watts * 0.00134102209
        self.assertAlmostEqual(self.test_rotor.get_horsepower(self.density), expected_horsepower)



# class TestPycopter(unittest.TestCase):
#     """Tests for `pycopter` package."""

#     def setUp(self):
#         """Set up test fixtures, if any."""

#     def tearDown(self):
#         """Tear down test fixtures, if any."""

#     def test_000_something(self):
#         """Test something."""
#         print("testing something...")

# testobject = TestPycopter
# testobject.test_000_something()


if __name__ =="__main__":
    unittest.main()