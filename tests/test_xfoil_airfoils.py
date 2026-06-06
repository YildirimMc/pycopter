import importlib.util
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch


def load_xfoil_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "pycopter" / "xfoil.py"
    spec = importlib.util.spec_from_file_location("xfoil_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.text.encode("utf-8")


class TestXfoilAirfoils(unittest.TestCase):
    def setUp(self):
        self.xfoil = load_xfoil_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_naca_airfoil_uses_direct_xfoil_command(self):
        commands, message = self.xfoil.get_airfoil_commands("NACA23012", self.repo_root)

        self.assertEqual(["naca23012"], commands)
        self.assertEqual("", message)

    def test_non_naca_airfoil_downloads_from_uiuc(self):
        dat_text = "S1223\n1.0 0.0\n0.5 0.1\n0.0 0.0\n1.0 0.0\n"

        with patch.object(self.xfoil, "urlopen", return_value=FakeResponse(dat_text)):
            commands, message = self.xfoil.get_airfoil_commands("S1223", self.repo_root)

        self.assertEqual(["load data/airfoils/s1223.dat", "pane"], commands)
        self.assertEqual("", message)
        self.assertTrue((self.repo_root / "data" / "airfoils" / "s1223.dat").exists())

    def test_missing_non_naca_airfoil_rejects_without_raising(self):
        with patch.object(self.xfoil, "urlopen", side_effect=URLError("offline")):
            commands, message = self.xfoil.get_airfoil_commands("missingfoil", self.repo_root)

        self.assertIsNone(commands)
        self.assertIn("missingfoil", message)

    def test_xfoil_default_alpha_sweep_stops_at_18_degrees(self):
        captured = {}
        polar_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
        polar_path.parent.mkdir(parents=True)

        class FakeProcess:
            def communicate(self, input, timeout):
                captured["command"] = input
                polar_path.write_text(
                    "\n" * 12
                    + "-3 0.0 0.01 0 0\n"
                    + "0 0.0 0.01 0 0\n"
                    + "18 1.0 0.05 0 0\n",
                    encoding="utf-8",
                )
                return "", ""

        runner = self.xfoil.Xfoil(new_polar=False)
        runner.repo_root = self.repo_root
        runner.output_path = polar_path
        runner.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

        with patch.object(self.xfoil.subprocess, "Popen", return_value=FakeProcess()):
            self.assertTrue(runner.simulate("naca0012", mach=0.1, reynolds=100000))

        alpha_lines = [
            line for line in captured["command"].splitlines() if line.startswith("alfa ")
        ]
        self.assertEqual("alfa 18", alpha_lines[-1])
        self.assertNotIn("alfa 19", alpha_lines)
        self.assertNotIn("alfa 20", alpha_lines)

    def test_xfoil_explicit_alpha_sweep_is_capped_at_18_degrees(self):
        captured = {}
        polar_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
        polar_path.parent.mkdir(parents=True)

        class FakeProcess:
            def communicate(self, input, timeout):
                captured["command"] = input
                polar_path.write_text(
                    "\n" * 12
                    + "-5 0.0 0.01 0 0\n"
                    + "0 0.0 0.01 0 0\n"
                    + "18 1.0 0.05 0 0\n",
                    encoding="utf-8",
                )
                return "", ""

        runner = self.xfoil.Xfoil(new_polar=False)
        runner.repo_root = self.repo_root
        runner.output_path = polar_path
        runner.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

        with patch.object(self.xfoil.subprocess, "Popen", return_value=FakeProcess()):
            self.assertTrue(
                runner.simulate(
                    "naca0012",
                    mach=0.1,
                    reynolds=100000,
                    alpha_min_deg=-5,
                    alpha_max_deg=25,
                )
            )

        alpha_lines = [
            line for line in captured["command"].splitlines() if line.startswith("alfa ")
        ]
        self.assertEqual("alfa 18", alpha_lines[-1])
        self.assertNotIn("alfa 19", alpha_lines)
        self.assertNotIn("alfa 25", alpha_lines)


if __name__ == "__main__":
    unittest.main()
