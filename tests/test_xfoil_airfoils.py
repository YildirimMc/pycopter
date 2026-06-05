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


if __name__ == "__main__":
    unittest.main()
