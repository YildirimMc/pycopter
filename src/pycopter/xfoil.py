import os
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UIUC_COORD_BASE_URL = "https://m-selig.ae.illinois.edu/ads/coord"
AIRFOIL_NAME_RE = re.compile(r"^[a-z0-9_.-]+$")
MAX_XFOIL_ALPHA_DEG = 15


def get_repo_root():
    """Returns the repository root from this source file location."""
    return Path(__file__).resolve().parents[2]


def normalize_airfoil_name(airfoil):
    """Normalizes GUI/user airfoil names for lookup."""
    airfoil = airfoil.strip().lower().replace(" ", "")
    if airfoil.endswith("-il"):
        airfoil = airfoil[:-3]
    return airfoil


def is_naca_airfoil(airfoil):
    """Returns True for the NACA profile names supported by the existing UI."""
    airfoil = normalize_airfoil_name(airfoil)
    digits = airfoil[4:]
    return airfoil.startswith("naca") and 4 <= len(digits) <= 6 and digits.isdigit()


def _is_valid_airfoil_dat(text):
    if not text.strip():
        return False
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return False

    coord_count = 0
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            float(parts[0])
            float(parts[1])
        except ValueError:
            continue
        coord_count += 1

    return coord_count >= 3


def ensure_airfoil_coordinates(airfoil, repo_root=None, timeout=10):
    """
    Ensures a non-NACA airfoil coordinate file exists locally.

    Returns (success, path, message). Missing UIUC records and network failures
    return False instead of raising so callers can reject user input cleanly.
    """
    airfoil = normalize_airfoil_name(airfoil)
    if not airfoil:
        return False, None, "Airfoil name is empty."
    if is_naca_airfoil(airfoil):
        return True, None, ""
    if airfoil.startswith("naca"):
        return False, None, "Invalid NACA airfoil name."
    if not AIRFOIL_NAME_RE.fullmatch(airfoil):
        return False, None, "Airfoil name contains unsupported characters."

    repo_root = Path(repo_root) if repo_root is not None else get_repo_root()
    airfoil_path = repo_root / "data" / "airfoils" / f"{airfoil}.dat"
    if airfoil_path.exists():
        text = airfoil_path.read_text(encoding="utf-8", errors="replace")
        if _is_valid_airfoil_dat(text):
            return True, airfoil_path, ""
        return False, None, f"Local airfoil file is invalid: {airfoil_path}"

    url = f"{UIUC_COORD_BASE_URL}/{airfoil}.dat"
    request = Request(url, headers={"User-Agent": "pycopter"})
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError) as err:
        return False, None, f"Airfoil '{airfoil}' was not found locally or could not be downloaded from UIUC: {err}"

    if not _is_valid_airfoil_dat(text):
        return False, None, f"Airfoil '{airfoil}' was not found in the UIUC coordinate database."

    airfoil_path.parent.mkdir(parents=True, exist_ok=True)
    airfoil_path.write_text(text, encoding="utf-8")
    return True, airfoil_path, ""


def get_airfoil_commands(airfoil, repo_root=None):
    """Returns XFOIL commands needed to load the requested airfoil."""
    airfoil = normalize_airfoil_name(airfoil)
    if is_naca_airfoil(airfoil):
        return [airfoil], ""

    success, _, message = ensure_airfoil_coordinates(airfoil, repo_root=repo_root)
    if not success:
        return None, message

    return [f"load data/airfoils/{airfoil}.dat", "pane"], ""

class Xfoil():
    """
    Contains methods to communicate with the XFOIL.exe.

    Methods
    -------
    simulate(airfoil : str, mach : float, reynolds : float) -> None
        Requests polar data for the airfoil and flow conditions defined by the user.
    read_polar() -> ndarray
        Reads and returns the polar data.
    """

    def __init__(self, new_polar=True, timeout=60):
        """
        Prepares XFOIL paths and runtime settings.

        parameters
        ----------
        new_polar : bool
            Whether to request new polars or use an existing one.
        """
        self.new_polar = new_polar
        self.repo_root = get_repo_root()
        self.exe_path = self.repo_root / "data" / "XFOIL6.99" / "xfoil.exe"
        self.output_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
        self.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"
        self.max_theta = MAX_XFOIL_ALPHA_DEG
        self.timeout = timeout
        self.error_message = ""
        
    def simulate(
        self,
        airfoil: str,
        mach: float,
        reynolds: float,
        alpha_min_deg: float = -8,
        alpha_max_deg: float | None = None,
    ):
        """
        Runs the xfoil process with the given parameters. Xfoil process saves the polar data in a temporary location.

        Parameters
        ----------
        airfoil : str
            NACA profile or UIUC coordinate-backed airfoil. E.g. 'naca0012' or 's1223'.
        mach : float
            Mach number of the airfoil.
        reynolds : float
            The Reynold's number.
        """
        if self.new_polar and os.path.exists(self.output_path):
            os.remove(self.output_path)

        if reynolds <= 0:
            self.error_message = "ERROR - XFOIL Reynolds number must be positive."
            return False
        if mach < 0:
            self.error_message = "ERROR - XFOIL Mach number must not be negative."
            return False

        airfoil_commands, error_message = get_airfoil_commands(airfoil, self.repo_root)
        if airfoil_commands is None:
            self.error_message = f"ERROR - {error_message}"
            return False

        inputs_init = airfoil_commands + [
            "oper",
            "iter 400",
            "v",
            str(reynolds),
            f"mach {mach}",
            "pacc",
            self.output_path_for_xfoil,
            "",
        ]
        if alpha_max_deg is None:
            alpha_max_deg = self.max_theta
        alpha_min = int(round(alpha_min_deg))
        alpha_max = min(int(round(alpha_max_deg)), self.max_theta)
        if alpha_min > alpha_max:
            self.error_message = (
                f"ERROR - XFOIL alpha range must end at or below {self.max_theta} deg."
            )
            return False
        inputs = [f"alfa {alfa}" for alfa in range(alpha_min, alpha_max + 1)]
        command = "\n".join(inputs_init + inputs + ["pacc", "", "quit"]) + "\n"

        try:
            process = subprocess.Popen(
                self.exe_path,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
            )
        except OSError as err:
            self.error_message = f"ERROR - Could not start XFOIL: {err}"
            return False

        try:
            output, error = process.communicate(input=command, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.error_message = f"ERROR - XFOIL timed out while generating a polar for '{airfoil}'."
            return False

        if not self.output_path.exists():
            self.error_message = f"ERROR - XFOIL did not create {self.output_path_for_xfoil}."
            if output or error:
                self.error_message += " Check XFOIL output for convergence or input errors."
            return False

        return True
        
        # This just gives error all the time.
        # if output.find("Convergence failed") != -1:
        #     print("XFOIL convergence has failed due to highly turbulent flow. Polars for high theta are expected to be faulty. Confirm the alfa vs cl vs cd plots.")

    def read_polar(self):
        """Reads and returns the polar data[ndarray] that was created by XFOIL.exe."""
        import numpy as np

        polar = np.genfromtxt(self.output_path, skip_header=12)
        if polar.size == 0:
            raise ValueError("XFOIL polar contains no data rows.")
        polar = np.atleast_2d(polar)
        if polar.shape[0] < 3:
            raise ValueError("XFOIL polar contains too few data rows.")
        return polar



if __name__ == "__main__":
    xfoil = Xfoil(True)
    xfoil.simulate("naca23012", 0.3, 4000000)


