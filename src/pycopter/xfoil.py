import subprocess
import os
import numpy as np

class Xfoil():
    """
    Contains methods to communicate with the XFOIL.exe. Xfoil process is created and waits in the background at initialization.

    Methods
    -------
    simulate(airfoil : str, mach : float, reynolds : float) -> None
        Requests polar data for the airfoil and flow conditions defined by the user.
    read_polar() -> ndarray
        Reads and returns the polar data.
    """

    def __init__(self, new_polar=True):
        """
        Prepares the XFOIL.exe process.

        parameters
        ----------
        new_polar : bool
            Whether to request new polars or use an existing one.
        """
        self.new_polar = new_polar
        self.exe_path = "data/XFOIL6.99/xfoil.exe"
        self.output_path = "data/XFOIL6.99/polar.txt"
        self.max_theta = 15
        
        self.process = subprocess.Popen(self.exe_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)

        
        
    def simulate(self, airfoil:str, mach:float, reynolds:float):
        """
        Runs the xfoil process with the given parameters. Xfoil process saves the polar data in a temporary location.

        Parameters
        ----------
        airfoil : str
            Only naca profiles are supported. E.g. 'naca0012'.
        mach : float
            Mach number of the airfoil.
        reynolds : float
            The Reynold's number.
        """
        if self.new_polar and os.path.exists(self.output_path):
            os.remove(self.output_path) 

        inputs_init = [airfoil, "oper", "iter 400", "v", str(reynolds), f"mach {mach}", "pacc", self.output_path + "\n"]
        inputs = [f"alfa {alfa}" for alfa in np.arange(-8, self.max_theta + 6)]
        command = ""
        for input in inputs_init:
            command = command + input + "\n"
        for input in inputs:
            command = command + input + "\n"
        
        output, error = self.process.communicate(input=command)
        
        # This just gives error all the time.
        # if output.find("Convergence failed") != -1:
        #     print("XFOIL convergence has failed due to highly turbulent flow. Polars for high theta are expected to be faulty. Confirm the alfa vs cl vs cd plots.")

    def read_polar(self):
        """Reads and returns the polar data[ndarray] that was created by XFOIL.exe."""
        return np.genfromtxt(self.output_path, skip_header=12)



if __name__ == "__main__":
    xfoil = Xfoil("naca23012", True)
    xfoil.simulate(0.3, 4000000)


