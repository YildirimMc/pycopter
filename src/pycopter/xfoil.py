import subprocess
import os
import numpy as np

class Xfoil():
    """Contains methods to communicate with the XFOIL.exe instance running in the background."""

    def __init__(self, airfoil, new_polar=True):
        self.exe_path = "data/XFOIL6.99/xfoil.exe"
        self.output_path = "data/XFOIL6.99/polar.txt"
        self.airfoil = airfoil
        self.max_theta = 15
        
        self.process = subprocess.Popen(self.exe_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)

        if new_polar and os.path.exists(self.output_path):
            os.remove(self.output_path) 
        
    def simulate(self, mach, reynolds):
        """
        """
        inputs_init = [self.airfoil, "oper", "iter 400", "v", str(reynolds), f"mach {mach}", "pacc", self.output_path + "\n"]
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
        """
        """
        return np.genfromtxt(self.output_path, skip_header=12)


if __name__ == "__main__":
    xfoil = Xfoil("naca23012", True)
    xfoil.simulate(0.3, 4000000)


