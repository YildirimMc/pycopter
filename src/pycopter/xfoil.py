import subprocess

class Xfoil():
    def __init__(self, airfoil, mach, reynolds):
        self.exe_path = "data/XFOIL6.99/xfoil.exe"
        self.output_path = "data/XFOIL6.99/polar.txt"
        self.airfoil = airfoil
        self.mach = mach
        self.reynolds = reynolds
        self.polar = []
        
        self.process = subprocess.Popen(self.exe_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)

    def readPolar(self, path):
        with open(path, "r") as txt:
            data = txt.readlines()
            data = data[-1].split("   ")
            return float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7].strip())

    def simulate(self, alfa):
        inputs = [self.airfoil, "oper", "iter 400", "v", str(self.reynolds), f"mach {self.mach}", "pacc", self.output_path + "\n", f"alfa {alfa}"]

        command = ""
        for input in inputs:
            command = command + input + "\n"
        
        output, error = self.process.communicate(input=command)
        self.polar = self.readPolar(self.output_path)
        return self.polar
    
    def getPolar(self):
        return self.polar

xfoil = Xfoil("naca0012", 0.6, 200000)
polar_info = xfoil.simulate(alfa=6)
# polar_info = xfoil.getPolar()
print(polar_info[0])


