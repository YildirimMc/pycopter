import numpy as np
from scipy.interpolate import interp1d

from xfoil import Xfoil


def find_nearest_idx(array, val):
    arr = np.asarray(array)
    return (np.abs(arr - val)).argmin()

def find_interval_idx(array, val):
    """
    """
    i = 0
    for i in range(len(array-1)):
        if array[i][0] <= val < array[i+1][0]:
            break
    return i

def interpolate(val, x1, x2, y1, y2):
    slope = (y2 - y1) / (x2 - x1)
    return (val - x1) * slope + y1

def read_txt(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        data = np.empty(shape=(len(lines), 2))
        for i, line in enumerate(lines):
            data[i] = line.split(" ")
        return data
    
def probe_txt(filepath, x):
    data = read_txt(filepath)
    idx = find_interval_idx(data, x)
    slope = (data[idx+1][1] - data[idx][1]) / (data[idx+1][0] - data[idx][0])
    probe = (x - data[idx][0]) * slope + data[idx][1] 
    return probe
    
def reynolds(velocity, ch_len, kin_vis):
    return velocity * ch_len / kin_vis

class Polar():
    def __init__(self, airfoil, mach, reynolds, new_polar=True):
        xfoil = Xfoil(airfoil, new_polar)
        if new_polar:
            print("Generating XFOIL polar...")
            xfoil.simulate(mach, reynolds)
        try:
            self.polar = xfoil.read_polar()
        except FileNotFoundError:
            print("Polar data not found. Generating new XFOIL polar...")
            xfoil.simulate(mach, reynolds)
            self.polar = xfoil.read_polar()

    def get_polar(self, alfa):
        func = interp1d(self.polar[:,0], [self.polar[:,1], self.polar[:,2]], kind='linear', axis=1)
        cl, cd = func(alfa)
        return cl, cd
    
    def get_cl_slope(self):
        return (self.polar[2,1] - self.polar[0,1]) / (np.deg2rad(self.polar[2,0]) - np.deg2rad(self.polar[0,0]))


    

if __name__ == "__main__":
    polar = Polar("naca23012", 0.3, 4000000, False)
    print(polar.get_polar(3.5))

