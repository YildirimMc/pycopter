import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import sys

from xfoil import Xfoil


def find_nearest_idx(array, val):
    """Returns the index of the entry in the array that is nearest(L1) to the input value."""
    arr = np.asarray(array)
    return (np.abs(arr - val)).argmin()

def find_interval_idx(array, val):
    """Returns the left index of the value found in an array."""
    idx = -1
    for i in range(len(array-1)):
        if array[i][0] <= val < array[i+1][0]:
            idx = i
            break
    if idx == -1: raise ValueError(f"Value {val} not found in the array.") 
    else:
        return idx

def interpolate(val, x1, x2, y1, y2):
    # Deprecated. Using numpy in the future.
    slope = (y2 - y1) / (x2 - x1)
    return (val - x1) * slope + y1

def read_txt(filepath):
    """Reads a two column text file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
        data = np.empty(shape=(len(lines), 2))
        for i, line in enumerate(lines):
            data[i] = line.split(" ")
        return data
    
def probe_txt(filepath, x):
    """Returns the value interpolated from the data in the filepath."""
    data = read_txt(filepath)
    idx = find_interval_idx(data, x)
    slope = (data[idx+1][1] - data[idx][1]) / (data[idx+1][0] - data[idx][0])
    probe = (x - data[idx][0]) * slope + data[idx][1] 
    return probe
    
def reynolds(velocity, ch_len, kin_vis):
    """Returns the Reynolds number."""
    return velocity * ch_len / kin_vis

class Polar():
    """
    A class to interface between the main program and the xfoil.py

    Functions
    ---------
    """
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
        """Returns (cl, cd) values for the requested angle of attack."""
        func = interp1d(self.polar[:,0], [self.polar[:,1], self.polar[:,2]], kind='linear', axis=1)
        cl, cd = func(alfa)
        return cl, cd
    
    def get_cl_slope(self):
        """Returns the initial slope[1/rad] of the cl vs alfa curve."""
        return (self.polar[2,1] - self.polar[0,1]) / (np.deg2rad(self.polar[2,0]) - np.deg2rad(self.polar[0,0]))
    
def get_flat_plate_area(helicopter="mi-8"): # Possible carry into the Body class in the future.
        """    """
        fpa_dict = {}
        with open("data/flat_plate_areas.txt", "r") as file:
            for line in file:
                key, value = line.split(",")
                fpa_dict[key] = float(value.strip())

        try:
            fpa = fpa_dict[helicopter]
        except KeyError:
            available_keys = list(fpa_dict.keys())
            print(f"The flat plate area for {helicopter} is not found in database. Availabe keys are: {available_keys}\n",
                  "If yours is missing, please calculate it first via 'calculate_flat_plate_area()' and save to 'data/flat_plate_areas.txt'.")
            sys.exit(1)
        return fpa

def calculate_flat_plate_area(image_path, width, height, body_cd): # TODO: Saves the calculated value to database.
    """
    Returns the flat_plate_area. Image must be an orthogonal picture with a black object and a white background. See examples in data/cra_images/.
    Save the value to 'data/flat_plate_areas.txt' for future use and use 'get_flat_plate_area()'.

    Arguments
    ---------
    image_path (str): Filepath of the image.
    height (float): Height of the image in meters.
    width (float): Width of the image in meters.
    body_cd (float): Rotorless body drag coefficient.
    """
    image = Image.open(image_path).convert("1")
    pixels = image.getdata()
    black_pixels = sum([1 for pixel in pixels if pixel == 0])
    ratio = black_pixels / len(pixels)
    component_reference_area = ratio * height * width
    return component_reference_area * body_cd


if __name__ == "__main__":
    # polar = Polar("naca23012", 0.3, 4000000, False)
    # print(polar.get_polar(3.5))

    print(get_flat_plate_area())
