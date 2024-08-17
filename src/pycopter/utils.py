import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import sys

from .xfoil import Xfoil


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
    # Deprecated. Using scipy in the future.
    """2D interpolation."""
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
    
def probe_txt(filepath, value):
    """Returns the value interpolated from the data in the given filepath."""
    data = read_txt(filepath)
    idx = find_interval_idx(data, value)
    slope = (data[idx+1][1] - data[idx][1]) / (data[idx+1][0] - data[idx][0])
    probe = (value - data[idx][0]) * slope + data[idx][1] 
    return probe
    
def reynolds(velocity, ch_len, kin_vis):
    """Returns the Reynolds number."""
    return velocity * ch_len / kin_vis

class Polar():
    """
    A class to interface between the main program and the xfoil.py. 
    Requests new polar data with the given parameters or reads an existing one on initialization.

    Methods
    -------
    get_polar(alfa: float) -> float, float
        Returns the cl, cd values of the airfoil for the given alfa.
    get_cl_slope() -> float
        Returns the cl/alfa slope of the airfoil around alfa=1°.
    """
    def __init__(self, airfoil, mach, reynolds, new_polar=True):
        """
        Initializes the Polar class. The parameters are used to create polar data from Xfoil.

        Parameters
        ----------
        airfoil : str
            Airfoil name. Only naca profiles are available. E.g. 'naca0012'.
        mach : float
            Mach speed of the airfoil.
        reynolds : float
            Reynolds number of the flow around the airfoil.
        new_polar : bool
            Whether to request new polar data or use the existing one.
        """
        self.reynolds = reynolds

        xfoil = Xfoil(new_polar)
        if new_polar:
            print("Generating XFOIL polar...")
            xfoil.simulate(airfoil, mach, reynolds)
            self.polar = xfoil.read_polar()
        else:
            try:
                self.polar = xfoil.read_polar()
            except FileNotFoundError:
                print("Polar data not found. Generating new XFOIL polar...")
                xfoil.simulate(airfoil, mach, reynolds)
                self.polar = xfoil.read_polar()

    def get_polar(self, alfa):
        """Returns (cl, cd) values for the requested angle of attack in degrees."""
        func = interp1d(self.polar[:,0], [self.polar[:,1], self.polar[:,2]], kind='linear', axis=1)
        cl, cd = func(alfa)
        return cl, cd
    
    def get_cl_slope(self):
        """Returns the initial slope[1/rad] of the cl vs alfa curve."""
        return (self.polar[2,1] - self.polar[0,1]) / (np.deg2rad(self.polar[2,0]) - np.deg2rad(self.polar[0,0]))
    
def get_flat_plate_area(helicopter="mi-8"):
        # Deprecated. User enters fpa in GUI.
        """Reads and returns the flat plate area from database."""
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

def calculate_component_reference_area(image_path:str, width:float, height:float):
    """
    Calculates the component reference area of the aircraft's body. Image must be an orthogonal picture with the object in black and the background in white. 
    See examples in data/cra_images/. Required for calculating the flat plate area.

    Parameters
    ---------
    image_path : str
        Filepath of the image.
    height : float
        Height of the image in meters.
    width : float
        Width of the image in meters.
    """
    image = Image.open(image_path).convert("1")
    pixels = image.getdata()
    black_pixels = sum([1 for pixel in pixels if pixel == 0])
    ratio = black_pixels / len(pixels)
    component_reference_area = ratio * height * width
    return component_reference_area

def walds_equation(normalized_flight_speed:float, downwash_velocity_ratio:float, alfa=0):
    """
    Wald's Equation as defined in ref: 'AMCP 706-201, pdf pg.92'.
    
    Parameters
    ----------
    normalized_flight_speed : float
        Forward flight speed divided by hover induced velocity. [V/v0]
    downwash_velocity_ratio : float
        Forward flight induced velocity divided by hover induced velocity. [v/v0]
    alfa : float [°]
        Rotor collective pitch. Assume 0° for level forward flight.

    Returns
    -------
    float
        Result of the Wald's Equation. Must equal to 1.
    """
    output = downwash_velocity_ratio**4 - 2 * downwash_velocity_ratio**3 * normalized_flight_speed * np.sin(np.deg2rad(alfa)) \
            + normalized_flight_speed**2 * downwash_velocity_ratio**2
    return output

def walds_solver(free_stream_velocity:float, hover_induced_velocity:float, alfa=0, iter=100):
    """
    Numerically solves the Wald's Equation and returns the downwash velocity ratio [v/v0].
    
    Parameters
    ----------
    free_stream_velocity : float [m/s]
        Also known as forward flight speed. Is the level velocity of the aircraft.
    hover_induced_velocity : float [m/s]
        Mean axial velocity of the air molecules being pushed down by the rotor at the rotor disk plane in hover.
    alfa : float [°]
        Rotor collective pitch. Assume 0° for level forward flight.
    iter : int
        Iterations if the solver. The solver doesn't have a convergence criteria but it usually converges after around 10 iterations.

    Returns
    -------
    float
        The downwash velocity ratio. Forward flight induced velocity divided by hover induced velocity. [v/v0]
    """    
    normalized_flight_speed = free_stream_velocity / hover_induced_velocity
    downwash_velocity_ratio = 1 # Initialization.
    gamma = 0.02 # Step size.
    target = 1

    for _ in range(iter):
        output = walds_equation(normalized_flight_speed, downwash_velocity_ratio, alfa)
        downwash_velocity_ratio -= (output - target) * gamma

    return downwash_velocity_ratio



if __name__ == "__main__":
    # polar = Polar("naca23012", 0.3, 4000000, False)
    # print(polar.get_polar(3.5))

    print(walds_solver(20, 10, 0))
