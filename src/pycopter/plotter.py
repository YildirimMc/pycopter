import matplotlib.pyplot as plt
from .pycopter import Rotor



# This won't work because not every value can be x(input).
# Make the plots predefined buttons, and generate plot into the mgl_widget when pressed.
# For now, only 1 plot visible. Add save button if possible.

class Plotter():
    def __init__(self, rotor: Rotor):
        self.rotor = rotor

    def plot_velocity_endurance_vs_range(xmin, xmax):
        pass
    # For now only one plot. Next version v0.2 with custom request plots.


