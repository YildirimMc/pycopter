from .pyrotorui import Ui_pycopter
from pycopter import Rotor

import matplotlib.pyplot as plt
import numpy as np

class Interface():
    def __init__(self, ui: Ui_pycopter):
        self.ui = ui

        self.ui.initRotorBtn.clicked.connect(self.init_rotor)
        self.ui.calcHoverBtn.clicked.connect(self.calculate_hover)
        self.ui.calcForwardFlightBtn.clicked.connect(self.calculate_forward_flight)

    def print(self, text):
        self.ui.textOutputWidget.appendPlainText(text)

    def init_rotor(self):
        airfoil = self.ui.airfoilText.toPlainText().lower()
        num_blades = self.ui.numBladesSpin.value()
        chord = self.ui.bladeChordSpin.value()
        diameter = self.ui.rotorDiameterSpin.value()
        tip_speed_mach = self.ui.tipSpeedMachSpin.value()
        washout = self.ui.washoutSpinner.value()
        rotor_root_cutout = self.ui.rotorRootCutoutSpinner.value()
        b_new_polars = self.ui.bNewPolars.isChecked() 

        if airfoil[:4] != "naca": 
            self.print("Airfoil must be a NACA profile.") 
            return

        self.print("Initializing rotor...")
        self.rotor = Rotor(airfoil, num_blades, chord, diameter, tip_speed_mach, 
                           washout, rotor_root_cutout, b_new_polars)
        self.print(f"Tip Speed: {self.rotor.tip_speed} [m/s] | Rotor Disk Area: {self.rotor.rotor_disk_area} [m2] | Solidity: {self.rotor.solidity}")
    
        self.ui.calcHoverBtn.setEnabled(True)
        
    def calculate_hover(self):
        gross = self.ui.grossSpinner.value()
        density = self.ui.densitySpinner.value()

        self.print("\nCalculating Hover Conditions...")
        self.rotor.hover(gross, density)
        self.print(f"Theta = {self.rotor.theta}° | Induced Velocity= {self.rotor.hover_induced_vel}[m/s] | Thrust = {self.rotor.hover_thrust/9.81}[kg]")
        self.print(f"SHP Induced: {self.rotor.hover_power_induced*0.00134102209} | SHP Profile: {self.rotor.hover_power_profile*0.00134102209} | SHP Total: {self.rotor.hover_power_total*0.00134102209}")
        self.print(f"Coeffs: {self.rotor.ct}, {self.rotor.cp} | Merits: {self.rotor.merit}, {self.rotor.merit_max}, {self.rotor.merit / self.rotor.merit_max} | Tip Loss: {self.rotor.tip_loss}")
        self.print(f"Thrust in ground effect at 10 meters = {self.rotor.ige(self.rotor.hover_thrust, 10)}")

        self.ui.calcForwardFlightBtn.setEnabled(True)

    def calculate_forward_flight(self):
        velocity = self.ui.velocitySpinner.value() / 3.6 # Converting to m/s
        density = self.ui.densitySpinner.value() 
        flat_plate_area = self.ui.fpaSpinner.value()

        self.print("\nCalculating Forward Flight Conditions...")
        self.rotor.forward_flight(velocity, density, flat_plate_area)
        self.print(f"Alfa: {self.rotor.alfa} | Downwash Velocity Ratio: {self.rotor.downwash_velocity_ratio} | Check if {self.rotor.power_profile/(2*density*self.rotor.rotor_disk_area*self.rotor.r)} is smaller than {velocity**2 / 2} for horsepowers below.")
        self.print(f"SHP Induced: {self.rotor.power_induced*0.00134102209} | SHP Profile:  {self.rotor.power_profile*0.00134102209} | SHP Parasite: {self.rotor.power_parasite*0.00134102209} | HP Total: {self.rotor.horsepower_total}")
        


def generate_plot(gui: Ui_pycopter, rotor: Rotor):
    pass

def initRotor_onClick():
    pass
# sfc = 0.35
# transmission_loss = 1.12 # ~0.9
# plot_range_endurance_vs_velocity(rotor, sfc, transmission_loss, 12000, 1900)
# plot_ctsigma_to_maxmerit(rotor, 12500)
# plot_nfs_to_dvr(rotor, 12500)
# plot_ground_effect(rotor, 12500)
# plot_range_electric(rotor, 12500, transmission_loss, 1000)
    
"""
def plot_range_endurance_vs_velocity(rotor: Rotor, sfc, transmission_loss, gross, fuel):
    rotor.hover(gross)
    lift = rotor.hover_thrust
    test_velocities = np.linspace(10,80, 50)
    effective_drags = np.zeros((50))
    total_powers = effective_drags.copy()

    for i, vel in enumerate(test_velocities):
        rotor.forward_flight(vel)
        effective_drags[i] = rotor.drag_induced + rotor.drag_profile + rotor.body_drag
        total_powers[i] = rotor.power_total
    total_powers /= 1000
    engine_powers = total_powers.copy()
    engine_powers *= transmission_loss
    test_velocities *= 3.6
    
    ld = lift/effective_drags
    brequet_ranges = 325/sfc * ld * np.log(gross/(gross-fuel))  
    fuel_consumptions = engine_powers * sfc
    specific_range = test_velocities / (fuel_consumptions) # km range / kg fuel
    endurance = fuel / fuel_consumptions
    
    fig, ax1 = plt.subplots()

    ax1.plot(test_velocities, total_powers*1.34102209, color="red", label="SHP")
    ax1.set_xlabel("Free Stream Velocity [km/hr]")
    ax1.set_ylabel("Shaft Horsepower [SHP]")
    ax1.set_ylim([0, max(total_powers*1.34102209)+100])

    ax2 = ax1.twinx()
    ax2.plot(test_velocities, brequet_ranges, label="Range")
    ax2.set_ylabel("Range [km]")

    fig.legend()
    ax1.grid()
    plt.title("Mil Mi-8")
    plt.show()
    
    fig, ax1 = plt.subplots()

    ax1.plot(test_velocities, endurance, color="red", label="Endurance [hr]")
    ax1.plot(test_velocities, ld, label="Lift/Drag Ratio")
    ax1.set_xlabel("Free Stream Velocity [km/hr]")

    ax2 = ax1.twinx()
    ax2.plot(test_velocities, specific_range, color="orange", label="Specific Range")
    ax2.set_ylabel("Specific Range [km/kgf]")

    fig.legend(loc="center")
    ax1.grid()
    plt.title("Mil Mi-8")
    plt.show()

def plot_ctsigma_to_maxmerit(rotor: Rotor, gross):
    rotor.hover(gross)
    
    ct_range = np.linspace(rotor.ct - rotor.ct/2, rotor.ct + rotor.ct/2, 50)
    merit_max = 0.707 * ct_range**1.5 / (ct_range**1.5 / (np.sqrt(2) * rotor.tip_loss) + rotor.solidity * rotor.cd_mean / 8)

    ct_sigma_range = ct_range / rotor.solidity
    plt.plot(ct_sigma_range, merit_max)
    plt.xlabel("Ratio of Thrust Coefficient to Solidity")
    plt.ylabel("Maximum Figure of Merit")
    plt.grid()
    plt.show()

def plot_nfs_to_dvr(rotor: Rotor, gross):
    rotor.hover(gross)

    velocities = np.linspace(0, 50, 50)
    downwash_velocity_ratios = np.zeros((50))
    for i, velocity in enumerate(velocities):
        rotor.forward_flight(velocity)
        downwash_velocity_ratios[i] = rotor.downwash_velocity_ratio
    normalized_flight_speeds = velocities / rotor.hover_induced_vel

    plt.plot(normalized_flight_speeds, downwash_velocity_ratios, label="Forward Flight")
    plt.title("Wald's Equation")
    plt.xlabel("Normalized Flight Speed V/v0")
    plt.ylabel("Downwash Velocity Ratio")
    plt.grid()
    plt.legend()
    plt.show()

def plot_ground_effect(rotor: Rotor, gross):
    rotor.hover(gross)
    heights = np.linspace(5.65, 50, 50)
    thrusts_ige = np.zeros((50))
    for i, height in enumerate(heights):
        thrusts_ige[i] = rotor.ige(rotor.hover_thrust, height)

    plt.plot(thrusts_ige/9.81, heights, label="Thrust")
    plt.axvline(rotor.hover_thrust/9.81, color="k", label="Base Thrust")
    plt.scatter(thrusts_ige[0]/9.81, 5.65, marker=(5,1), color="red", label="Landed Thrust")
    plt.xlabel("Thrust in Ground Effect [kg]")
    plt.ylabel("Rotor Height [m]")
    plt.grid()
    plt.legend()
    plt.show()

def plot_range_electric(rotor: Rotor, gross, transmission_loss, battery_capacity):
    rotor.hover(gross)
    test_velocities = np.linspace(10,80, 50)
    total_powers = np.zeros((50))

    for i, vel in enumerate(test_velocities):
        rotor.forward_flight(vel)
        total_powers[i] = rotor.power_total
    total_powers /= 1000
    total_powers *= transmission_loss
    test_velocities *= 3.6

    endurance = battery_capacity / total_powers
    range = endurance * test_velocities

    fig, ax1 = plt.subplots()
    ax1.plot(test_velocities, endurance, label="Endurance")
    ax1.set_xlabel("Velocity [km/hr]")
    ax1.set_ylabel("Endurance [hr]")

    ax2 = ax1.twinx()
    ax2.plot(test_velocities, range, label="Range", color="Orange")
    ax2.set_ylabel("Range [km]")
    ax1.grid()
    fig.legend()
    plt.show()
    """