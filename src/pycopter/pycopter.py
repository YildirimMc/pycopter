"""Main module."""

import numpy as np
import matplotlib.pyplot as plt

from .utils import *

class Rotor():
    """
    Default values for Mil Mi-8.
    """
    def __init__(self, airfoil="naca23012", num_blades=5, chord=0.52, rotor_diameter=21.29, tip_speed_mach=0.624, washout=-8, rotor_root_cutout=0.01, new_polar=True): # new_polar only for debug. # TODO: Add spellchecking for airfoil.
        self.airfoil = airfoil
        self.num_blades = num_blades
        self.chord = chord
        self.rotor_diameter = rotor_diameter
        self.tip_speed_mach = tip_speed_mach
        self.washout = washout
        self.rotor_root_cutout = rotor_root_cutout

        print("\nInitializing rotor...")
        self.speed_of_sound = 343 # TODO: Needs to be calculated per density. Also the mach variables below.
        self.r = self.rotor_diameter / 2
        self.omega_mach = self.tip_speed_mach / self.r
        self.omega = self.omega_mach * self.speed_of_sound
        self.rpm = self.omega * 30 / np.pi
        self.tip_speed = self.omega * self.r
        self.rotor_disk_area = np.pi * self.r**2 - np.pi * (self.r * self.rotor_root_cutout)**2
        self.solidity = num_blades * chord / (np.pi * self.r)
        print("Tip Speed:", self.tip_speed, "[m/s] | Rotor Disk Area:", self.rotor_disk_area, "[m2] | Solidity:", self.solidity)

        self.polar = Polar(self.airfoil, self.tip_speed_mach / 2, reynolds(self.tip_speed/2, self.chord, 1.5e-5), new_polar) # Creating XFOIL polar data for the middle element.
        self.is_hovered = False

    def calculate(self, density=1.225, theta=8):
        # Deprecated
        """# Hover
        self.thrust = density * self.num_blades * self.chord * (self.R*0.5)**3 * self.omega**2 * self.cl / 6
        self.power = self.thrust**1.5 / (self.tip_loss * np.sqrt(2*density*self.rotor_disk_area)) + density*self.num_blades*self.chord*(self.r**4) * (self.omega**3) * self.cd / 8
        self.horsepower = self.power * 0.00134102209

        # Thrust, power, and torque coefficients.
        self.ct = self.thrust / (density * np.pi * self.r**4 * self.omega**2)
        self.cp = self.power / (density * np.pi * self.r**5 * self.omega**3) 
        self.cq = self.cp

        # Calculations for main rotor Figure of Merit.
        self.M = 0.707 * self.ct**(3/2) / self.cq # M ~ 0.75 can be considered representative of a relatively efficient rotor.
        self.cl_mean = 6 * self.ct / (self.tip_loss**3 * self.solidity)
        self.alfa = 6 * self.ct / (self.solidity * self.a * self.tip_loss**3) # This equation and below are empirical from NACA0012.
        self.cd_mean = 0.0087 - 0.0216 * self.alfa + 0.4 * self.alfa**2
        self.M_actual = 0.707 * self.ct**(3/2) / (self.ct**(3/2) / (self.tip_loss * np.sqrt(2)) + (self.solidity * self.cd_mean / 8))
        self.FMR = self.M_actual / self.M # TODO: Figure this thing out.
        
        self.tip_loss_2 = 1 - np.sqrt(2 * self.ct) / self.num_blades

        print("T&HP:", self.thrust, self.horsepower, "Coefs:", self.ct, self.cp, self.cq, "Merits:", self.M, self.M_actual, self.FMR, "B =", self.tip_loss_2)"""
    
    def ige(self, thrust, rotor_height): # In Ground Effect
        return thrust / (1 - (self.r**2 / (16 * rotor_height**2)))

    def hover(self, weight=13000, density=1.225, n=10): 
        weight *= 9.81
        print("\nCalculating Hover Conditions...")
        self.is_hovered = True
        dr = self.r / n # Blade element radial length.
        
        # Blade elemental induced velocity.
        phi = np.zeros((n))
        alfa = np.zeros((n))
        for theta in np.linspace(0, 15, 31, endpoint=True):
            induced_vel_old = np.zeros((n))
            induced_vel = np.ones((n))
            iter = 0
            while np.linalg.norm(induced_vel - induced_vel_old) > 1e-12:
                induced_vel_old = induced_vel.copy()
                for i, r_i in enumerate(np.linspace(0, self.r, n, endpoint=False)):
                    element_center = (r_i + dr/2)
                    elemental_theta = theta + (element_center / self.r) * (self.washout) # TODO: Implement dual washout
                    phi[i] = np.arctan(induced_vel[i] / (self.omega * element_center))
                    alfa[i] = elemental_theta - np.rad2deg(phi[i])
                    if alfa[i] < -1: alfa[i] = -1
                    if alfa[i] > 20: alfa[i] = 20
                    cl, cd = self.polar.get_polar(alfa[i]) 
                    inner_root = element_center * (cl * np.cos(phi[i]) - cd * np.sin(phi[i]))
                    if inner_root < 0:
                        induced_vel[i] = -np.sqrt(-inner_root)
                    else:
                        induced_vel[i] = np.sqrt(inner_root)
                induced_vel *= np.sqrt(self.num_blades * self.chord / (8 * np.pi)) * self.omega
                iter +=1
                if iter > 10:
                    break

            # Blade elemental thrust.
            thrust = 0
            drag = 0
            for i, r_i in enumerate(np.linspace(0, self.r, n, endpoint=False)):
                element_center = (r_i + dr/2)
                thrust += element_center * induced_vel[i]**2
            thrust *= 4 * density * np.pi * dr * 0.97 # 0.97 for tip loss.

            if thrust >= weight: break

        print(f"Theta = {theta}° | Induced Velocity= {np.mean(induced_vel)}[m/s] | Thrust = {thrust/9.81}[kg]")

        # Parameters calculation based on maaaany simplifications.
        self.theta = theta
        self.hover_induced_vel = np.mean(induced_vel)
        self.hover_thrust = thrust

        self.ct = thrust / (density * np.pi * self.r**4 * self.omega**2)
        self.tip_loss = 1 - np.sqrt(2 * self.ct) / self.num_blades
        self.cl_mean = 6 * self.ct / (self.tip_loss**3 * self.solidity)
        self.a = self.polar.get_cl_slope()
        self.alfa = 6 * self.ct / (self.solidity * self.a * self.tip_loss**3)
        self.cd_mean = 0.0087 - 0.0216*self.alfa + 0.4*self.alfa**2 # TODO: This depends on empirical data based on naca0012. Get the relation from the polar, instead. Furthermore, this needs to satisfy Fig3-1 for naca0012(prolly).
        self.hover_power_induced = thrust**1.5 / (self.tip_loss * np.sqrt(2*density * self.rotor_disk_area))
        self.hover_power_profile = density * self.num_blades * self.chord * (self.r**4) * (self.omega**3) * self.cd_mean / 8 
        self.hover_power_total = self.hover_power_induced + self.hover_power_profile
        self.cp = self.hover_power_total / (density * np.pi * self.r**5 * self.omega**3)
        self.merit = 0.707 * self.ct**1.5 / self.cp
        self.merit_max = 0.707 * self.ct**1.5 / (self.ct**1.5 / (np.sqrt(2) * self.tip_loss) + self.solidity * self.cd_mean / 8)

        print("SHP Induced:", self.hover_power_induced*0.00134102209, "| SHP Profile:", self.hover_power_profile*0.00134102209, "| SHP Total:", self.hover_power_total*0.00134102209)
        print("Coeffs:", self.ct, self.cp, "| Merits:", self.merit, self.merit_max, self.merit / self.merit_max, "| Tip Loss:", self.tip_loss)

    def forward_flight(self, velocity, density=1.225, flat_plate_area=3.5):
        if not (isinstance(velocity, float) or isinstance(velocity, int)):
            raise ValueError("Forward flight velocity must be a numeric singleton.")
        if not self.is_hovered: 
            print("Running hover calculations for level blade first...")
            self.hover()

        print("\nCalculating Forward Flight Conditions...")
        # Simplifications
        self.alfa = 0 # pg.93, AMCP 706-201

        advance_ratio = velocity / (self.omega * self.r) # velocity * np.cos(self.alfa) / (self.omega * self.r)
        self.downwash_velocity_ratio = walds_solver(velocity, self.hover_induced_vel, self.alfa)
        
        self.power_induced = self.downwash_velocity_ratio * self.hover_power_induced
        self.drag_induced = self.power_induced / (2*self.hover_induced_vel) # v2. not v.
        if self.power_induced < 0: self.power_induced = 0
        
        self.power_profile = (density * self.num_blades * self.chord * self.r**4 * self.omega**3 * self.cd_mean / 8) * (1 + 4.65*advance_ratio**2)
        self.drag_profile = self.power_profile / (self.omega * self.r)

        self.body_drag = density * velocity**2 * flat_plate_area / 2 # Flat plate area is component_reference_area * drag_coefficient for the body.
        self.power_parasite = self.body_drag * velocity

        self.power_total = self.power_induced + self.power_profile + self.power_parasite
        self.horsepower_total = self.power_total * 0.00134102209

        thrust_profile = self.power_profile / self.r
        print(f"Alfa: {self.alfa} | Downwash Velocity Ratio: {self.downwash_velocity_ratio} | Check if {thrust_profile/(2*density*self.rotor_disk_area)} is smaller than {velocity**2 / 2} for horsepowers below.")
        print("SHP Induced:", self.power_induced*0.00134102209, "| SHP Profile:", self.power_profile*0.00134102209, "| SHP Parasite:", self.power_parasite*0.00134102209, "| HP Total:", self.horsepower_total)
        
        # clr = self.hover_thrust / (0.5 * density * velocity**2 * self.rotor_disk_area)
        # clr_sigma_ratio = clr / self.solidity
        # De_Lr_ratio = 1 # Assuming initially.
        # inflow_ratio = (velocity * np.sin(self.alfa) - self.hover_induced_vel) / (self.omega * self.r)
        # Di_Lr_ratio = self.ct / (2 * advance_ratio * np.sqrt(advance_ratio**2 + inflow_ratio**2))
        # Dp_Lr_ratio = flat_plate_area / (self.rotor_disk_area * clr)

        # TODO: Implement blade stall.
    

    def plot(self):
        print("\nPlot debug.")
        print("Ct/sigma:", self.ct / self.solidity)
        
class Engine():
    def __init__(self, sfc, test_wg, test_shp, test_wf):
        self.sfc = sfc
        pass

class Body():
    def __init__(self, dry_weight, fuel_capacity, flat_plate_area):
        self.dry_weight = dry_weight
        self.fuel_capacity = fuel_capacity
        self.fpa = flat_plate_area

class Copter():
    def __init__(self):
        pass

        
    
class RotorImperial(Rotor):
    def __init__(self, a):
        pass
        print(a) # TODO: Test the imperial version

class Optimizer():
    """A class that tests the Rotor with different thetas and densities, saves the values, and goes on with further combined calculations."""
    def __init__(self):
        pass

if __name__ == "__main__":
    rotor = Rotor(new_polar=False)
    rotor.hover(4)
    rotor.forward_flight(40)
    rotor.plot()

    