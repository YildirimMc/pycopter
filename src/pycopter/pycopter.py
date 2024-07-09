"""Main module."""

import numpy as np
import matplotlib.pyplot as plt

import utils

class Rotor():
    """
    Default values for Mil Mi-8.
    """
    def __init__(self, airfoil="naca23012", num_blades=5, chord=0.53, rotor_diameter=21.29, tip_speed_mach=0.624, tip_loss=0.97, rotor_root_cutout=0.2, new_polar=True): # new_polar only for debug. # TODO: Add spellchecking for airfoil.
        self.airfoil = airfoil
        self.num_blades = num_blades
        self.chord = chord
        self.rotor_diameter = rotor_diameter
        self.tip_speed_mach = tip_speed_mach
        self.tip_loss = tip_loss
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

        self.polar = utils.Polar(self.airfoil, self.tip_speed_mach / 2, utils.reynolds(self.tip_speed/2, self.chord, 1.5e-5), new_polar) # Creating XFOIL polar data for the middle element.
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
        return thrust / (1 - (self.R**2 / rotor_height**2))

    def hover(self, theta, Z=100, density=1.225, n=10): 
        print("\nHover conditions.")
        self.is_hovered = True
        dr = self.r / n # Blade element radial length.
        if theta > 15: 
            print("Theta values are limited to 15 degrees.")
            theta = 15.

        induced_vel_old = np.zeros((n))
        induced_vel = np.ones((n))
        while np.linalg.norm(induced_vel - induced_vel_old) > 1e-8:
            induced_vel_old = induced_vel.copy()
            for i, r_i in enumerate(np.linspace(0, self.r, n, endpoint=False)):
                element_center = (r_i + dr/2)
                phi = np.arctan(induced_vel[i] / (self.omega * element_center))
                alfa = theta - phi
                cl, cd = self.polar.get_polar(alfa)
                induced_vel[i] = np.sqrt(element_center * (cl * np.cos(phi) - cd * np.sin(phi)))
            induced_vel *= np.sqrt(self.num_blades * self.chord / (8 * np.pi)) * self.omega

        # Blade element-wise thrust calculation.
        thrust = 0
        power = 0
        for i, r_i in enumerate(np.linspace(0, self.r, n, endpoint=False)):
            thrust += 2 * density * np.pi * (r_i + dr/2) * dr * induced_vel[i]**2

        print("Induced Velocity:", np.mean(induced_vel), "[m/s] | Thrust:", thrust, "[N]")

        # Parameters calculation based on maaaany simplifications.
        self.hover_induced_vel = np.mean(induced_vel)
        self.hover_thrust = thrust
        self.ct = thrust / (density * np.pi * self.r**4 * self.omega**2)
        self.tip_loss = 1 - np.sqrt(2 * self.ct) / self.num_blades
        self.cl_mean = 6 * self.ct / (self.tip_loss**3 * self.solidity)
        self.a = self.polar.get_cl_slope()
        self.alfa = 6 * self.ct / (self.solidity * self.a * self.tip_loss**3)
        self.cd_mean = 0.0087 - 0.0216*self.alfa + 0.4*self.alfa**2 # TODO: This depends on empirical data based on naca0012. Get the relation from the polar, instead. Furthermore, this needs to satisfy Fig3-1 for naca0012(prolly).
        self.hover_power = thrust**1.5 / (self.tip_loss * np.sqrt(2*density * self.rotor_disk_area)) + density * self.num_blades * self.chord * (self.r**4) * (self.omega**3) * self.cd_mean / 8 
        self.hover_horsepower = self.hover_power * 0.00134102209
        self.cp = self.hover_power / (density * np.pi * self.r**5 * self.omega**3)
        self.merit = 0.707 * self.ct**1.5 / self.cp
        self.merit_max = 0.707 * self.ct**1.5 / (self.ct**1.5 / (np.sqrt(2) * self.tip_loss) + self.solidity * self.cd_mean / 8)
        print("HP:", self.hover_horsepower, "| Coefs:", self.ct, self.cp, "| Merits:", self.merit, self.merit_max, self.merit / self.merit_max, "| B:", self.tip_loss)

    def forward_flight(self, velocity, density=1.225, body="mi-8"):
        print("\nForward flight conditions.")
        if not self.is_hovered: 
            print("Running hover calculations for level blade first...")
            self.hover(0)
        advance_ratio = velocity * np.cos(self.alfa) / (self.omega * self.r)
        v_prime = np.sqrt((velocity - self.hover_induced_vel * np.sin(self.alfa))**2 + (self.hover_induced_vel * np.cos(self.alfa))**2) # Pythagorean of free-stream velocity and induced velocity.
        thrust_induced = density * self.rotor_disk_area * v_prime * 2 * self.hover_induced_vel
        power_induced = thrust_induced * (self.hover_induced_vel - v_prime * np.sin(self.alfa)) # TODO: Power too high.

        power_profile = (density * self.num_blades * self.chord * self.r**4 * self.omega**3 * self.cd_mean / 8) * (1 + 4.65*advance_ratio**2)
        thrust_profile = power_profile / self.r

        body_drag = density * velocity**2 * utils.get_flat_plate_area(body) / 2 # Flat plate area is component_reference_area * drag_coefficient for the body.
        power_parasite = body_drag * velocity

        power_total = power_induced + power_profile + power_parasite
        horsepower_total = power_total * 0.00134102209
        print("Horsepower ### Induced:", power_induced*0.00134102209, "| Profile:", power_profile* 0.00134102209, "| Parasite:", power_parasite* 0.00134102209, "| Total:", horsepower_total)

        # TODO: Implement blade stall.
    
    def plot(self):
        print("\nPlot debug.")
        print("Ct/sigma:", self.ct / self.solidity)
        
    
    
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
    rotor.hover(7)
    rotor.forward_flight(20)
    rotor.plot()