"""Main module."""

import numpy as np
import matplotlib.pyplot as plt

import utils

class Rotor():
    def __init__(self, num_blades, chord, rotor_diameter, rpm, cl, cd, tip_loss=0.97, rotor_root_cutout=0.2): # TODO: Add selectables profiles and explanations while simplifying the param names.
        self.num_blades = num_blades
        self.chord = chord
        self.R = rotor_diameter
        self.rpm = rpm
        self.cl = cl
        self.cd = cd
        self.tip_loss = tip_loss
        self.rotor_root_cutout = rotor_root_cutout

        self.r = self.R / 2
        self.omega = self.rpm * np.pi / 30
        self.tip_speed = self.omega * self.r
        self.rotor_disk_area = np.pi * self.r**2 - np.pi * (self.r * self.rotor_root_cutout)**2
        self.solidity = num_blades * chord / (np.pi * self.R)
        self.a = 5.73 # TODO: d_cl/d_alfa. Slope of cl vs alfa. (From Xfoil.)

        self.induced_vel = 1 # m/s. Dummy initialization value.

    def calculate(self, density=1.225, theta=8, tip_speed=0.6):
        # Hover
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

        """DEBUG"""
        print("T&HP:", self.thrust, self.horsepower, "Coefs:", self.ct, self.cp, self.cq, "Merits:", self.M, self.M_actual, self.FMR, "B =", self.tip_loss_2)

    def simulate(self, theta, Z=100, density=1.225, n=10): # TODO: Implement xfoil reading in utils.
        self.induced_vel = 0
        while True:
            self.phi = np.arctan(self.induced_vel / (self.omega * self.r))
            self.alfa = theta - self.phi
            temp = np.sqrt(self.num_blades * self.chord * self.r * self.omega**2 * (self.cl * np.cos(self.phi) - self.cd * np.sin(self.phi)) / (8*np.pi))
            if np.abs(self.induced_vel - temp) < 1e-8:
                break
            self.induced_vel = temp
        
        self.thrust_iter = 0
        for dr in np.linspace(0,self.r, n): # 10 step blade element.
            self.thrust_iter += density * np.pi * self.r * dr * self.induced_vel**2 * 2

        self.thrust_ge = self.thrust_iter / (1 - (self.R**2 / Z**2))
        print("vi =", self.induced_vel, "Thrust =", self.thrust_iter, "With GE = ", self.thrust_ge)
    

    def plot(self, args, kwargs):
        # TODO: args. If two ys(y1 and y2), then twin plot, else normal plot.
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('x-var')
        ax1.set_ylabel('Thrust [N]', color=color)
        ax1.plot(self.tip_speed, args.T, color=color, label="Thrust [N]")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Watts', color=color)
        ax2.plot(self.tip_speed, self.get_horsepower(args.density), color=color, label="Power")
        ax2.tick_params(axis='y', labelcolor=color)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.grid()
        return fig
    
    
class RotorImperial(Rotor):
    def __init__(self, a):
        pass
        print(a) # TODO: Test the imperial version


if __name__ == "__main__":
    rotor = Rotor(4, 0.5, 16, 500, 0.8, 0.02)
    rotor.calculate(1.293)
    rotor.simulate(8)