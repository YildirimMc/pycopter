"""Main module."""

import numpy as np
import matplotlib.pyplot as plt

class Rotor():
    def __init__(self, num_blades, chord, rotor_diameter, rpm, cl, cd, tip_loss=0.95, rotor_disk_hole=0.1): # TODO: Add selectables profiles and explanations while simplifying the param names.
        self.num_blades = num_blades
        self.chord = chord
        self.rotor_diameter = rotor_diameter
        self.rpm = rpm
        self.cl = cl
        self.cd = cd
        self.tip_loss = tip_loss
        self.rotor_disk_hole = rotor_disk_hole

        self.rotor_radius = self.rotor_diameter / 2
        self.rpm_in_hz = self.rpm * np.pi / 30
        self.tip_speed = self.rpm_in_hz * self.rotor_radius
        self.rotor_disk_area = np.pi * self.rotor_radius**2 - np.pi * (self.rotor_radius * self.rotor_disk_hole)**2

    def get_thrust(self, density):
        return density * self.num_blades * self.chord * (self.rotor_diameter*0.5)**3 * self.rpm_in_hz**2 * self.cl / 6

    def get_power(self, density):
        return self.get_thrust(density)**1.5 / (self.tip_loss * np.sqrt(2*density*self.rotor_disk_area)) + density*self.num_blades*self.chord*(self.rotor_radius**4) * (self.rpm_in_hz**3) * self.cd / 8

    def get_horsepower(self, density):
        watts = self.get_power(density)
        return watts * 0.00134102209
    
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

        plt.grid()
        return plt

    
class RotorImperial(Rotor):
    def __init__(self, a):
        print(a) # TODO: Test the imperial version

