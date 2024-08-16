from .pyrotorui import Ui_pycopter
from pycopter import Rotor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import json

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QFileDialog

class MplCanvas(FigureCanvasQTAgg):
    """Used for drawing figures in the dedicated area in pycopter GUI."""
    def __init__(self, figure):
        super().__init__(figure)

class Interface():
    """This class defines the functionalities of the pycopter's main window."""
    def __init__(self, ui: Ui_pycopter, main_window: QMainWindow):
        self.ui = ui
        self.main_window = main_window

        self.ui.initRotorBtn.clicked.connect(self.init_rotor)
        self.ui.calcHoverBtn.clicked.connect(self.calculate_hover)
        self.ui.calcForwardFlightBtn.clicked.connect(self.calculate_forward_flight)
        self.ui.generatePlotBtn.clicked.connect(self.generate_plot)
        self.ui.actionNew.triggered.connect(self.action_new)
        self.ui.actionSave.triggered.connect(self.action_save)
        self.ui.actionSaveAs.triggered.connect(self.action_save_as)
        self.ui.actionLoad.triggered.connect(self.action_load)
        self.ui.actionClear_outputs.triggered.connect(self.action_clear_outputs)

        self.mpl_layout = QVBoxLayout()
        self.ui.mpl_widget.setLayout(self.mpl_layout)

        self.plots_dict = {"Range, Endurance vs. Velocity": self.plot_range_endurance_vs_velocity, 
                           "Specific Range, LD vs. Velocity": self.plot_sr_ld_vs_velocity,
                           "Max FMR vs. Ct/σ": self.plot_ctsigma_to_maxmerit,
                           "Downwash Velocity Ratio vs. Normalized Flight Speed": self.plot_nfs_to_dvr,
                           "Ground Effect vs. Height": self.plot_ground_effect,
                           "Electric Range vs. Velocity": self.plot_range_electric,
                           "Forward Flight Powers vs. Velocity": self.plot_powers_vs_velocity}
        
        self.spinners_dict = {"airfoil": self.ui.airfoilText,
                              "num_blades": self.ui.numBladesSpin,
                              "chord": self.ui.bladeChordSpin,
                              "rotor_diam": self.ui.rotorDiameterSpin,
                              "tip_speed_mach": self.ui.tipSpeedMachSpin,
                              "washout": self.ui.washoutSpinner,
                              "root_cutout": self.ui.rotorRootCutoutSpinner,
                              "gross": self.ui.grossSpinner,
                              "density": self.ui.densitySpinner,
                              "transmission_loss": self.ui.transmissionLossSpinner,
                              "fuel_cap": self.ui.fuelCapSpinner,
                              "sfc": self.ui.sfcSpinner,
                              "battery_cap": self.ui.batteryCapSpinner,
                              "velocity": self.ui.velocitySpinner,
                              "fpa": self.ui.fpaSpinner,
                              "output": self.ui.textOutputWidget}
        
        self.save_path = ""

    def print(self, text):
        """Prints text into the dedicated area in GUI."""
        self.ui.textOutputWidget.appendPlainText(text)

    def init_rotor(self):
        """Implements the functionality of the 'Init Rotor' button."""
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
        self.ui.generatePlotBtn.setEnabled(True)
        
    def calculate_hover(self):
        """'Calculate Hover' button."""
        gross = self.ui.grossSpinner.value()
        density = self.ui.densitySpinner.value()

        self.print("\nCalculating Hover Conditions...")
        self.rotor.hover(gross, density)
        self.print(f"Theta = {self.rotor.theta}° | Induced Velocity= {self.rotor.hover_induced_vel:.3f}[m/s] | Thrust = {self.rotor.hover_thrust/9.81:.3f}[kg]")
        self.print(f"SHP Induced: {self.rotor.hover_power_induced*0.00134102209:.3f} | SHP Profile: {self.rotor.hover_power_profile*0.00134102209:.3f} | SHP Total: {self.rotor.hover_power_total*0.00134102209:.3f} | BHP: {self.rotor.hover_power_total*0.00134102209/self.ui.transmissionLossSpinner.value():.3f}")
        self.print(f"Coeffs: {self.rotor.ct}, {self.rotor.cp} | Merits: {self.rotor.merit}, {self.rotor.merit_max}, {self.rotor.merit / self.rotor.merit_max} | Tip Loss: {self.rotor.tip_loss}")
        self.print(f"Thrust in ground effect at 10 meters = {self.rotor.ige(self.rotor.hover_thrust, 10)/9.81} [kg]")

        self.ui.calcForwardFlightBtn.setEnabled(True)

    def calculate_forward_flight(self):
        """'Calculate Forward Flight' button."""
        velocity = self.ui.velocitySpinner.value() / 3.6 # Converting to m/s
        density = self.ui.densitySpinner.value() 
        flat_plate_area = self.ui.fpaSpinner.value()

        self.print("\nCalculating Forward Flight Conditions...")
        self.rotor.forward_flight(velocity, density, flat_plate_area)
        self.print(f"Alfa: {self.rotor.alfa} | Downwash Velocity Ratio: {self.rotor.downwash_velocity_ratio} | Check if {self.rotor.power_profile/(2*density*self.rotor.rotor_disk_area*self.rotor.r)} is smaller than {velocity**2 / 2} for horsepowers below.")
        self.print(f"SHP Induced: {self.rotor.power_induced*0.00134102209} | SHP Profile:  {self.rotor.power_profile*0.00134102209} | SHP Parasite: {self.rotor.power_parasite*0.00134102209} | HP Total: {self.rotor.horsepower_total}")
        
    def action_new(self):
        """'File' -> 'New' functionality."""
        for key, value in self.spinners_dict.items():
            value.clear() # Soft reference! Careful!

    def action_save(self):
        """'File' -> 'Save' functionality."""
        with open(self.save_path, 'w') as file:
            configuration_dict = {}
            for key, value in self.spinners_dict.items():
                if key == "airfoil" or key == "output":
                    configuration_dict[key] = value.toPlainText().lower()
                else:
                    configuration_dict[key] = value.value()
            json.dump(configuration_dict, file, indent=4)
        self.print(f"Configuration saved to: {self.save_path}\n")

    def action_save_as(self):
        """'File' -> 'Save As' functionality."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save configuration...",
            "pycopter",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        
        # Saving the file if the user selects one
        if file_path:
            with open(file_path, 'w') as file:
                configuration_dict = {}
                for key, value in self.spinners_dict.items():
                    if key == "airfoil" or key == "output":
                        configuration_dict[key] = value.toPlainText().lower()
                    else:
                        configuration_dict[key] = value.value()
                json.dump(configuration_dict, file, indent=4)
            self.print(f"Configuration saved to: {file_path}\n")
            self.save_path = file_path
            self.ui.actionSave.setEnabled(True)
        else:
            self.print("No file selected.\n")

    def action_load(self):
        """'File' -> 'Load' functionality."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load configuration...",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        
        # Loading the file if the user selects one
        if file_path:
            with open(file_path, 'r') as file:
                configuration_dict = json.load(file)
            for key, value in configuration_dict.items():
                if key == "airfoil" or key == "output":
                    self.spinners_dict[key].setPlainText(value)
                else:
                    self.spinners_dict[key].setValue(value)
            self.print(f"Configuration loaded.\n")
        else:
            self.print("No file selected.\n")

    def action_clear_outputs(self):
        self.ui.textOutputWidget.clear()
        
    def generate_plot(self):
        """'Generate Plot' button."""
        fig = self.plots_dict[self.ui.selectedPlotCombo.currentText()]()
        canvas = FigureCanvasQTAgg(fig)
        
        while self.mpl_layout.count():
            item = self.mpl_layout.takeAt(0)
        self.mpl_layout.addWidget(canvas)
        
    ##################### PLOT FUNCTIONS #####################
    """Plot functions calculate the data, create the figure, and return it to 'Generate Plot' function above."""

    def plot_powers_vs_velocity(self):
        """Creates the plot and returns the figure to 'Generate Plot' function."""
        gross = self.ui.grossSpinner.value()
        flat_plate_area = self.ui.fpaSpinner.value()
        density = self.ui.densitySpinner.value()
        self.ui.textOutputWidget.setPlainText

        self.rotor.hover(gross)
        test_velocities = np.linspace(10,80, 50)
        induced_powers = np.zeros((50))
        profile_powers = induced_powers.copy()
        parasite_powers = induced_powers.copy()
 
        for i, vel in enumerate(test_velocities):
            self.rotor.forward_flight(vel, density, flat_plate_area)
            induced_powers[i] = self.rotor.power_induced
            profile_powers[i] = self.rotor.power_profile
            parasite_powers[i] = self.rotor.power_parasite
        induced_powers /= 1000
        profile_powers /= 1000
        parasite_powers /= 1000
        total_powers = induced_powers + profile_powers + parasite_powers
        test_velocities *= 3.6
               
        fig, ax1 = plt.subplots()

        ax1.plot(test_velocities, induced_powers*1.34102209, color="blue", label="Induced")
        ax1.plot(test_velocities, profile_powers*1.34102209, color="orange", label="Profile")
        ax1.plot(test_velocities, parasite_powers*1.34102209, color="red", label="Parasite")
        ax1.plot(test_velocities, total_powers*1.34102209, color="black", label="Total")
        ax1.set_title("Forward Flight Powers")
        ax1.set_xlabel("Free Stream Velocity [km/hr]")
        ax1.set_ylabel("Shaft Horsepower [SHP]")
        fig.legend()
        ax1.grid()
        return fig

    def plot_range_endurance_vs_velocity(self):
        sfc = self.ui.sfcSpinner.value()
        transmission_loss = self.ui.transmissionLossSpinner.value()
        gross = self.ui.grossSpinner.value()
        fuel = self.ui.fuelCapSpinner.value()
        flat_plate_area = self.ui.fpaSpinner.value()
        density = self.ui.densitySpinner.value()

        self.rotor.hover(gross)
        lift = self.rotor.hover_thrust
        test_velocities = np.linspace(10,80, 50)
        effective_drags = np.zeros((50))
        total_powers = effective_drags.copy()

        for i, vel in enumerate(test_velocities):
            self.rotor.forward_flight(vel, density, flat_plate_area)
            effective_drags[i] = self.rotor.drag_induced + self.rotor.drag_profile + self.rotor.body_drag
            total_powers[i] = self.rotor.power_total
        total_powers /= 1000
        engine_powers = 1.13 * total_powers.copy() / (1 - transmission_loss) #0.13 goes to tail rotor
        test_velocities *= 3.6
        
        sfc = 1.13 * sfc / (1 - transmission_loss) # 1.13 for tail rotor

        ld = lift/effective_drags
        fuel_consumptions = engine_powers * sfc
        endurance = fuel / fuel_consumptions
        breguet_ranges = 366/sfc * ld * np.log(gross/(gross-fuel)) # 366 for km instead of n-miles
        
        fig, ax1 = plt.subplots()

        ax1.plot(test_velocities, endurance, color="red", label="Endurance")
        ax1.set_title("Range, Endurance - Velocity")
        ax1.set_xlabel("Free Stream Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")

        ax2 = ax1.twinx()
        ax2.plot(test_velocities, breguet_ranges, label="Range")
        ax2.set_ylabel("Range [km]")

        fig.legend()
        ax1.grid()
        return fig
        
    def plot_sr_ld_vs_velocity(self):
        sfc = self.ui.sfcSpinner.value()
        transmission_loss = self.ui.transmissionLossSpinner.value()
        gross = self.ui.grossSpinner.value()
        fuel = self.ui.fuelCapSpinner.value()
        flat_plate_area = self.ui.fpaSpinner.value()
        density = self.ui.densitySpinner.value()

        self.rotor.hover(gross)
        lift = self.rotor.hover_thrust
        test_velocities = np.linspace(10,80, 50)
        effective_drags = np.zeros((50))
        total_powers = effective_drags.copy()

        for i, vel in enumerate(test_velocities):
            self.rotor.forward_flight(vel, density, flat_plate_area)
            effective_drags[i] = self.rotor.drag_induced + self.rotor.drag_profile + self.rotor.body_drag
            total_powers[i] = self.rotor.power_total
        total_powers /= 1000
        engine_powers = total_powers.copy()
        engine_powers = 1.13 * engine_powers / (1 - transmission_loss) #0.13 goes to tail rotor
        test_velocities *= 3.6
        
        ld = lift/effective_drags
        fuel_consumptions = engine_powers * sfc
        specific_range = test_velocities / (fuel_consumptions) # km range / kg fuel

        fig, ax1 = plt.subplots()

        ax1.plot(test_velocities, specific_range, color="orange", label="Specific Range")
        ax1.set_title("Specific Range, Lift/Drag Ratio - Velocity")
        ax1.set_xlabel("Free Stream Velocity [km/hr]")
        ax1.set_ylabel("Specific Range [km/kgf], LD Ratio")

        ax2 = ax1.twinx()
        ax2.plot(test_velocities, ld, label="Lift/Drag Ratio")
        ax2.set_ylabel("Lift / Drag Ratio")

        fig.legend(loc="upper left")
        ax1.grid()
        return fig

    def plot_ctsigma_to_maxmerit(self):
        gross = self.ui.grossSpinner.value()
        self.rotor.hover(gross)
        
        ct_range = np.linspace(self.rotor.ct - self.rotor.ct/2, self.rotor.ct + self.rotor.ct/2, 50)
        merit_max = 0.707 * ct_range**1.5 / (ct_range**1.5 / (np.sqrt(2) * self.rotor.tip_loss) + self.rotor.solidity * self.rotor.cd_mean / 8)

        ct_sigma_range = ct_range / self.rotor.solidity

        fig, ax = plt.subplots()
        ax.plot(ct_sigma_range, merit_max)
        ax.set_title("Max Figure of Merit vs. Ct/σ")
        ax.set_xlabel("Ratio of Thrust Coefficient to Solidity")
        ax.set_ylabel("Maximum Figure of Merit")
        ax.grid()
        return fig

    def plot_nfs_to_dvr(self):
        gross = self.ui.grossSpinner.value()
        flat_plate_area = self.ui.fpaSpinner.value()
        density = self.ui.densitySpinner.value()

        self.rotor.hover(gross)
        velocities = np.linspace(0, 50, 50)
        downwash_velocity_ratios = np.zeros((50))
        for i, velocity in enumerate(velocities):
            self.rotor.forward_flight(velocity, density, flat_plate_area)
            downwash_velocity_ratios[i] = self.rotor.downwash_velocity_ratio
        normalized_flight_speeds = velocities / self.rotor.hover_induced_vel

        fig, ax = plt.subplots()
        ax.plot(normalized_flight_speeds, downwash_velocity_ratios, label="Forward Flight")
        ax.set_title("Wald's Equation")
        ax.set_xlabel("Normalized Flight Speed [V/v₀]")
        ax.set_ylabel("Downwash Velocity Ratio [v/v₀]")
        ax.legend()
        ax.grid()
        return fig

    def plot_ground_effect(self):
        gross = self.ui.grossSpinner.value()
        self.rotor.hover(gross)
        heights = np.linspace(5.65, 50, 50)
        thrusts_ige = np.zeros((50))
        for i, height in enumerate(heights):
            thrusts_ige[i] = self.rotor.ige(self.rotor.hover_thrust, height)

        fig, ax = plt.subplots()
        ax.plot(thrusts_ige/9.81, heights, label="Thrust")
        ax.set_title(f"Thrusts at {self.rotor.theta}° Collective")
        ax.axvline(self.rotor.hover_thrust/9.81, color="k", label="Base Thrust")
        ax.scatter(thrusts_ige[0]/9.81, 5.65, marker=(5,1), color="red", label="Landed Thrust")
        ax.set_xlabel("Thrust in Ground Effect [kg]")
        ax.set_ylabel("Rotor Height [m]")
        ax.grid()
        ax.legend()
        return fig
    
    def plot_range_electric(self):
        gross = self.ui.grossSpinner.value()
        transmission_loss = self.ui.transmissionLossSpinner.value()
        battery_capacity = self.ui.batteryCapSpinner.value()
        flat_plate_area = self.ui.fpaSpinner.value()
        density = self.ui.densitySpinner.value()

        self.rotor.hover(gross)
        test_velocities = np.linspace(10,80, 50)
        total_powers = np.zeros((50))

        for i, vel in enumerate(test_velocities):
            self.rotor.forward_flight(vel, density, flat_plate_area)
            total_powers[i] = self.rotor.power_total
        total_powers /= 1000
        motor_powers = 1.13 * total_powers.copy() / (1 - transmission_loss)
        test_velocities *= 3.6

        endurance = battery_capacity / motor_powers
        range = endurance * test_velocities

        fig, ax1 = plt.subplots()
        ax1.plot(test_velocities, endurance, label="Endurance")
        ax1.set_title("Electric Range and Endurance")
        ax1.set_xlabel("Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")

        ax2 = ax1.twinx()
        ax2.plot(test_velocities, range, label="Range", color="Orange")
        ax2.set_ylabel("Range [km]")
        ax1.grid()
        fig.legend()
        return fig
        