from pycopter import Rotor

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Example rotor parameters.
    num_blades = 6
    chord = 0.025 # [m]
    rotor_diameter = 0.6 # [m]
    rpm = 2500 
    cl = 1.1
    cd = 0.04
    tip_loss=0.95 # [0-1]
    rotor_disk_hole=0.1 # [0-1]

    air_density = 1.225 # kg/m3 or slug/ft


    # rotor = Rotor("naca0012", 2, .53, 14.63, .6, -4, 0.01, False)
    # rotor.hover(4300)
    rotor = Rotor(new_polar=False)
   
    # rotor.forward_flight(20)
    
    sfc = 0.35
    transmission_loss = 1.12 # ~0.9
    plot_range_endurance_vs_velocity(rotor, sfc, transmission_loss, 12000, 1900)
    # plot_ctsigma_to_maxmerit(rotor, 12500)
    # plot_nfs_to_dvr(rotor, 12500)
    # plot_ground_effect(rotor, 12500)

    

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

def plot_range_battery(rotor: Rotor, gross):
    pass

if __name__ == "__main__":
    main()