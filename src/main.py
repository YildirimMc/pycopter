from pycopter import Rotor

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

    air_density = 1.293 # kg/m3 or slug/ft

    rotor = Rotor(num_blades, chord, rotor_diameter, rpm, cl, cd, tip_loss, rotor_disk_hole)
    thrust = rotor.get_thrust(air_density)
    power = rotor.get_power(air_density)
    horsepower = rotor.get_horsepower(air_density)

    print("Thrust:", thrust)
    print("Horsepower:", horsepower, f"({power} Watts)")




if __name__ == "__main__":
    main()