## Getting Started

This program is used to approximate final performance characteristics of a rotor design. Basic mathematical methodoliges such as *blade element theory* and *momentum theory* are used. Numerical simplicity of the algorithm allows the user to quickly assess the rotor. The outputs of the program are easy enough to be interpretable by anyone, as well as sophisticated enough for the experienced users.

Program uses *XFOIL* (link here) to generate polar data for the given profile instead of keeping a polar database. Users also can use their own polar data.

## Tutorial

#### Start the Graphical User Interface

To start the GUI, simply run the main.py, or use the release executable. The main window consists of 4 main areas. In *Rotor Parameters*, the main properties of the rotor can be defined. In *Hover and Forward Flight Parameters*, the flight conditions can be further defined. The big emtpy area on the right is reserved for rendering plots. Finally, the *outputs* zone can be found below, where important results will be shown.

#### Initialize Your Rotor

The default values when the program is launched are for the *Mil Mi-8* Russian medium transport helicopter.

*Airfoil*: Must be text that starts with "naca" followed by 4, 5, or 6 digits that designate the profile. <br />
*Generate New Polars*: The program doesn't delete the last session's polar data. The user doesn't need to generate new polars if the same profile will be used. <br />
*Number of Blades*: Number of blades on a single rotor. Multiple rotors are not yet supported. However, since the inter-blade disturbances are not calculated for, the user can simply write the total number of blades here to mimic a multi-rotor design. E.g. 2 rotors with 3 blades each  (such as in Chinook) can be represented by 6 blades here. <br />
*Blade Chord*: The blade chord length in meters. <br />
*Rotor Diameter*: Rotor disk diameter in meters. <br />
*Tip Speed Mach*: Rotor blades' tip mach speed during hover. For a helicopter rotor, the standard is between 0.55 - 0.65. <br />
*Washout*: Amount of blade twist from blade root to tip in degrees. This twist comes with the blade and is designed by the manufacturer. <br />
*Rotor Root Cutout*: In the top-down view of a helicopter rotor, there exists a negative disk area before the blades begin. Where the pylon/rotorhead is. This area doesn't contribute to lift and must be removed from the main rotor's disk area. <br />

#### Calculate for Hover and Forward Flight Conditions

Now that we have initialized our designed rotor, we can see how it behaves under(or above) the weight of the fuselage.

*Gross Weight*: Kg gross weight of the helicopter in which the calculations are needed. Including fuel.
*Density*: Both hover and forward flight take place at constant altitude. Density changes by the altitude. Or by the planet.
*Transmission Loss*: ...

Hover conditions can be calculated now. Blade twists(collective pitch) from 0° to 15° are considered until the necessary lift for the given gross weight is satisfied. In case the output twist equals 15.0°, it is possible that the necessary lift was not satisfied and the program stopped searching. Always check "thrust" in the output as well and see that it is higher than the gross weight. 

*Velocity*: Forward flight speed in km/h.
*Flat Plate Area*: Equivalent flat plate area for the fuselage in square meters. This is used in forward flight conditions. Can be directly found online. Or calculated by component_reference_area * fuselage_drag_coefficient. Code includes a method to calculate the component_reference_area accesible in pycopter/utils.py.

Forward Flight conditions can be calculated now.

For Range

*Fuel Capacity*: The amount of fuel(in gross weight) that is loaded on the aircraft or the battery capacity. <br />
*Specific Fuel Consumption*: of single turboshaft engine. Can be found online.

#### Plots

A very useful aspect of the program is the *Plots* section. Simply select the desired plot and generate. 

## Results Verification

To be implemented.
