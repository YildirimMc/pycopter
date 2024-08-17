from .pycopterui import Ui_pycopter

class InputChecker():
    def __init__(self, ui: Ui_pycopter):
        self.ui = ui
        
        self.error_message = "" # If functions return false, print this on the interface.

    def airfoil_checker(self):
        airfoil = self.ui.airfoilText.toPlainText().lower()

        if airfoil[:4] != "naca": 
            self.print("Airfoil must be a NACA profile.") 
            return False
        elif 4 <= len(airfoil[4:]) <= 6:
            try:
                _ = int(airfoil[4:])
            except ValueError:
                self.print("\nERROR - Invalid airfoil. Only 4, 5, and 6 digits NACA profiles are supported.\n")
                return False
        else:
            self.print("ERROR - Invalid airfoil. Only 4, 5, and 6 digits NACA profiles are supported.")
            return False
        
        return True
