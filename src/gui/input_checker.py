from .pycopterui import Ui_pycopter

class InputChecker():
    """
    Spawn this class and use its functions to check GUI inputs. Functions return False if an issue was detected.
    In that case, property 'error_message' carries the reason. 

    In initial version, error checker is only implemented for a UI.

    Methods
    -------
    airfoil_checker() -> bool
        Checks the airfoil string input for correctness and returns True.
    """
    def __init__(self, ui: Ui_pycopter):
        """Pass in UI that contains the user inputs.
        
        Parameters
        ----------
        ui : PyQt UI class generated by pyuic5
        """
        self.ui = ui
        
        self.error_message = "" # If functions return false, print this on the interface.

    def airfoil_checker(self):
        """Checks airfoil input for proper NACA naming and returns False if an issue was detected. Returns True otherwise."""
        airfoil = self.ui.airfoilText.toPlainText().lower()

        if airfoil[:4] != "naca": 
            self.error_message("ERROR - Airfoil must be a NACA profile.") 
            return False
        elif 4 <= len(airfoil[4:]) <= 6:
            try:
                _ = int(airfoil[4:])
            except ValueError:
                self.error_message("ERROR - Invalid airfoil. Only 4, 5, and 6 digits NACA profiles are supported.")
                return False
        else:
            self.error_message("ERROR - Invalid airfoil. Only 4, 5, and 6 digits NACA profiles are supported.")
            return False
        
        return True
