import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt

from pycopter import Rotor
from gui import Ui_pycopter
from gui.interface import Interface

def main():
    app = QtWidgets.QApplication(sys.argv)
    pycopter = QtWidgets.QMainWindow()
    ui = Ui_pycopter()
    ui.setupUi(pycopter)
    pycopter.show()

    interface = Interface(ui)

    # rotor = Rotor("naca0012", 2, .53, 14.63, .6, -4, 0.01, False)
    # rotor.hover(4300)
    
    # rotor.forward_flight(20)
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()