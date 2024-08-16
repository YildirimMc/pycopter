import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt

from pycopter import Rotor
from gui import Ui_pycopter
from gui.interface import Interface

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_pycopter()
    ui.setupUi(main_window)
    main_window.show()

    interface = Interface(ui, main_window)

    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()