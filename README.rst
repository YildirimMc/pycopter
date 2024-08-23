========
PyCopter
========


.. image:: https://img.shields.io/pypi/v/pycopter.svg
        :target: https://pypi.python.org/pypi/pycopter

.. image:: https://img.shields.io/travis/YildirimMc/pycopter.svg
        :target: https://travis-ci.com/YildirimMc/pycopter

.. image:: https://readthedocs.org/projects/pycopter/badge/?version=latest
        :target: https://pycopter.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




PyCopter - A model-based rotary-wing platform design package.

Getting Started
--------

This program is used to approximate final performance characteristics of a rotor design. 
A computational tool designed for quick rotor design exploration 
and performance examining using fundamental aerodynamic theories such as Blade Element Theory and Momentum 
Theory applied for rotorcrafts is implemented. The program calculates essential aerodynamic 
metrics such as rotor lift, drag, power consumption and can estimate flight ranges. The outputs of the program are easy 
enough to be interpretable by anyone, as well as sophisticated enough for the experienced users.

Program uses *XFOIL* (link here) to generate polar data for the given profile instead of keeping a polar database. Users also can use their own polar data.

The tool also includes a graphical user interface module.

* Free software: Apache Software License 2.0
* Documentation: https://pycopter.readthedocs.io. (To be implemented.)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
