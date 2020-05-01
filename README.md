# 3D-Motion-Estimator
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/motion-estimator-3d/blob/master/LICENSE)

## Overview

This project estimates 3-D motion to implement visual odometry. Motion estimating is implemented on an Oxford dataset that can be found [*here*](https://drive.google.com/file/d/12Ir2kZ3kRgCe8vqaedT-dlpj-Z8vqhPO/view?usp=sharing).

The output videos can be viewed from [*here*](https://drive.google.com/open?id=1Ew3KA2jCv9skvMRkmsExJqWqKfWjQpm2).

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Numpy, OpenCV-Python

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3 python3-tk
sudo apt install python3-pip
pip3 install numpy 
pip3 install opencv-python==3.4.2.16
pip3 install opencv-contrib-python==3.4.2.16
```

- Check if your system successfully installed all the dependencies
- Open terminal using ```Ctrl+Alt+T``` and enter ```python3```.
- The terminal should now present a new area represented by ```>>>``` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ```Ctrl + Z``` if an error pops up while
running the below commands)

```
import tkinter
import numpy
import cv2
import scipy
```
