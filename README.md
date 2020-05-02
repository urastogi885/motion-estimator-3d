# 3D-Motion-Estimator
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/motion-estimator-3d/blob/master/LICENSE)

## Overview

This project estimates 3-D motion to implement visual odometry. Motion estimating is implemented on an Oxford dataset that can be found [*here*](https://drive.google.com/file/d/12Ir2kZ3kRgCe8vqaedT-dlpj-Z8vqhPO/view?usp=sharing).

The output videos can be viewed from [*here*](https://drive.google.com/open?id=1Ew3KA2jCv9skvMRkmsExJqWqKfWjQpm2).

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: numpy, opencv-python, scipy, random, glob, sys, os

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3 python3-tk
sudo apt install python3-pip
pip3 install numpy, scipy, random, glob
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
import cv2
import numpy, scipy, glob, random
```
## Run


- Download each of the dataset mentioned in the [*Overview Section*](https://github.com/urastogi885/motion-estimator-3d#overview).
- It is recommended that you save the dataset within outer-most directory-level of the project otherwise it will become 
too cumbersome for you to reference the correct location of the file.
- Using the terminal, clone this repository and go into the project directory, and run the main program:

```
https://github.com/urastogi885/motion-estimator-3d
cd motion-estimator-3d/Code
python3 visual_odometry.py dataset_location model_location
```

- If you have a compressed version of the project, extract it, go into project directory, open the terminal by 
right-clicking on an empty space, and type:

```
cd Code/
python3 visual_odometry.py dataset dataset_location output_location select_roi
```
- For instance:
```
python3 visual_odometry.py ../Oxford_dataset/stereo/centre/ ../Oxford_dataset/model
```

- For further documentation on the input arguments, refer 
[*main.py*](https://github.com/urastogi885/lucas-kanade-tracker/blob/master/Code/visual_odometry.py)
