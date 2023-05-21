[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10789844&assignment_repo_type=AssignmentRepo)
# PROJECT TITLE

CIT 128 Student Directed Project

## Student Info
Jared Mendoza
37902
2nd semester
2023

## Program Description

My program takes the user's web camera input and runs that input frame by frame through a model that will be trained to detect humans, dogs, and cats. It will count how many of each it saw and display those numbers at the end. If there aren't any categories mentioned in the frame, then the program will show "nothing detected." 
### Video Demonstration

Add a Link to your video demonstration

### Install Instructions

Add any install instructions, if needed. This includes how to install included modules or libraries as well as configurations. You may remove this section if no special instructions are required.

## Software Engineering

Describe the software engineering techniques used for the design and development of this program.

## Testing Script

Describe the testing process using paragraphs and numbered bullet lists how to manually test the software here. 

## Directions and Grading Rubric

To review the project directions or update the grading rubric review the [DIRECTIONS.md](DIRECTIONS.md) file.

Credits:  MobileNet SSD 21 classes

Youtube: https://youtu.be/zQ_E622FRc8

Run instructions:
Version: python 3.10
Using "pip install" download the following packages: 
import os
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import keyboard
import threading
import logging
from queue import Queue

for Mac Users: 
Using "pip install" download the following packages: 
import os
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import keyboard
import threading
import logging
from queue import Queue

Open Terminal:
cd desktop 
ls 
sudo python3 Mac_version.py

FAQ: 
Q: I get the error no module named cv2 even after i downloaded it through pip.
A: try pip3 install opencv-python

Q: I get the error "no frame captured" when i run the program

A: make sure nothing is already using your webcam including screen recording software or a previous run, make sure your camera / wires are not damaged or that they are not in a USB hub, connect the cameras directly into your computer as your usb hub may not have enough bandwith to transport the information, make sure only one camera is connected, finally, try running smaller perameters. 

Q: I get the error "no protxtfile found" or "no model files found" or both
A: download the files from the directory here under "Mac_version": https://github.com/RHC-CIT-128-SP23/project-steezystudent 
Alternativly: 
Protxtfile: https://gist.github.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f
Model file: https://developer.qualcomm.com/sites/default/files/docs/snpe/convert_mobilenetssd.html#mobilenetssd_conversion_caffe






