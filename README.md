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






