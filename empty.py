import Calibration
import ColorSpaces
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle
import Undistort
import Unwarp
import FindPix
from IPython.display import HTML

"""
L = ColorSpaces.R_gradients(undist, thresh = (180, 255)) #Useful for test img, 2, 3, 6, s1, s2
A = ColorSpaces.G_gradients(undist, thresh = (130, 255)) #2, 3, 6, s1, s2
B = ColorSpaces.B_gradients(undist, thresh = (200, 255))
"""

fname = 'test_images/straight_lines1.jpg'
img = cv2.imread(fname)

L = ColorSpaces.LAB_gradients(img, thresh = (000, 255))