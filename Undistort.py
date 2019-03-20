"""
Input the image and correction factors mtx and dist obtain by Calibration.py and get_calibration_factors.py
Output the corrected image without the natural lense distortion.

For comparison, uncomment the plotting functions at the end.
"""

import Calibration
import ColorSpaces
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle


def undistort(img, mtx, dist):
	# 1. TURN TO GRAYSCALE
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 2.USING THE CALIBRATED OBJPOINTS AND IMGPOINTS FROM THE CHECKERBOARD
	undist = cv2.undistort(img, mtx, dist, None, mtx)

	return undist