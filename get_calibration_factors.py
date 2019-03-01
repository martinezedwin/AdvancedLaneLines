import Calibration
import ColorSpaces
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

###########################################RUN THIS ONLY ONCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###
#Get camera calibration. Correct distortion of images on the camera_cal/calibration images
#This will get the correction factors for the camera lense and safe the mto calibration.p

####
a = range(1,21)
for i in range(len(a)):
	nx = [6, 8, 9]#[4, 5, 6, 7, 8, 9]
	ny = [5, 6, 7]
	fname = 'camera_cal/calibration' + str(a[i]) + '.jpg'
	Calibration.image_calib(fname, nx, ny)