import Calibration
import ColorSpaces
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle

"""
#Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"] #objpoints = dist_pickle["objpoints"]
dist = dist_pickle["dist"]

#read in test imageimage
fname = 'test_images/test2.jpg'
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#UNIDSTORT IMAGE
"""
def undistort(img, mtx, dist):
	# 1. TURN TO GRAYSCALE
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 2.USING THE CALIBRATED OBJPOINTS AND IMGPOINTS FROM THE CHECKERBOARD
	undist = cv2.undistort(img, mtx, dist, None, mtx)

	return undist

"""
a = undistort(img, mtx, dist)

plt.figure(1)
	#regular
plt.subplot(221)
plt.title('original')
plt.imshow(img)


plt.subplot(222)
plt.title('a')
plt.imshow(a, cmap = 'gray')

plt.show()
"""