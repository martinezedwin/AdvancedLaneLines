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

#1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#1. DONE using get_calibration_factors.py and Calibration.py saved to calibration.p

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#2. Apply a distortion correction to raw images.

#Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"] #objpoints = dist_pickle["objpoints"]
dist = dist_pickle["dist"]

#read in test imageimage
fname = 'test_images/straight_lines2.jpg'
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#UNIDSTORT IMAGE

undist = Undistort.undistort(img, mtx, dist)

"""
plt.figure(1)
	#regular
plt.subplot(221)
plt.title('original')
plt.imshow(img)


plt.subplot(222)
plt.title('undist')
plt.imshow(undist, cmap = 'gray')

plt.show()
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#3. Use color transforms, gradients, etc., to create a thresholded binary image.



# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements


# Apply each of the thresholding functions
gradx = ColorSpaces.abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = ColorSpaces.abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = ColorSpaces.mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(50, 200))
#dir_binary = ColorSpaces.dir_threshold(undist, sobel_kernel=ksize, thresh=(1.4, np.pi/2))#np.pi/2))

combined = np.zeros_like(mag_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1))] = 1    # & (dir_binary == 1)

#ColorSpaces.gray_gradients(undist, thresh = (100, 130))
#ColorSpaces.RGB_gradients(undist, thresh = (100, 130))
R = ColorSpaces.R_gradients(undist, thresh = (180, 255)) #Useful for test img, 2, 3, 6, s1, s2
G = ColorSpaces.G_gradients(undist, thresh = (180, 255)) #2, 3, 6, s1, s2


#ColorSpaces.HLS_gradients(undist, thresh = (100, 130))
#ColorSpaces.H_gradients(undist, thresh = (100, 130))
#ColorSpaces.L_gradients(image, thresh = (100, 130))
S = ColorSpaces.S_gradients(undist, thresh = (130, 255))

color_combined = np.zeros_like(mag_binary)
color_combined[((combined == 1)) | ((S == 1))] = 1    #& (R == 1)


"""
plt.figure(1)
	#regular
plt.subplot(221)
plt.title('original')
plt.imshow(img)


plt.subplot(222)
plt.title('color_combined')
plt.imshow(color_combined, cmap = 'gray')
plt.show()
"""

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#4. Apply birds eye view
vertical_middle = color_combined.shape[0]/2
horizontal_middle = color_combined.shape[1]/2
BR_h = horizontal_middle + 700
BR_v = color_combined.shape[0]
BL_h = horizontal_middle - 700
BL_v = BR_v
TL_h = horizontal_middle - 100
TL_v = vertical_middle + 100
TR_h = horizontal_middle + 100
TR_v = TL_v
#print(BR)
#vertices = np.array([[BR_h, color_combined.shape[0]], [0, color_combined.shape[0]], [560, 460], [720, 460]], dtype=np.int32)
vertices_polylines = np.array([[(BR_h, BR_v), (BL_h,BL_v), (TL_h, TL_v), (TR_h, TR_v) ]], dtype=np.int32)
vertices = np.array([[(BR_h, BR_v), (BL_h,BL_v), (TL_h, TL_v), (TR_h, TR_v) ]], dtype=np.float32)
region = cv2.polylines(img, vertices_polylines,True, 255, 3)
"""
plt.imshow(region)
plt.show()
"""


h, w = img.shape[:2]

src = vertices
dst = np.array([[w, h], [0, h], [0, 0], [w, 0]], dtype = np.float32)

warped = Unwarp.unwarp(color_combined, src, dst)

"""
plt.figure(1)
	#regular
plt.subplot(221)
plt.title('original')
plt.imshow(color_combined, cmap = 'gray')


plt.subplot(222)
plt.title('warped')
plt.imshow(warped, cmap = 'gray')
plt.show()

"""

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#5. Detect lane pixels and fit to find the lane boundary.

histogram = np.sum(warped[warped.shape[0]//2:,:], axis = 0)
"""
plt.figure(1)
	#regular
plt.subplot(221)
plt.title('warped')
plt.imshow(warped, cmap = 'gray')


plt.subplot(223)
plt.title('histogram')
plt.plot(histogram)
plt.show()
"""

out_img, left_fit, right_fit = FindPix.fit_polynomial(warped)

#print(out_img)
#plt.imshow(out_img)
#plt.show()



# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
result = FindPix.search_around_poly(warped, left_fit, right_fit)

# View your output
plt.imshow(result)
plt.show()
