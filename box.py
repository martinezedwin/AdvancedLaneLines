"""
box.py is used to develop the pipeline that will eventually be used in videos.py to work on videos
on images first.
"""

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
fname = 'test_images/straight_lines1.jpg'
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#UNIDSTORT IMAGE

undist = Undistort.undistort(img, mtx, dist)

#plt.imshow(undist)
#plt.show()

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

#plt.imshow(color_combined, cmap = 'gray')
#plt.show()

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
#region = cv2.polylines(img, vertices_polylines,True, 255, 3)

#plt.imshow(region)
#plt.show()



h, w = img.shape[:2]

src = vertices
dst = np.array([[w, h], [0, h], [0, 0], [w, 0]], dtype = np.float32)

warped = Unwarp.unwarp(color_combined, src, dst)

#plt.imshow(warped, cmap = 'gray')
#plt.show()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#5. Detect lane pixels and fit to find the lane boundary.

histogram = np.sum(warped[warped.shape[0]//2:,:], axis = 0)
#plt.plot(histogram)
#plt.show()


out_img, left_fit, right_fit = FindPix.fit_polynomial(warped)

#print(out_img)
#plt.imshow(out_img)
#plt.show()



# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
result, left_fitx, right_fitx, ploty= FindPix.search_around_poly(warped, left_fit, right_fit)

#plt.imshow(result)
#plt.show()

#Fill in the lane between the detected lane lines
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

f = cv2.fillPoly(color_warp, np.int_([pts]), (0,100, 0))
#plt.imshow(f, cmap = 'gray')
#plt.show()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#6. Determine the curvature of the lane and vehicle position with respect to center.

left_curverad, right_curverad = FindPix.measure_curvature_pixels(warped)
#print(left_curverad)
#print(right_curverad)

left_curverad_m, right_curverad_m = FindPix.measure_curvature_real(warped)
#print(left_curverad_m)
#print(right_curverad_m)

curv = (left_curverad_m + right_curverad_m)/2

offset = FindPix.get_offset(warped, left_fit, right_fit)
#print(offset)

if offset > 0:
	case1 = 'Vehicle is ' + str(offset) + ' m to the right of center line'
	#print(case1)
elif offset < 0:
	case2 = 'Vehicle is ' + str(offset) + ' m to the left of center line'
	#print(case2)
else:
	case3 = 'Vehicle is in the center'
	#print(case3)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 7. Warp the detected lane boundaries back onto the original image.

dst_reverse = vertices
src_reverse = np.array([[w, h], [0, h], [0, 0], [w, 0]], dtype = np.float32)
reverse_warp = Unwarp.unwarp(f, src_reverse, dst_reverse)

#plt.imshow(reverse_warp)
#plt.show() 

final = cv2.addWeighted(img, 0.8, reverse_warp, 1, 0)
#plt.imshow(final)
#plt.show()


h = final.shape[0]
font = cv2.FONT_HERSHEY_DUPLEX
text = 'CURVE RADIUS: ' + '{:04.2f}'.format(curv) + ' (m)'
cv2.putText(final, text, (40,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
direction = ''
if offset > 0:
	direction = 'right'
elif offset < 0:
	direction = 'left'
abs_center_dist = abs(offset)
text = '{:04.3f}'.format(abs_center_dist) + ' (m) ' + direction + ' of center'
cv2.putText(final, text, (40,120), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

plt.imshow(final)
plt.show()





