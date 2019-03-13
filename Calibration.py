#Camera calibration
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle

def image_calib(img, nx, ny):
	img = cv2.imread(img)

	#Convert to grayscale	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#print(image)

	for i in range(len(nx)):
		for j in range(len(ny)):
			ret, corners = cv2.findChessboardCorners(gray, (nx[i], ny[j]), None)
			#print(ret)
			objpoints = [] #3D points in real world space
			imgpoints = [] #2D points in image plane

			objp = np.zeros((ny[j]*nx[i], 3), np.float32)
			objp[:,:2] = np.mgrid[0:nx[i], 0:ny[j]].T.reshape(-1,2)

			#IF FOUND, DRAW CORNERS
			if ret == True:
				imgpoints.append(corners)
				objpoints.append(objp)
				#cv2.drawChessboardCorners(img, (nx[i], ny[j]), corners, ret)
				#plt.imshow(img)
				#plt.show()
				
				ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

				dist_pickle = {}
				dist_pickle['mtx'] = mtx
				dist_pickle['dist'] = dist
				pickle.dump(dist_pickle, open('calibration.p', 'wb'))

				undist = cv2.undistort(img, mtx, dist, None, mtx)
				#plt.imshow(undist, cmap = 'gray')
				#plt.show()
				offset = 100
				img_size = (gray.shape[1], gray.shape[0])

				src = np.float32([corners[0], corners[nx[i]-1], corners[-1], corners[-nx[i]]])

				dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])

				M = cv2.getPerspectiveTransform(src, dst)

				warped = cv2.warpPerspective(undist, M, img_size)
				#plt.imshow(warped, cmap = 'gray')
				#plt.show()


