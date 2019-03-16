#Gradients and color spaces
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = [0, 255]):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    abs_gradient_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(abs_gradient_direction)
    binary_output[(abs_gradient_direction >= thresh[0]) & (abs_gradient_direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output


def gray_gradients(img, thresh = (0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	binary_gray = np.zeros_like(gray)
	binary_gray[(gray > thresh[0]) & (gray <= thresh[1])] = 1
	
	plt.figure(1)
	#regular
	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)


	plt.subplot(222)
	plt.title('gray')
	plt.imshow(gray, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary gray')
	plt.imshow(binary_gray, cmap = 'gray')

	plt.show()

def RGB_gradients(img, thresh = (0, 255)):
	#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	R = img[:,:,0]
	G = img[:,:,1]
	B = img[:,:,2]

	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('R')
	plt.imshow(R, cmap = 'gray')

	plt.subplot(223)
	plt.title('G')
	plt.imshow(G, cmap = 'gray')

	plt.subplot(224)
	plt.title('B')
	plt.imshow(B, cmap = 'gray')

	plt.show()

def R_gradients(img, thresh = (0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	R = img[:,:,0]
	binary_R = np.zeros_like(R)
	binary_R[(R > thresh[0]) & (R <= thresh[1])] = 1

	"""
	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('R')
	plt.imshow(R, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary R')
	plt.imshow(binary_R, cmap = 'gray')

	plt.show()
	"""
	return binary_R

def G_gradients(img, thresh = (0, 255)):
	#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	G = img[:,:,1]
	binary_G = np.zeros_like(G)
	binary_G[(G > thresh[0]) & (G <= thresh[1])] = 1
	"""
	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('G')
	plt.imshow(G, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary G')
	plt.imshow(binary_G, cmap = 'gray')

	plt.show()
	"""
	return binary_G

def B_gradients(img, thresh = (0, 255)):
	#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	B = img[:,:,2]
	binary_B = np.zeros_like(B)
	binary_B[(B > thresh[0]) & (B <= thresh[1])] = 1
	"""
	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('B')
	plt.imshow(B, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary B')
	plt.imshow(binary_B, cmap = 'gray')

	plt.show()
	"""

def HLS_gradients(img, thresh = (0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	H = hls[:,:,0]
	L = hls[:,:,1]
	S = hls[:,:,2]

	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('H')
	plt.imshow(H, cmap = 'gray')

	plt.subplot(223)
	plt.title('L')
	plt.imshow(L, cmap = 'gray')

	plt.subplot(224)
	plt.title('S')
	plt.imshow(S, cmap = 'gray')

	plt.show()

def H_gradients(img, thresh = (0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	H = hls[:,:,0]
	binary_H = np.zeros_like(H)
	binary_H[(H > thresh[0]) & (H <= thresh[1])] = 1

	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('H')
	plt.imshow(H, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary H')
	plt.imshow(binary_H, cmap = 'gray')

	plt.show()

def L_gradients(img, thresh = (0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	L = hls[:,:,1]
	binary_L = np.zeros_like(L)
	binary_L[(L > thresh[0]) & (L <= thresh[1])] = 1
"""
	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('L')
	plt.imshow(L, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary L')
	plt.imshow(binary_L, cmap = 'gray')

	plt.show()
"""
def S_gradients(img, thresh = (0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	S = hls[:,:,2]
	binary_S = np.zeros_like(S)
	binary_S[(S > thresh[0]) & (S <= thresh[1])] = 1
	"""
	plt.figure(1)

	plt.subplot(221)
	plt.title('original')
	plt.imshow(img)

	plt.subplot(222)
	plt.title('S')
	plt.imshow(S, cmap = 'gray')

	plt.subplot(223)
	plt.title('binary S')
	plt.imshow(binary_S, cmap = 'gray')

	plt.show()
	"""
	return binary_S