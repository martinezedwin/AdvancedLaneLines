## Advanced Lane Finding
![Lanes Image](./output_images/straight_lines1_final_edit.jpg)

Final output video: https://www.youtube.com/watch?v=g0TxAKaxdNk

Goals
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Report
---
### Camera Calibration

The camera calibration consisted of obtaining chessboard images taken by a similar camera as the project vide. 

The code for this step is contained in lines #7 through #53 of the file called Calibration.py in conjuction with the get_calibration_factors.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

![Calibration Image](./camera_cal/calibration1.jpg)
![Calibration Image Undistorted](./output_images/calibration1_undistorted_edit.jpg)

### 1. Provide an example of a distortion-corrected image.

We will be using test_images/straight_lines1.jpg as an example for the rest of this tutorial:
![Test Imiage](./test_images/straight_lines1.jpg)


Using the distortion coefficients obtained in the camera calibration, we can correct the distortion of the image:
![Undistorted](./output_images/straight_lines1_undist_edit.jpg)

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

A conbination of color transforms and gradients were tested to see which would bring out the lane lines the best in binary images. A conbination of L from HLS color space and B from LAB color space where used as shown in lines 68 thorugh 83 of box.py

In the end the output looked something like this:
![Color transform and gradient](./output_images/straight_lines1_color_transform_and_gradients_edit.jpg)

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to perform a perspective transform or "Birds-eye view" a trapezoid was defined by four vertices that correspond to coordinates on the image.

These verteceis became the src. The destination points or dst where also define using the shape of the image. By using the Unwarp.unwarp funciton that contains hte cv2.getPersepectiveTransform() and cv2.warpPerspective() in lines 88 through 114 of box.py a birds-eye view was obtained.
![Birds-eye view](./output_images/striaght_lines_bird_eye_view_edit.jpg)


### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the pixels of a given binary image a historgram that shows high peaks where the lane lines are detected serves as a guide. 

For new undetected pixel images we start from the bottom of the image which would be closest to the car and form "windows" of specified size to continue seraching for lane lines along the image forward in the lane for each lane line.

For videos where we have already identified pixels in the previous frame, a focused search to where the prvious lane pixels where identified helps speed things up. This is shown in FindPix.py in find_lane_pils() and search_around_poly(). The pixels are then used to fit a polynomial by the fit_poly() functions.

![Lane pixels](./output_images/straight_lines1_lane_boxes_edit.jpg)

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using the radius curviture equation seen here: https://www.intmath.com/applications-differentiation/8-radius-curvature.php
and the coefficients calculated by the np.polyfit() function in the fit_polynomial_cr(), the curvature for both lane lines was calculated. (FindPix.py lines 287 thorugh 307)


Assuming that the camera is placed in the center of the car and thus the center of the image, and calculating the center of the lane relative to the image, you can calculate how far off the car is from the center of the lane to the right or left. This is done in FIndPix.py in the get_offset() function.

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![Final Output](./output_images/straight_lines1_final_edit.jpg)


---

## Pipeline (video)
### 1.  Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Here's a [link to my video result](./output_images/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part for me was finding a good combination of RGB, HLS color spaces and gradients to make it robust. This to me is a crusial part because the rest of the pipeline relies on highlighting the correct pixels (lane lines) while leaving anything extra out. By using more examples of shadows, colors, etc. it could be made more robust.

