import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import cv2


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #inspiration for this section drawn from repo: upul/CarND-LaneLines-P1
    
    #initialize empty vectors to collect slopes and intercepts values for each side
    left_m =[]
    left_intercept = []
    
    right_m = []
    right_intercept = []
    
    #itterate through lines by slope and save x, y, and slope values
    #if slope is positive --> line is right side 
    #if slope is negative --> line is left side
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            m = (y2 - y1)/ (x2 - x1)
            
            if m > 0:
                right_m.append(m)
                right_intercept.append(y1 - m*x1)
            else:
                left_m.append(m)
                left_intercept.append(y1 - m*x1)
    
    #average slopes and intercepts
    left_average_m = np.average(left_m)
    left_average_intercept = np.average(left_intercept)
    right_average_m = np.average(right_m)
    right_average_intercept = np.average(right_intercept)
    
    # y bounderies for line segments
    y1 = img.shape[0]  #bottom of page
    y2 = 350 #325      #height of region masking
    
    #Find x1 and x2 from y1 and y2. x = (y-b)/m
    #use cv2.line to draw lines
    x1 = (y1 - left_average_intercept) / left_average_m
    x1 = int(x1)
    x2 = (y2 - left_average_intercept) / left_average_m
    x2 = int(x2)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    x1 = (y1 - right_average_intercept) / right_average_m
    x1 = int(x1)
    x2 = (y2 - right_average_intercept) / right_average_m
    x2 = int(x2)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, 
                            np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img