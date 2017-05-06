import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def store_image(img, img_dir, fname):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = img_dir + fname.split('/')[-1]
    cv2.imwrite(img_path, img)

def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #store every images with detected corners for post analyse and writeup
            img_dir = 'output_images/chessboard_corners/'
            img_path = img_dir + fname.split('/')[-1]
            store_image(img, img_dir, img_path)
    #calibrate camera with identified objpoints and imgpoints
    img_size = (img.shape[1], img.shape[0])
    return cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(150, 255), sx_thresh=(20, 200),l_thresh=(50,255)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
    return  combined_binary

def warp(img,src2dst=True):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[585, 460],[203, 720],[1127, 720],[695, 460]])
    dst = np.float32([[320, 0],[320, 720],[960, 720],[960, 0]])
    #print(src, dst)

    if src2dst:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)
    return warped

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def draw_lines(img, vertices):
    pts= np.int32([vertices])
    cv2.polylines(img, pts, True, (255,0,0),5)

def preprocess_img(img, mtx, dist):
    undist_img = undistort(img, mtx, dist)
    binary_image = pipeline(undist_img)
    transformed = warp(binary_image)
    return transformed

def detect_start(binary_warped, LL, RL, fname=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #plot
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    LL.update_params(left_fit, right_fit, leftx, lefty, ploty)
    RL.update_params(right_fit, left_fit, rightx, righty, ploty)

    if(fname!=None):
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(fname)

#next frame fit poly
def detect_next(binary_warped, LL, RL):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (LL.current_fit[0]*(nonzeroy**2) + LL.current_fit[1]*nonzeroy + LL.current_fit[2] - margin)) & (nonzerox < (LL.current_fit[0]*(nonzeroy**2) + LL.current_fit[1]*nonzeroy + LL.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (RL.current_fit[0]*(nonzeroy**2) + RL.current_fit[1]*nonzeroy + RL.current_fit[2] - margin)) & (nonzerox < (RL.current_fit[0]*(nonzeroy**2) + RL.current_fit[1]*nonzeroy + RL.current_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    LL.update_params(left_fit, right_fit, leftx, lefty, ploty)
    RL.update_params(right_fit, left_fit, rightx, righty, ploty)


def calculate_curverad(leftx, rightx, ploty):
    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(ploty)
    #print(y_eval)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print('left:', left_curverad, 'm', 'right:', right_curverad, 'm')
    return left_curverad, right_curverad

def calculate_position(pts_left, pts_right):
    # 10th value for x
    lane_middle = int((pts_right[0][10][0] - pts_left[0][10][0])/2.)+pts_left[0][10][0]
    #lane_middle = int((pts_right[10] - pts_left[10])/2.)+pts_left[10]

    if (lane_middle-640 > 0):
        leng = 3.66/2
        mag = ((lane_middle-640)/640.*leng)
        #head = ("Right",mag)
        return "Right", mag
    else:
        leng = 3.66/2.
        mag = ((lane_middle-640)/640.*leng)*-1
        #head = ("Left",mag)
        return "Left", mag

def line_base_pos(current_fit, ploty):
    y_eval = np.max(ploty)
    center_pos = 640
    line_pos = current_fit[0]*y_eval**2 + current_fit[1]*y_eval + current_fit[2]
    line_base_pos = (line_pos - center_pos)*3.7/600.0 #3.7 meters is about 600 pixels in the x direction

    # avoid negative distances, which would be the case for left lane
    # there is sure an more elegant way to handle this
    if line_base_pos < 0:
        line_base_pos *= (-1.0)

    return line_base_pos

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, name='ABC'):
        #Line name
        self.line_name = name
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #store detection miss here
        self.detection_miss = 0
        #rejection counter
        self.reject_counter = 0

    #check against big radius number doesn't make sense
    #since in case of straight line it becomes very big
    #mr - my radius, otr - other radius
    def radius_in_range(self, mr, otr):
        result = False
        #radius check, reject too small radius
        if mr > 100: #200
            #check difference btw. two radius
            rd = np.abs(mr-otr)
            if rd < 60000:
                result = True
        return result

    #mp - my position, otp - other position
    def position_in_range(self, mp, otp):
        result = False
        lane_width = 3.7    #in m
        max_tolerance = 0.5 # +/- tolerance in m
        #check lane width
        if (mp + otp) < lane_width + max_tolerance and (mp + otp) > lane_width - max_tolerance:

            result = (mp > ((lane_width / 2) - max_tolerance) and \
                     mp < ((lane_width / 2) + max_tolerance))

        return result

    #use both fits to have more sanity checks possibilities
    def update_params(self, my_fit, other_fit, allx, ally, ploty):
        # sanity check of the incoming params
        my_fitx = my_fit[0]*ploty**2 + my_fit[1]*ploty + my_fit[2]
        other_fitx = other_fit[0]*ploty**2 + other_fit[1]*ploty + other_fit[2]

        mr, otr = calculate_curverad(my_fitx, other_fitx, ploty)
        mp = line_base_pos (my_fit, ploty)
        otp = line_base_pos (other_fit, ploty)

        if self.radius_in_range(mr, otr) and self.position_in_range(mp, otp):
            self.detection_miss = 0
            self.detected = True
            self.recent_xfitted = my_fitx
            self.current_fit = my_fit
            self.radius_of_curvature = mr
            self.line_base_pos = mp
            self.allx = allx
            self.ally = ally
        else: # in case a miss increase the counter
            self.detection_miss +=1
            self.reject_counter +=1
            if self.detection_miss >= 2:
                self.detection_miss=0
                self.detected = False

# draw final lane on the original undistorted image
# add radius and distance information to every image
def draw_lane(undist, warped, LL, RL, fname=None):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([LL.recent_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([RL.recent_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, False)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    #put radius and distance info on image
    color = (255,255,255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    pos_rad_text = (50,50)
    pos_dis_text = (50,150)

    #Calcuate lane radius as a mean of both lines
    l = round(LL.radius_of_curvature, 2)
    r = round(RL.radius_of_curvature, 2)
    medianr = round((l+r)/2, 2)

    #Calcuate vehicle position
    posstr, pos = calculate_position(pts_left, pts_right)

    mrad = str('Lane curvature: ' + str(medianr) + 'm')
    offc = str('Vehicle is ' + str(round(pos,2)) + 'm' + ' ' + posstr + ' ' + 'of center')

    cv2.putText(result, mrad, pos_rad_text, font, 2, color, 2, cv2.LINE_AA)
    cv2.putText(result, offc, pos_dis_text, font, 2, color, 2, cv2.LINE_AA)

    if fname!=None:
        plt.savefig(fname)

    return result

def get_calibration():
    global calibrated
    global mtx
    global dist
    if calibrated==False:
        pkl_file = open('lanesdetection/camcalibration.pkl', 'rb')
        d = pickle.load(pkl_file)
        mtx = d['mtx']
        dist = d['dist']
        pkl_file.close()
        calibrated = True
    return mtx, dist

def lane_image(img):
    global LLine
    global RLine
    mtx, dist = get_calibration()
    undist = undistort(img, mtx, dist)
    prep_img = preprocess_img(img, mtx, dist)

    if LLine.detected == False or RLine.detected == False:
        detect_start(prep_img, LLine, RLine)
    else:
        detect_next(prep_img, LLine, RLine)

    result = draw_lane(undist, prep_img, LLine,  RLine)

    return result

LLine = Line('L')
RLine = Line('R')
calibrated = False
mtx=0
dist=0
