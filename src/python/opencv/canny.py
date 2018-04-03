import sys
import numpy as np
import cv2

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
# r,h,c,w - region of image simply hardcoded the values
r,h,c,w = 300,50,300,50  
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        edges = cv2.Canny(frame, 100, 200)
        cv2.imshow('img2',edges)

        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            pass
    else:
        break

cv2.destroyAllWindows()
cap.release()
