import sys
import numpy as np
import cv2

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)

# take first frame of the video
ret,frame = cap.read()
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


while(1):
    ret ,frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,1000,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    cv2.imshow('gemfield',frame)

    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
    else:
        pass

cv2.destroyAllWindows()
cap.release()
