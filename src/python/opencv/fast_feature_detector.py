import sys
import numpy as np
import cv2

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)

# take first frame of the video
ret,frame = cap.read()
fast = cv2.FastFeatureDetector()

while(1):
    ret ,frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    



    # find and draw the keypoints
    kp = fast.detect(frame,None)
    img2 = cv2.drawKeypoints(frame, kp, color=(255,0,0))
    # Print all default params
    print "Threshold: ", fast.getInt('threshold')
    print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    print "neighborhood: ", fast.getInt('type')
    print "Total Keypoints with nonmaxSuppression: ", len(kp)
    #cv2.imwrite('fast_true.png',img2)
    # Disable nonmaxSuppression
    # fast.setBool('nonmaxSuppression',0)
    # kp = fast.detect(frame,None)
    # print "Total Keypoints without nonmaxSuppression: ", len(kp)
    # img3 = cv2.drawKeypoints(frame, kp, color=(255,0,0))

    cv2.imshow('gemfield',img2)

    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
    else:
        pass

cv2.destroyAllWindows()
cap.release()
