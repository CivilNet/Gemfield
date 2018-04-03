import numpy as np
import cv2
from matplotlib import pyplot as plt

import time

cap = cv2.VideoCapture('gemfield.mp4')

while 1:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30,50,50])
    upper_green = np.array([50,255,255])

    lower_white = np.array([0,50,150])
    upper_white = np.array([180,180,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    res2 = cv2.bitwise_and(frame,frame, mask= mask2)


    res1 = cv2.medianBlur(res,11)

    # edges = cv2.Canny(res1,50,150,apertureSize = 3)
    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    # try:
    #     for x1,y1,x2,y2 in lines[0]:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    # except:
    #     pass

    cv2.imshow('frame',frame)
    #cv2.imshow('mask',res)
    cv2.imshow('res',res1)
    cv2.imshow('res2',res2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
