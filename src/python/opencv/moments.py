import numpy as np
import cv2
from matplotlib import pyplot as plt

import time

cap = cv2.VideoCapture('gemfield.mp4')

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(gray,127,255,0)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0,len(contours)): 
        x, y, w, h = cv2.boundingRect(contours[i])  
        cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 5)

    cv2.imshow('frame',image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
