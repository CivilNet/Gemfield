import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


def resize(img,width=400.0):
    r = float(width) / img.shape[0]
    dim = (int(img.shape[1] * r), int(width))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

video_file = sys.argv[1]

cap = cv2.VideoCapture(video_file)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = resize(frame, width=400)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    roi_hist_A = cv2.calcHist([frame2],[0,1],None,[180,256],[0,180,0,256])
    roi_hist_A = cv2.normalize(roi_hist_A, roi_hist_A, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('frame',frame)
    cv2.imshow('frame2',frame2)
    cv2.imshow('frame3',roi_hist_A)

    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
