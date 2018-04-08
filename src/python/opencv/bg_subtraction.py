import numpy as np
import cv2
cap = cv2.VideoCapture('gemfield.mp4')
#
fgbg = cv2.createBackgroundSubtractorMOG2(history=20,detectShadows=False)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=20, detectShadows=False,varThreshold=100)

while 1:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    fgmask2 = fgbg2.apply(frame)

    cv2.imshow('frame',fgmask)
    cv2.imshow('frame2',fgmask2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

# import numpy as np
# import cv2

# cap = cv2.VideoCapture('gemfield.mp4')
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorKNN()

# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
