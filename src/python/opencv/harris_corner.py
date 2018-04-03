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

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    #frame[dst>0.02*dst.max()]=[0,0,255]

    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    frame[res[:,1],res[:,0]]=[0,0,255]
    frame[res[:,3],res[:,2]] = [0,255,0]



    cv2.imshow('gemfield',frame)

    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
    else:
        pass

cv2.destroyAllWindows()
cap.release()
