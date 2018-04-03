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

    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray,50,150,apertureSize = 3)
    #img = cv2.medianBlur(gray,5)
    #circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=30,minRadius=30,maxRadius=40)

    mask = np.zeros(frame.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    frame = frame*mask2[:,:,np.newaxis]

    # if circles is None:
    #     continue

    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     # draw the outer circle
    #     cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    # minLineLength = 600
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    # print('================>')
    # for x1,y1,x2,y2 in lines[0]:
    #     print('aaaa')
    #     cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # lines = cv2.HoughLines(edges,1,np.pi/130,200)
    # for rho,theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))
    #     cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    #edges = cv2.Canny(frame, 100, 200)
    #cv2.imshow('gemfield',gray)

    cv2.imshow('gemfield',frame)

    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
    else:
        pass

cv2.destroyAllWindows()
cap.release()
