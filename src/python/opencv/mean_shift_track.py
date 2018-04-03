import sys
import cv2
import numpy as np

def displayImage(image):
    cv2.namedWindow('ImageDisplayer',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ImageDisplayer', 1920,1080)
    cv2.imshow('ImageDisplayer',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def cropImage(img):
    mask = np.zeros(img.shape,dtype=np.uint8)
    roi = np.array([[(1151,618),(14,1167),(40,1242),(812,1632),(1685,1887),(3088,1795),(3659,1627),(4445,1247),(4449,1084),(3230,532),(2155,468)]], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask,roi,ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def PreProcessVideoFile(initialFrame, boundingBox):
    # Gimp H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # take first frame of the video
    # r, h, c, w = 765, 85, 1265, 45
    # track_window = (c, r, w, h)
    r,h,c,w = boundingBox
    track_window = (c,r,w,h)
    roi = initialFrame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_roi, np.array((0, 0, 250)), np.array((100, 255, 255))) white
    mask = cv2.inRange(hsv_roi, np.array((0, 0, 0)), np.array((0, 0, 0)))
    cv2.imwrite('E:\Python\PlayerTracker\Images\mask.png', mask)
    cv2.imwrite('E:\Python\PlayerTracker\Images\initialframe.png', initialFrame)
    # cv2.waitKey(0)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # displayImage(roi_hist)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    return track_window, roi_hist, term_crit


def TrackImageInVideo(videoFilePath):
    cap = cv2.VideoCapture(videoFilePath)
    ret, frame = cap.read()
    if not ret:
        print("Error Reading the file")
        return

    croppedImage = cropImage(frame)
    # boundingBox = 765,85,1265,45
    boundingBox = 1087,116,1152,70
    track_window, roi_hist, term_crit = PreProcessVideoFile(croppedImage,boundingBox)

    while(1):
        ret,frame = cap.read()
        frame = cropImage(frame)
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            displayImage(img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    TrackImageInVideo(sys.argv[1])
