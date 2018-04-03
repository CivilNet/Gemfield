import sys
import cv2

if __name__ == '__main__':
    camera = cv2.VideoCapture(sys.argv[1])
    #orig_op = cv2.imread('soccer_half_field.jpeg')
    #op = orig_op.copy()
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, detectShadows=False)
    flag = False