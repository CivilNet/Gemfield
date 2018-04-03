import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

image = sys.argv[1]

img = cv2.imread(image)

img = cv2.imread(image)
#mask = cv2.imread('mask2.png',0)

dst = cv2.inpaint(img, img[:,1,1], 3,cv2.INPAINT_TELEA)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
