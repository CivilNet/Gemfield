import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

image = sys.argv[1]

img = cv2.imread(image)
dst = cv2.fastNlMeansDenoisingColored(img,None,4,4,7,21)
plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122),plt.imshow( cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) )
plt.show()
