from PIL import Image
import numpy as np
import os
files = os.listdir('training')
num = 0
for f in files:
    im = Image.open('training/'+f)
    im = np.array(im)
    im = np.where(im==3,255,0)
    im = Image.fromarray(np.uint8(im))
    im.save("training-sky/"+f)
    print('done:',num)
    num += 1
