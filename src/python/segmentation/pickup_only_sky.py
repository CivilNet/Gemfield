from PIL import Image
import numpy as np
import os
import shutil
num = 0
files = os.listdir('training-sky')
for f in files:
    img = Image.open('training-sky/'+f)
    img = np.array(img)
    if(np.sum(img)!=0):
        shutil.copyfile('training-sky/'+f,'training-only-sky/'+f)
    num += 1
    print('done:',num)
