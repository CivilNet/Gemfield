import sys

images_dir = sys.argv[1]

import os
the_list = ['']
dataset_list = {}
for d in os.listdir(images_dir):
    fd = os.path.join(images_dir,d)
    attrs = d.split('_')
    cat = attrs[1]
    video = attrs[2]

    if cat not in dataset_list:
        dataset_list[cat] = {}
    if video not in dataset_list[cat]:
        dataset_list[cat][video] = []
    
    for f in os.listdir(fd):
        file = os.path.join(d,f)
        dataset_list[cat][video].append(file)

cat_id = 0
for cat in dataset_list:
    for video in dataset_list[cat]:
        start_frame = 1
        del the_list[-1]
        for file in dataset_list[cat][video]:
            if start_frame % 16 == 1:
                item = '{} {} {}\n'.format(file, start_frame, cat_id)
                the_list.append(item)
            
            start_frame += 1
    cat_id += 1

with open('train_01.lst','w') as train_f, open('test_01.lst','w') as test_f:
    train_f.writelines(the_list)