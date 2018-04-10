import os
import sys
input_dirs = sys.argv[1]
the_list = ['']
dataset_list = {}
for d in os.listdir(input_dirs):
    if 'Preshot' in d or 'DFKick' in d:
        continue
    abs_d = os.path.join(input_dirs,d)
    attrs = d.split('__')
    cat = attrs[0]

    if cat not in dataset_list:
        dataset_list[cat] = {}
    if abs_d not in dataset_list[cat]:
        dataset_list[cat][abs_d] = []
    
    for img_id in range(6,54):
        abs_f = os.path.join(abs_d,'{:06d}.jpg'.format(img_id))
        if not os.path.isfile(abs_f):
            raise Exception('File not found: {}'.format(abs_f))
        dataset_list[cat][abs_d].append(img_id)

cat_id = 0
start_frame = 6
train_list = []
test_list = []
for cat in dataset_list:
    for video in dataset_list[cat]:
        item = '{} {} {}\n'.format(video, start_frame, cat_id)
        if 'lj2018baxiVSagt' in video or 'ogzntVSdtmdshang' in video or 'penalty2_' in video:
            test_list.append(item)
        else:
            train_list.append(item)
    cat_id += 1

with open('train_01.lst','w') as train_f, open('test_01.lst','w') as test_f:
    train_f.writelines(train_list)
    test_f.writelines(test_list)


