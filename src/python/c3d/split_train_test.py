import sys
file = sys.argv[1]
file_train = sys.argv[2]
file_test = sys.argv[3]
#/tmp/gemfield/v_PlayingViolin_g10_c02 48 53
previous = None
counter = 0
the_list = []
for i in range(101):
    the_list.append(dict())

with open(file, 'r')  as f, open(file_train, 'w') as train_f, open(file_test,'w') as test_f:
    for l in f.readlines():
        attrs = l.split()
        video,start_frame,cat = attrs
        cat = int(cat)
        if video not in the_list[cat]:
            the_list[cat][video] = []

        the_list[cat][video].append(l)
    for c in the_list:
        total = len(c)
        test_num = int(total/5)
        train_num = total - test_num
        offset = 0
        for i in c:
            offset += 1
            if offset < train_num:
                train_f.writelines(c[i])
            else:
                test_f.writelines(c[i])
