import numpy as np
import timeit
from sklearn import svm
import struct

TRAIN_ITMES = 60000
TEST_ITEMS = 1000

def loadMnistData():
    mnist_data = []
    for img_file,label_file,items in zip(['gemfield_data/train-images-idx3-ubyte','gemfield_data/t10k-images-idx3-ubyte'],
                                   ['gemfield_data/train-labels-idx1-ubyte','gemfield_data/t10k-labels-idx1-ubyte'],
                                   [TRAIN_ITMES, TEST_ITEMS]):
        data_img = open(img_file, 'rb').read()
        data_label = open(label_file, 'rb').read()
        #fmt of struct unpack, > means big endian, i means integer, well, iiii mean 4 integers
        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset)
        print('magic number is {}, image number is {}, height is {} and width is {}'.format(magic_number, img_number, height, width))
        #slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        #28x28
        image_size = height * width
        #B means unsigned char
        fmt = '>{}B'.format(image_size)
        #because gemfield has insufficient memory resource
        if items > img_number:
            items = img_number
        images = np.empty((items, image_size))
        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
            #0~255 to 0~1
            images[i] = images[i]/256
            offset += struct.calcsize(fmt)

        #fmt of struct unpack, > means big endian, i means integer, well, ii mean 2 integers
        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)
        print('magic number is {} and label number is {}'.format(magic_number, label_number))
        #slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        #B means unsigned char
        fmt = '>B'
        #because gemfield has insufficient memory resource
        if items > label_number:
            items = label_number
        labels = np.empty(items)
        for i in range(items):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)
        
        mnist_data.append((images, labels.astype(int)))

    return mnist_data


def forwardWithSVM():
    start_time = timeit.default_timer()
    training_data, test_data = loadMnistData()
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    train_time = timeit.default_timer()
    print('gemfield train cost {}'.format(str(train_time - start_time) ) )
    # test
    print('Begin the test...')
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    print("%s of %s values correct." % (num_correct, len(test_data[1])))
    test_time = timeit.default_timer()
    print('gemfield test cost {}'.format(str(test_time - start_time) ) )

if __name__ == "__main__":
    forwardWithSVM()
