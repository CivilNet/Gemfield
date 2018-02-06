#!/usr/bin/env python
import os
import logging
import optparse
from PIL import Image
import numpy as np
import caffe
import time


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))



class ImagenetClassifier(object):
    default_args = {
        'model_def_file': ('{}/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': ('{}/model.caffemodel'.format(REPO_DIRNAME)),
        'labels_cn_file': ('{}/food_cn.txt'.format(REPO_DIRNAME)),
        'labels_en_file': ('{}/food_en.txt'.format(REPO_DIRNAME))
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception("File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, model_def_file, pretrained_model_file, labels_cn_file, labels_en_file, gpu_mode):
        logging.info('Loading net and associated files...')

        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(model_def_file, pretrained_model_file, image_dims=(256, 256), 
            raw_scale=255,
            mean=np.array([98.0, 129.0, 153.0], dtype='f4'),
            #mean=None,
            channel_swap=(2, 1, 0))

        with open(labels_cn_file) as f:
            self.label_list = [l.strip() for l in f.readlines()]

    def open_oriented_im(self, im_path):
        im = Image.open(im_path)
        if hasattr(im, '_getexif'):
            exif = im._getexif()
            if exif is not None and 274 in exif:
                orientation = exif[274]
                im = apply_orientation(im, orientation)

        img = np.asarray(im).astype(np.float32) / 255.
        
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=False).flatten()
            #scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            logging.info('gemfield debug indices: %s and spent time: %s ', str(indices), '%.3f' % (endtime - starttime))
            predictions = [self.label_list[indice - 1] for indice in indices] 

            for i in range(0,5):
                print('gemfield debug each value: ' + predictions[i])

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [ (p, '%.5f' % scores[i]) for i, p in zip(indices, predictions)]

            logging.info('result: %s', str(meta))

            return meta

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the image. Maybe try another one?')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = optparse.OptionParser()
    parser.add_option('-g', '--gpu',help="use gpu mode",action='store_true', default=False)
    parser.add_option('-f', '--file', help="image path")


    opts, args = parser.parse_args()
    if not opts.file:
        parser.error('filename not given')

    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    starttime = time.time()
    clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    endtime = time.time()
    print('gemfield construct time is: %.3f\n\n' % (endtime - starttime))

    clf.net.forward()
    endtime2 = time.time()
    print('gemfield forward time is: %.3f\n\n' % (endtime2 - endtime))

    my_image = clf.open_oriented_im(opts.file)
    result = clf.classify_image(my_image)
    endtime3 = time.time()
    print('gemfield classify time is: %.3f\n\n' % (endtime3 - endtime2))
    for i in result:
        print i[0] + ' : ' + i[1]